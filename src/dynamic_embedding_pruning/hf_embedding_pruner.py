from typing import Optional

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .data import constants
from .data.token_converter import DatasetT, TokenConverter
from .data.token_counter import TokenCounter
from .embedding_pruner import EmbeddingPruner
from .utils.parameter_tracker import ParameterTracker


class HFEmbeddingPruner:
    def __init__(
        self,
        model: PreTrainedModel,
        token_converter: TokenConverter = TokenConverter(),
        token_counter: TokenCounter = TokenCounter(),
        embedding_optimizer: EmbeddingPruner = EmbeddingPruner(),
        output_dir: Optional[str] = None,
    ) -> None:
        self._model = model
        self._token_converter = token_converter
        self._token_counter = token_counter
        self._embedding_optimizer = embedding_optimizer
        self._output_dir = output_dir
        self._model_context = None

    @staticmethod
    def create_parallelized(
        model: PreTrainedModel,
        embedding_optimizer: EmbeddingPruner = EmbeddingPruner(),
        output_dir: Optional[str] = None,
        batch_size: int = constants.DEFAULT_BATCH_SIZE,
        num_proc: Optional[int] = None,
        load_from_cache_file: Optional[bool] = None,
    ) -> "HFEmbeddingPruner":
        token_converter = TokenConverter(
            batch_size=batch_size,
            num_proc=num_proc,
            load_from_cache_file=load_from_cache_file,
        )
        token_counter = TokenCounter(
            batch_size=batch_size,
            num_proc=num_proc,
            load_from_cache_file=load_from_cache_file,
        )
        return HFEmbeddingPruner(
            model=model,
            token_converter=token_converter,
            token_counter=token_counter,
            embedding_optimizer=embedding_optimizer,
            output_dir=output_dir,
        )

    def prepare_model(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset: DatasetT,
    ) -> tuple[DatasetT, np.ndarray]:
        if self._model_context is not None:
            raise RuntimeError(
                "`restore` must be called before calling `prepare` again."
            )

        with ParameterTracker(model=self._model, output_dir=self._output_dir):
            token_counts = self._token_counter.count(
                dataset=dataset,
                vocabulary_size=len(tokenizer),
                columns=[self._model.main_input_name],
            )
            selected_token_ids = np.flatnonzero(token_counts)

            token_map = np.full_like(token_counts, constants.UNUSED_TOKEN_ID)
            token_map[selected_token_ids] = np.arange(len(selected_token_ids))
            dataset = self._token_converter.convert(
                dataset=dataset,
                token_map=token_map,
                columns=[self._model.main_input_name],
            )

            pad_token_id = tokenizer.pad_token_id
            if pad_token_id is not None and token_map[pad_token_id] != pad_token_id:
                raise ValueError("Reassigning `pad_token` is not yet supported.")

            embedding = self._model.get_input_embeddings()
            if not isinstance(embedding, torch.nn.Embedding):
                raise ValueError(
                    "`get_input_embeddings` must return an instance of `torch.nn.Embedding`."
                )

            self._model_context = self._embedding_optimizer.prepare(
                model=self._model,
                embedding=embedding,
                selected_token_ids=torch.from_numpy(selected_token_ids),
            )

            return dataset, token_map

    def restore_model(self) -> None:
        if self._model_context is None:
            raise ValueError("`prepare` must be called before `restore`.")

        self._embedding_optimizer.restore(model_context=self._model_context)
        self._model_context = None
