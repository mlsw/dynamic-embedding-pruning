from dataclasses import dataclass
from io import BufferedRandom
from tempfile import TemporaryFile
from typing import Any, Optional

import torch
from torch.utils import hooks


@dataclass
class ModelContext:
    _embedding: torch.nn.Embedding
    _embedding_weight_file: BufferedRandom
    _handle_state_dict: hooks.RemovableHandle
    _handle_load_state_dict_pre: hooks.RemovableHandle
    _selected_token_ids: torch.Tensor


class EmbeddingPruner:
    def prepare(
        self,
        model: torch.nn.Module,
        embedding: torch.nn.Embedding,
        selected_token_ids: torch.Tensor,
    ) -> ModelContext:
        embedding_weight_name = self._get_name(target=embedding.weight, module=model)
        if not isinstance(embedding_weight_name, str):
            raise ValueError("The embedding is not a parameter of the supplied model.")

        # Retain a copy of the original input embedding weights.
        # TODO: Use the original model weights file directly instead.
        embedding_weight_file = TemporaryFile()
        torch.save(embedding.weight, embedding_weight_file)

        self._reduce_embedding(
            embedding=embedding,
            selected_token_ids=selected_token_ids,
        )

        handle_state_dict = self._register_state_dict_hook(
            model=model,
            selected_token_ids=selected_token_ids,
            embedding_weight_file=embedding_weight_file,
            embedding_weight_name=embedding_weight_name,
        )
        handle_load_state_dict_pre = self._register_load_state_dict_pre_hook(
            model=model,
            selected_token_ids=selected_token_ids,
            embedding_weight_name=embedding_weight_name,
        )

        return ModelContext(
            _handle_state_dict=handle_state_dict,
            _handle_load_state_dict_pre=handle_load_state_dict_pre,
            _embedding=embedding,
            _embedding_weight_file=embedding_weight_file,
            _selected_token_ids=selected_token_ids,
        )

    def restore(self, model_context: ModelContext) -> None:
        model_context._handle_state_dict.remove()
        model_context._handle_load_state_dict_pre.remove()

        model_context._embedding.weight = torch.nn.Parameter(
            self._create_restored_embedding_weight(
                selected_token_ids=model_context._selected_token_ids,
                embedding_weight_file=model_context._embedding_weight_file,
                reduced_embedding_weight=model_context._embedding.weight,
            )
        )

    def _reduce_embedding(
        self,
        embedding: torch.nn.Embedding,
        selected_token_ids: torch.Tensor,
    ) -> None:
        embedding.weight = torch.nn.Parameter(
            self._create_reduced_embedding_weight(
                embedding_weight=embedding.weight,
                selected_token_ids=selected_token_ids,
            )
        )
        embedding.num_embeddings = len(embedding.weight)

    @torch.no_grad()
    def _create_reduced_embedding_weight(
        self,
        embedding_weight: torch.Tensor,
        selected_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        reduced_embedding_weight = torch.zeros(
            (len(selected_token_ids), embedding_weight.shape[1]),
            dtype=embedding_weight.dtype,
            device=embedding_weight.device,
            requires_grad=embedding_weight.requires_grad,
        )
        selected_weights = embedding_weight[selected_token_ids]
        reduced_embedding_weight[: len(selected_token_ids)] = selected_weights
        return reduced_embedding_weight

    @torch.no_grad()
    def _create_restored_embedding_weight(
        self,
        selected_token_ids: torch.Tensor,
        embedding_weight_file: BufferedRandom,
        reduced_embedding_weight: torch.Tensor,
    ) -> torch.Tensor:
        embedding_weight_file.seek(0)
        embedding_weight = torch.load(
            embedding_weight_file,
            map_location=reduced_embedding_weight.device,
        )
        embedding_weight[selected_token_ids] = reduced_embedding_weight
        return embedding_weight

    def _register_load_state_dict_pre_hook(
        self,
        model: torch.nn.Module,
        selected_token_ids: torch.Tensor,
        embedding_weight_name: str,
    ) -> hooks.RemovableHandle:
        def _prepare_embedding_weight(
            state_dict: dict[str, Any],
            prefix: str,
            local_metadata: dict,
            strict: bool,
            missing_keys: list[str],
            unexpected_keys: list[str],
            error_msgs: list[str],
        ):
            state_dict[embedding_weight_name] = self._create_reduced_embedding_weight(
                embedding_weight=state_dict[embedding_weight_name],
                selected_token_ids=selected_token_ids,
            )

        return model._register_load_state_dict_pre_hook(_prepare_embedding_weight)

    def _register_state_dict_hook(
        self,
        model: torch.nn.Module,
        selected_token_ids: torch.Tensor,
        embedding_weight_file: BufferedRandom,
        embedding_weight_name: str,
    ) -> hooks.RemovableHandle:
        def _restore_embedding_weight(
            _: torch.nn.Module,
            state_dict: dict[str, Any],
            prefix: str,
            local_metadata: dict,
        ) -> dict[str, Any]:
            state_dict[embedding_weight_name] = self._create_restored_embedding_weight(
                selected_token_ids=selected_token_ids,
                embedding_weight_file=embedding_weight_file,
                reduced_embedding_weight=state_dict[embedding_weight_name],
            )
            return state_dict

        return model._register_state_dict_hook(_restore_embedding_weight)

    def _get_name(
        self,
        target: torch.Tensor,
        module: torch.nn.Module,
    ) -> Optional[str]:
        for name, parameter in module.named_parameters():
            if parameter is target:
                return name

        return None
