from functools import partial
from typing import Optional, TypeVar

import numpy as np
from datasets import Dataset, DatasetDict

from . import constants

DatasetT = TypeVar("DatasetT", bound=Dataset | DatasetDict)


class TokenConverter:
    def __init__(
        self,
        batch_size: int = constants.DEFAULT_BATCH_SIZE,
        num_proc: Optional[int] = None,
        load_from_cache_file: Optional[bool] = None,
    ) -> None:
        self._batch_size = batch_size
        self._num_proc = num_proc
        self._load_from_cache_file = load_from_cache_file

    def convert(
        self,
        dataset: DatasetT,
        columns: list[str],
        token_map: np.ndarray,
    ) -> DatasetT:
        dataset.set_format("numpy")
        dataset = dataset.map(
            partial(
                self._token_batch_map,
                columns=columns,
                token_map=token_map,
            ),
            batched=True,
            batch_size=self._batch_size,
            remove_columns=columns,
            load_from_cache_file=self._load_from_cache_file,
            num_proc=self._num_proc,
            desc="Converting tokens",
        )
        dataset.reset_format()
        return dataset

    def _token_batch_map(
        self,
        examples: dict[str, np.ndarray],
        columns: list[str],
        token_map: np.ndarray,
    ) -> dict[str, np.ndarray]:
        for column in columns:
            examples[column] = token_map[examples[column]]
        return examples
