from functools import partial
from typing import Optional

import numpy as np
from datasets import Dataset, DatasetDict
from numpy.typing import DTypeLike

from . import constants


class TokenCounter:
    def __init__(
        self,
        batch_size: int = constants.DEFAULT_BATCH_SIZE,
        num_proc: Optional[int] = None,
        load_from_cache_file: Optional[bool] = None,
        dtype: DTypeLike = np.int64,
    ) -> None:
        self._batch_size = batch_size
        self._num_proc = num_proc
        self._load_from_cache_file = load_from_cache_file
        self._dtype = dtype

    def count(
        self,
        dataset: Dataset | DatasetDict,
        vocabulary_size: int,
        columns: list[str],
    ) -> np.ndarray:
        if isinstance(dataset, DatasetDict):
            return sum(
                (
                    self._token_frequencies(d, columns, vocabulary_size)
                    for d in dataset.values()
                ),
                start=np.zeros(vocabulary_size, dtype=self._dtype),
            )
        elif isinstance(dataset, Dataset):
            return self._token_frequencies(dataset, columns, vocabulary_size)
        else:
            raise ValueError("`dataset` must be a `Dataset` or `DatasetDict`")

    def _token_batch_frequencies(
        self,
        examples: dict[str, np.ndarray],
        columns: list[str],
        vocabulary_size: int,
    ) -> dict[str, list[np.ndarray]]:
        token_counts = sum(
            (
                np.bincount(examples[column].reshape(-1), minlength=vocabulary_size)
                for column in columns
            ),
            start=np.zeros(vocabulary_size, dtype=self._dtype),
        )
        return {"token_frequencies": [token_counts]}

    def _token_frequencies(
        self,
        dataset: Dataset,
        columns: list[str],
        vocabulary_size: int,
    ) -> np.ndarray:
        dataset.set_format("numpy")
        dataset = dataset.map(
            partial(
                self._token_batch_frequencies,
                columns=columns,
                vocabulary_size=vocabulary_size,
            ),
            batched=True,
            batch_size=self._batch_size,
            remove_columns=dataset.column_names,
            load_from_cache_file=self._load_from_cache_file,
            num_proc=self._num_proc,
            desc="Counting tokens",
        )

        return sum(
            (example["token_frequencies"] for example in dataset),  # type: ignore
            start=np.zeros(vocabulary_size, dtype=self._dtype),
        )
