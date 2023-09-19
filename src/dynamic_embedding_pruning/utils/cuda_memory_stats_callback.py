import json
from pathlib import Path

import torch
from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


class CUDAMemoryStatsMixin:
    def _write_stats(self, args: TrainingArguments, suffix: str) -> None:
        stats = torch.cuda.memory_stats()
        output_path = Path(args.output_dir) / f"cuda_memory_stats_{suffix}.json"
        with open(output_path, "w+") as f:
            json.dump(stats, f, indent=4)

    def _reset_stats(self) -> None:
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.reset_peak_memory_stats()


class CUDAMemoryStatsTrainCallback(TrainerCallback, CUDAMemoryStatsMixin):
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if not torch.cuda.is_available():
            return

        self._reset_stats()

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if not torch.cuda.is_available():
            return

        self._write_stats(args, "train")
        self._reset_stats()


class CUDAMemoryStatsEvalCallback(TrainerCallback, CUDAMemoryStatsMixin):
    def on_init_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if not torch.cuda.is_available():
            return

        self._reset_stats()

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if not torch.cuda.is_available():
            return

        self._write_stats(args, "eval")
        self._reset_stats()
