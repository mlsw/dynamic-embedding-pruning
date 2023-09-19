import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from types import TracebackType
from typing import Optional, Type, TypeVar

from transformers import PreTrainedModel

T = TypeVar("T")


@dataclass
class ParameterMeasurement:
    time: float
    embedding_parameters: int
    model_parameters: int


class ParameterTracker:
    def __init__(
        self, model: PreTrainedModel, output_dir: Optional[str] = None
    ) -> None:
        self._model = model
        self._output_dir = output_dir
        self._initial_measurement = None

    def __enter__(self) -> None:
        self._initial_measurement = self._create_measurement()

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        if self._initial_measurement is None:
            return

        final_measurement = self._create_measurement()
        self._write_measurement(self._initial_measurement, final_measurement)

    def _calculate_parameters(self) -> tuple[int, int]:
        input_embeddings = self._model.get_input_embeddings()
        embedding_parameters = sum(p.numel() for p in input_embeddings.parameters())
        model_parameters = sum(p.numel() for p in self._model.parameters())

        return embedding_parameters, model_parameters

    def _create_measurement(self) -> ParameterMeasurement:
        time = perf_counter()
        embedding_parameters, model_parameters = self._calculate_parameters()
        return ParameterMeasurement(
            time=time,
            embedding_parameters=embedding_parameters,
            model_parameters=model_parameters,
        )

    def _prefix_keys(self, input_dict: dict[str, T], prefix: str) -> dict[str, T]:
        return {f"{prefix}_{key}": value for key, value in input_dict.items()}

    def _write_measurement(
        self,
        initial_measurement: ParameterMeasurement,
        final_measurement: ParameterMeasurement,
    ) -> None:
        if self._output_dir is None:
            return

        initial_dict = self._prefix_keys(asdict(initial_measurement), "initial")
        final_dict = self._prefix_keys(asdict(final_measurement), "adapted")
        complete_dict = {**initial_dict, **final_dict}

        output_dir_path = Path(self._output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        with open(output_dir_path / "parameter_stats.json", "w+") as f:
            output = json.dumps(complete_dict, indent=4)
            f.write(output)
