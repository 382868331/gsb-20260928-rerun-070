from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

from .errors import IncompatibleInputError


@runtime_checkable
class Model(Protocol):
    name: str
    version: str

    def check(self, request_shape: tuple[int, ...], dtype: type) -> None: ...

    def predict(self, samples: Sequence[Any]) -> list[Any]: ...


@dataclass
class BuiltInModel:
    """Common validation for the tiny built-in, directly computed models."""

    name: str
    version: str
    input_shape: tuple[int, ...]
    dtype: type = float

    def check(self, request_shape: tuple[int, ...], dtype: type) -> None:
        if dtype is not self.dtype:
            raise IncompatibleInputError(
                f"{self.name} {self.version} expects dtype {self.dtype}, got {dtype}"
            )
        if tuple(request_shape) != self.input_shape:
            raise IncompatibleInputError(
                f"{self.name} {self.version} expects shape {self.input_shape}, "
                f"got {tuple(request_shape)}"
            )

    def predict_one(self, sample: Any) -> Any:
        raise NotImplementedError

    def predict(self, samples: Sequence[Any]) -> list[Any]:
        return [self.predict_one(sample) for sample in samples]


class AddConstantModel(BuiltInModel):
    """Adds a deterministic constant to every scalar/vector element."""

    def __init__(
        self,
        input_shape: tuple[int, ...] = (3,),
        bias: float = 1.0,
        version: str = "add-v1",
        name: str = "builtin-add",
    ):
        super().__init__(name=name, version=version, input_shape=input_shape)
        self.bias = bias

    def predict_one(self, sample: Any) -> tuple[Any, ...]:
        return tuple(value + self.bias for value in sample)


class LinearModel(BuiltInModel):
    """Tiny deterministic fully-connected layer evaluated in pure Python.

    No network access or large weight file is involved: weights are generated
    from a small integer seed. The model is deliberately small but performs a
    real matrix multiplication so batched outputs can be checked against
    serial execution.
    """

    def __init__(
        self,
        in_dim: int = 4,
        out_dim: int = 2,
        seed: int = 7,
        version: str = "linear-v1",
        name: str = "builtin-linear",
    ):
        super().__init__(name=name, version=version, input_shape=(in_dim,))
        self.in_dim = in_dim
        self.out_dim = out_dim
        generator = _DeterministicSequence(seed)
        self.weights = tuple(
            tuple((generator.next_float() - 0.5) * 0.2 for _ in range(in_dim))
            for _ in range(out_dim)
        )
        self.bias = tuple((generator.next_float() - 0.5) * 0.1 for _ in range(out_dim))

    def predict_one(self, sample: Any) -> tuple[float, ...]:
        return tuple(
            sum(weight * value for weight, value in zip(row, sample)) + offset
            for row, offset in zip(self.weights, self.bias)
        )


class _DeterministicSequence:
    """Small deterministic LCG so built-in weights need no third-party package."""

    def __init__(self, seed: int):
        self._state = seed or 1

    def next_float(self) -> float:
        self._state = (self._state * 1103515245 + 12345) & 0x7FFFFFFF
        return (self._state % 10000) / 10000.0
