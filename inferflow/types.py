from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class RequestState(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    FAILED = "failed"


@dataclass(frozen=True)
class Request:
    """A single inference invocation.

    ``data`` is one unpadded sample. Its shape is a tuple of positive integers;
    a one-dimensional sample of three floats uses ``shape=(3,)``.
    """

    data: tuple[Any, ...]
    shape: tuple[int, ...]
    dtype: type = float
    priority: int = 0
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    @property
    def batching_key(self) -> tuple[type, tuple[int, ...]]:
        return self.dtype, self.shape


class PredictionFuture:
    """Result handle tied to exactly one :class:`Request`."""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.batch_id: Optional[str] = None
        self.model_version: Optional[str] = None
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._state = RequestState.QUEUED
        self._result: Any = None
        self._exception: Optional[BaseException] = None

    @property
    def state(self) -> RequestState:
        with self._lock:
            return self._state

    def done(self) -> bool:
        return self._event.is_set()

    def result(self, timeout: Optional[float] = None) -> Any:
        if not self._event.wait(timeout):
            raise TimeoutError(f"request {self.request_id} result not ready")
        with self._lock:
            if self._exception is not None:
                raise self._exception
            return self._result

    def set_result(self, value: Any) -> bool:
        return self._complete(RequestState.FINISHED, result=value)

    def set_exception(self, exc: BaseException, state: RequestState) -> bool:
        return self._complete(state, exception=exc)

    def cancel(self, exc: Optional[BaseException] = None) -> bool:
        from .errors import RequestCancelledError

        return self._complete(
            RequestState.CANCELLED,
            exception=exc or RequestCancelledError("request cancelled"),
        )

    def mark_running(self) -> None:
        with self._lock:
            if self._state == RequestState.QUEUED:
                self._state = RequestState.RUNNING

    def _complete(
        self,
        state: RequestState,
        result: Any = None,
        exception: Optional[BaseException] = None,
    ) -> bool:
        with self._lock:
            if self._event.is_set():
                return False
            self._state = state
            self._result = result
            self._exception = exception
            self._event.set()
            return True

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"PredictionFuture(request_id={self.request_id!r}, state={self._state.value})"
