from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum, IntEnum
from typing import Any, Optional


class RequestState(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    FAILED = "failed"


@dataclass(frozen=True)
class ExecutionInfo:
    """Immutable snapshot of a request's execution attribution.

    ``batch_id``/``model_name``/``model_version`` are bound exactly once, when
    the worker admits the request into a batch. Requests still queued are not
    bound; terminal snapshots preserve whatever attribution existed.
    """

    state: RequestState
    batch_id: Optional[int] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    batch_index: Optional[int] = None

    @property
    def bound(self) -> bool:
        return self.batch_id is not None


class Priority(IntEnum):
    """Scheduling priority; larger values are served first."""

    LOW = 0
    NORMAL = 1
    HIGH = 2


@dataclass(frozen=True)
class Request:
    """A single inference invocation.

    ``data`` is one unpadded sample. Its shape is a tuple of positive integers;
    a one-dimensional sample of three floats uses ``shape=(3,)``.
    """

    data: tuple[Any, ...]
    shape: tuple[int, ...]
    dtype: type = float
    priority: Priority = Priority.NORMAL
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    @property
    def batching_key(self) -> tuple[type, tuple[int, ...]]:
        return self.dtype, self.shape


class PredictionFuture:
    """Result handle tied to exactly one :class:`Request`."""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._info = ExecutionInfo(state=RequestState.QUEUED)
        self._history: list[ExecutionInfo] = [self._info]
        self._result: Any = None
        self._exception: Optional[BaseException] = None

    @property
    def state(self) -> RequestState:
        with self._lock:
            return self._info.state

    @property
    def info(self) -> ExecutionInfo:
        """Stable attribution snapshot; never changes after a terminal state."""
        with self._lock:
            return self._info

    @property
    def history(self) -> tuple[ExecutionInfo, ...]:
        """Queued -> bound -> terminal attribution snapshots, in order."""
        with self._lock:
            return tuple(self._history)

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

    def bind_batch(
        self,
        batch_id: int,
        model_name: str,
        model_version: str,
        batch_index: int,
    ) -> bool:
        """Bind the request to the batch that will execute it.

        Happens atomically with dequeue under the executor lock, so a request
        is either cancelled while still queued (unbound) or admitted exactly
        once (bound).
        """
        with self._lock:
            if self._event.is_set() or self._info.state != RequestState.QUEUED:
                return False
            self._info = ExecutionInfo(
                state=RequestState.RUNNING,
                batch_id=batch_id,
                model_name=model_name,
                model_version=model_version,
                batch_index=batch_index,
            )
            self._history.append(self._info)
            return True

    def _complete(
        self,
        state: RequestState,
        result: Any = None,
        exception: Optional[BaseException] = None,
    ) -> bool:
        with self._lock:
            if self._event.is_set():
                return False
            self._info = replace(self._info, state=state)
            self._history.append(self._info)
            self._result = result
            self._exception = exception
            self._event.set()
            return True

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"PredictionFuture(request_id={self.request_id!r}, "
            f"state={self._info.state.value})"
        )
