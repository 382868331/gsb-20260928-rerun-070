from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Optional


class RequestState(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    FAILED = "failed"


class Priority(IntEnum):
    """Scheduling priority; larger values are served first."""

    LOW = 0
    NORMAL = 1
    HIGH = 2


class VersionContract(Enum):
    """Model-version delivery agreement for a request group."""

    SAME_VERSION = "same_version"
    CURRENT_VERSION = "current_version"


class HedgeDelivery(Enum):
    """When a hedged request item may be delivered."""

    FIRST_CONFORMING = "first_conforming"
    CONSISTENT = "consistent"


class AllocationStrategy(Enum):
    """How the group spends its dynamic redundant-execution budget."""

    FAIR = "fair"
    DEADLINE_FIRST = "deadline_first"


class CandidateState(Enum):
    """Lifecycle of one redundant execution within a request item."""

    PENDING = "pending"
    COMMITTED = "committed"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GroupState(Enum):
    WAITING = "waiting"
    EXECUTED = "executed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ItemState(Enum):
    WAITING = "waiting"
    EXECUTED = "executed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class BatchAttribution:
    """Execution ownership of a request once it enters a batch.

    Immutable and assigned exactly once, so repeated reads never observe
    contradictory batch identities or model versions. batch_index is the
    request's position within the formed batch.
    """

    batch_id: str
    model_name: str
    model_version: str
    batch_size: int
    batch_index: int


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
    required_model_version: Optional[str] = None
    # Seconds after group submission by which the caller needs delivery;
    # used only by hedged groups to steer budget allocation.
    deadline: Optional[float] = None
    # Maximum number of redundant executions this item may receive.
    max_redundancy: Optional[int] = None
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
        self._state = RequestState.QUEUED
        self._result: Any = None
        self._exception: Optional[BaseException] = None
        self._attribution: Optional[BatchAttribution] = None
        self._history: list[RequestState] = [RequestState.QUEUED]
        self._done_callbacks: list = []

    def add_done_callback(self, callback) -> None:
        """Invoke ``callback(future)`` exactly once when the future settles."""
        with self._lock:
            if not self._event.is_set():
                self._done_callbacks.append(callback)
                return
        try:
            callback(self)
        except Exception:
            pass

    @property
    def state(self) -> RequestState:
        with self._lock:
            return self._state

    @property
    def attribution(self) -> Optional[BatchAttribution]:
        """Batch/model ownership; None while the request never entered a batch."""
        with self._lock:
            return self._attribution

    @property
    def history(self) -> tuple[RequestState, ...]:
        """Ordered state transitions, e.g. QUEUED -> RUNNING -> FINISHED."""
        with self._lock:
            return tuple(self._history)

    def set_attribution(self, attribution: BatchAttribution) -> bool:
        """Bind execution ownership once; later attempts are ignored."""
        with self._lock:
            if self._attribution is not None:
                return False
            self._attribution = attribution
            return True

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
                self._history.append(RequestState.RUNNING)

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
            self._history.append(state)
            self._event.set()
            callbacks = list(self._done_callbacks)
            self._done_callbacks.clear()
        for callback in callbacks:
            try:
                callback(self)
            except Exception:
                pass
        return True

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"PredictionFuture(request_id={self.request_id!r}, state={self._state.value})"
