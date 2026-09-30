from __future__ import annotations

import threading
import time
from threading import Timer
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Optional, Sequence, Union

from .errors import (
    EngineStoppedError,
    ModelReloadError,
    QueueFullError,
    RequestCancelledError,
    RequestTimeoutError,
)
from .models import Model
from .types import PredictionFuture, Request, RequestState


@dataclass(frozen=True)
class ExecutorConfig:
    max_queue_size: int = 128
    max_batch_size: int = 32
    max_wait_time: float = 0.02
    default_timeout: Optional[float] = 5.0
    # Seconds of queueing that raise effective priority by one; None disables
    # aging. Aging guarantees normal requests are not starved by a sustained
    # stream of higher-priority arrivals.
    priority_aging: Optional[float] = 0.05


@dataclass
class _Entry:
    request: Request
    future: PredictionFuture
    deadline: Optional[float]
    enqueued_at: float = 0.0
    seq: int = 0


class DynamicBatchingExecutor:
    """One worker thread that groups compatible requests into batches."""

    def __init__(
        self,
        model: Model,
        config: Optional[ExecutorConfig] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._model = model
        self._config = config or ExecutorConfig()
        self._clock = clock
        self._queue: Deque[_Entry] = deque()
        self._entries: dict[str, _Entry] = {}
        self._condition = threading.Condition()
        self._running = False
        self._stopped = False
        self._worker: Optional[threading.Thread] = None
        self._batch_in_flight = 0
        self._running_entries: list[_Entry] = []
        self._pending_reload: tuple[Optional[Model], Optional[float]] = (None, None)
        self._enqueue_seq = 0
        self._batch_seq = 0

    @property
    def model(self) -> Model:
        with self._condition:
            return self._model

    @property
    def queue_size(self) -> int:
        with self._condition:
            return len(self._queue)

    def start(self) -> "DynamicBatchingExecutor":
        with self._condition:
            if self._running or self._stopped:
                raise EngineStoppedError("executor cannot be restarted")
            self._running = True
            self._worker = threading.Thread(
                target=self._run, name="inferflow-worker", daemon=True
            )
            self._worker.start()
        return self

    def submit(self, request: Request, timeout: Optional[float] = -1) -> PredictionFuture:
        if timeout == -1:
            timeout = self._config.default_timeout
        future = PredictionFuture(request.request_id)
        deadline = None if timeout is None else self._clock() + timeout
        with self._condition:
            if self._stopped or not self._running:
                raise EngineStoppedError("executor is not running")
            self._expire_requests_locked()
            if len(self._queue) >= self._config.max_queue_size:
                raise QueueFullError(
                    f"inference queue is full: {len(self._queue)}/{self._config.max_queue_size}"
                )
            entry = _Entry(request=request, future=future, deadline=deadline)
            entry.enqueued_at = self._clock()
            entry.seq = self._enqueue_seq
            self._enqueue_seq += 1
            self._queue.append(entry)
            self._entries[request.request_id] = entry
            self._condition.notify_all()
        return future

    def cancel(self, request_id: str) -> bool:
        """Cancel a queued/running request without changing other result order."""
        with self._condition:
            entry = self._entries.pop(request_id, None)
            if entry is not None:
                try:
                    self._queue.remove(entry)
                except ValueError:
                    pass
        if entry is None:
            return False
        return entry.future.cancel(
            RequestCancelledError("request cancelled before result delivery")
        )

    def reload_model(
        self,
        loader: Union[Callable[[], Model], Model],
        timeout: Optional[float] = 5.0,
    ) -> Model:
        try:
            replacement = loader() if callable(loader) else loader
            _validate_model(replacement)
        except ModelReloadError:
            raise
        except Exception as exc:
            raise ModelReloadError(f"failed to load model: {exc}") from exc

        deadline = None if timeout is None else self._clock() + timeout
        with self._condition:
            if self._stopped or not self._running:
                raise EngineStoppedError("executor is not running")
            self._pending_reload = (replacement, deadline)
            self._condition.notify_all()

        while True:
            with self._condition:
                current = self._model
                pending = self._pending_reload[0]
            if current is replacement:
                return replacement
            if pending is not replacement:
                raise ModelReloadError("model reload expired before in-flight batches finished")
            if deadline is not None and self._clock() >= deadline:
                raise ModelReloadError("timed out waiting for in-flight batches")
            time.sleep(0.001)

    def stop(self, timeout: Optional[float] = 5.0, reject_queued: bool = True) -> None:
        with self._condition:
            if not self._running:
                self._stopped = True
                return
            self._running = False
            self._stopped = True
            worker = self._worker
            self._condition.notify_all()
        if worker is not None:
            worker.join(timeout)
        # Whatever is still unfinished after the worker exits must not wait
        # forever: fail it with an explicit shutdown reason.
        self._fail_remaining(EngineStoppedError("executor stopped before request ran"))

    def __enter__(self) -> "DynamicBatchingExecutor":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def _run(self) -> None:
        while True:
            batch: list[_Entry] = []
            model: Optional[Model] = None
            with self._condition:
                batch_deadline: Optional[float] = None
                while True:
                    if not self._running:
                        self._drain_queue_locked(
                            EngineStoppedError("executor stopped while requests were queued")
                        )
                        return
                    self._expire_requests_locked()
                    self._apply_reload_locked()
                    if self._queue:
                        now = self._clock()
                        anchor = self._select_anchor_locked(now)
                        compatible = [
                            entry
                            for entry in self._queue
                            if entry.request.batching_key == anchor.request.batching_key
                        ]
                        ready = len(compatible) >= self._config.max_batch_size
                        waited_enough = (
                            batch_deadline is not None
                            and self._clock() >= batch_deadline
                        )
                        if ready or waited_enough:
                            model = self._model
                            batch = self._take_batch_locked(anchor.request.batching_key, now)
                            self._batch_in_flight += 1
                            self._running_entries.extend(batch)
                            break
                        if batch_deadline is None:
                            batch_deadline = self._clock() + self._config.max_wait_time
                    wait_for = self._config.max_wait_time
                    if self._queue and batch_deadline is not None:
                        wait_for = min(wait_for, max(0.0, batch_deadline - self._clock()))
                    self._condition.wait(wait_for)

            self._execute(batch, model)
            with self._condition:
                self._batch_in_flight -= 1
                for entry in batch:
                    if entry in self._running_entries:
                        self._running_entries.remove(entry)
                for entry in batch:
                    self._entries.pop(entry.request.request_id, None)
                self._apply_reload_locked()

    def _effective_priority(self, entry: _Entry, now: float) -> float:
        aging = self._config.priority_aging
        if not aging or aging <= 0:
            return float(entry.request.priority)
        aged = max(0.0, now - entry.enqueued_at) / aging
        return entry.request.priority + aged

    def _select_anchor_locked(self, now: float) -> _Entry:
        """Highest effective priority wins; ties break by arrival order."""
        return min(
            self._queue,
            key=lambda entry: (-self._effective_priority(entry, now), entry.seq),
        )

    def _take_batch_locked(
        self, key: tuple[type, tuple[int, ...]], now: float
    ) -> list[_Entry]:
        compatible = [
            entry for entry in self._queue if entry.request.batching_key == key
        ]
        compatible.sort(key=lambda entry: (-self._effective_priority(entry, now), entry.seq))
        selected = compatible[: self._config.max_batch_size]
        chosen = set(id(entry) for entry in selected)
        self._queue = deque(
            entry for entry in self._queue if id(entry) not in chosen
        )
        self._batch_seq += 1
        batch_id = f"batch-{self._batch_seq}"
        for entry in selected:
            entry.future.batch_id = batch_id
            entry.future.model_version = self._model.version
        return selected

    def _drain_queue_locked(self, exc: BaseException) -> None:
        drained = list(self._queue)
        self._queue.clear()
        for entry in drained:
            self._entries.pop(entry.request.request_id, None)
            entry.future.set_exception(exc, RequestState.FAILED)

    def _fail_remaining(self, exc: BaseException) -> None:
        with self._condition:
            self._drain_queue_locked(exc)
            running = list(self._running_entries)
        for entry in running:
            entry.future.set_exception(exc, RequestState.FAILED)

    def _execute(self, batch: Sequence[_Entry], model: Model) -> None:
        active: list[_Entry] = []
        for entry in batch:
            if entry.future.state == RequestState.CANCELLED:
                continue
            try:
                model.check(entry.request.shape, entry.request.dtype)
            except Exception as exc:
                entry.future.set_exception(exc, RequestState.FAILED)
                continue
            entry.future.mark_running()
            active.append(entry)

        active = self._fail_timed_out(active, "request timed out before execution")
        if not active:
            return

        timers = self._start_timeout_timers(active)
        samples = [entry.request.data for entry in active]
        try:
            outputs = model.predict(samples)
        except Exception as exc:
            for timer in timers:
                timer.cancel()
            for entry in active:
                entry.future.set_exception(exc, RequestState.FAILED)
            return

        if not isinstance(outputs, Sequence) or len(outputs) != len(active):
            actual = len(outputs) if hasattr(outputs, "__len__") else "non-sequence"
            error = RuntimeError(
                f"model returned {actual} outputs for {len(active)} requests"
            )
            for timer in timers:
                timer.cancel()
            for entry in active:
                entry.future.set_exception(error, RequestState.FAILED)
            return

        for entry, output in zip(active, outputs):
            entry.future.set_result(output)
        for timer in timers:
            timer.cancel()

    def _start_timeout_timers(self, entries: list[_Entry]) -> list[Timer]:
        timers: list[Timer] = []
        now = self._clock()
        for entry in entries:
            if entry.deadline is None:
                continue
            delay = entry.deadline - now
            if delay <= 0:
                entry.future.set_exception(
                    RequestTimeoutError("request timed out before execution"),
                    RequestState.TIMED_OUT,
                )
                continue

            def fail_on_timeout(target=entry):
                target.future.set_exception(
                    RequestTimeoutError("request timed out during execution"),
                    RequestState.TIMED_OUT,
                )

            timer = Timer(delay, fail_on_timeout)
            timer.daemon = True
            timer.start()
            timers.append(timer)
        return timers

    def _fail_timed_out(self, entries: list[_Entry], message: str) -> list[_Entry]:
        now = self._clock()
        alive: list[_Entry] = []
        for entry in entries:
            if entry.deadline is not None and entry.deadline <= now:
                entry.future.set_exception(
                    RequestTimeoutError(message), RequestState.TIMED_OUT
                )
            else:
                alive.append(entry)
        return alive

    def _expire_requests_locked(self) -> None:
        now = self._clock()
        expired: list[_Entry] = []
        remaining: Deque[_Entry] = deque()
        for entry in self._queue:
            if entry.deadline is not None and entry.deadline <= now:
                expired.append(entry)
            else:
                remaining.append(entry)
        self._queue = remaining
        for entry in expired:
            self._entries.pop(entry.request.request_id, None)
            entry.future.set_exception(
                RequestTimeoutError("request timed out while queued"),
                RequestState.TIMED_OUT,
            )

    def _apply_reload_locked(self) -> None:
        replacement, deadline = self._pending_reload
        if replacement is None or self._batch_in_flight:
            return
        if deadline is not None and self._clock() >= deadline:
            self._pending_reload = (None, None)
            return
        self._model = replacement
        self._pending_reload = (None, None)


def _validate_model(model: Any) -> None:
    if not hasattr(model, "predict") or not callable(model.predict):
        raise ModelReloadError("replacement model lacks a callable predict method")
    if not hasattr(model, "check") or not callable(model.check):
        raise ModelReloadError("replacement model lacks a callable check method")
    if not hasattr(model, "version"):
        raise ModelReloadError("replacement model lacks a version attribute")
