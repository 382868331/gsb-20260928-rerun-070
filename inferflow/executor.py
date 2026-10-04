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
    VersionContractError,
)
from .models import Model
from .types import BatchAttribution, PredictionFuture, Request, RequestState


@dataclass(frozen=True)
class ExecutorConfig:
    max_queue_size: int = 128
    max_batch_size: int = 32
    max_wait_time: float = 0.02
    default_timeout: Optional[float] = 5.0
    # Number of concurrently executing batch slots. Redundant (hedged)
    # executions need more than one slot so a slow primary batch does not
    # prevent its hedged copy from starting. The default of one preserves the
    # original single-worker ordering.
    worker_count: int = 1
    # Seconds of queue wait that raise a request's effective priority by one
    # level. Lower-priority requests therefore cannot starve under a
    # continuous stream of higher-priority traffic.
    aging_interval: float = 0.5


@dataclass
class _Entry:
    request: Request
    future: PredictionFuture
    deadline: Optional[float]
    enqueued_at: float = 0.0


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
        self._batch_sequence = 0
        self._workers: list[threading.Thread] = []

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
            for index in range(max(1, self._config.worker_count)):
                worker = threading.Thread(
                    target=self._run, name=f"inferflow-worker-{index}", daemon=True
                )
                worker.start()
                self._workers.append(worker)
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
            entry = _Entry(
                request=request,
                future=future,
                deadline=deadline,
                enqueued_at=self._clock(),
            )
            self._queue.append(entry)
            self._entries[request.request_id] = entry
            self._condition.notify_all()
        return future

    def submit_group(
        self,
        requests: Sequence[Request],
        same_version: bool = True,
        timeout: Optional[float] = -1,
        hedge=None,
    ):
        from .groups import RequestGroup
        from .types import VersionContract

        with self._condition:
            if self._stopped or not self._running:
                raise EngineStoppedError("executor is not running")
        group_requests = list(requests)
        with self._condition:
            pinned_version = self._model.version
        required_version = pinned_version if same_version else None
        prepared: list[Request] = []
        for request in group_requests:
            if same_version and request.required_model_version not in (None, required_version):
                raise VersionContractError(
                    f"request requires {request.required_model_version!r}; group pins {required_version!r}"
                )
            prepared.append(
                Request(
                    data=request.data,
                    shape=request.shape,
                    dtype=request.dtype,
                    priority=request.priority,
                    required_model_version=required_version,
                    request_id=request.request_id,
                )
            )
        return RequestGroup(
            self,
            prepared,
            VersionContract.SAME_VERSION if same_version else VersionContract.CURRENT_VERSION,
            pinned_version=pinned_version if same_version else None,
            submit_timeout=timeout,
            hedge=hedge,
        )

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
            queued = list(self._queue) if reject_queued else []
            self._condition.notify_all()
        for worker in self._workers:
            worker.join(timeout)
        for entry in queued:
            entry.future.set_exception(
                EngineStoppedError("executor stopped before request ran"),
                RequestState.FAILED,
            )
        # Any request still unresolved after the worker exits (for example a
        # batch stuck in a slow model past the join timeout) must reach a
        # terminal state instead of waiting forever.
        with self._condition:
            leftovers = list(self._entries.values())
            self._entries.clear()
            self._queue.clear()
        for entry in leftovers:
            entry.future.set_exception(
                EngineStoppedError("executor stopped before request completed"),
                RequestState.FAILED,
            )

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
                        return
                    self._expire_requests_locked()
                    self._apply_reload_locked()
                    executable = [
                        entry
                        for entry in self._queue
                        if entry.request.required_model_version is None
                        or entry.request.required_model_version == self._model.version
                    ]
                    if executable:
                        anchor = min(executable, key=self._scheduling_key_locked)
                        compatible = [
                            entry
                            for entry in executable
                            if entry.request.batching_key == anchor.request.batching_key
                        ]
                        ready = len(compatible) >= self._config.max_batch_size
                        waited_enough = (
                            batch_deadline is not None
                            and self._clock() >= batch_deadline
                        )
                        # Each formed batch is removed from the queue, so
                        # concurrent workers never execute the same entries.
                        if ready or waited_enough:
                            model = self._model
                            batch = self._take_batch_locked(anchor.request.batching_key)
                            self._batch_in_flight += 1
                            self._running_entries.extend(batch)
                            break
                        if batch_deadline is None:
                            batch_deadline = self._clock() + self._config.max_wait_time
                    wait_for = self._config.max_wait_time
                    if self._config.worker_count > 1:
                        wait_for = min(wait_for, 0.005)
                    if executable and batch_deadline is not None:
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
                if not self._batch_in_flight:
                    self._apply_reload_locked()
                self._condition.notify_all()

    def _take_batch_locked(self, key: tuple[type, tuple[int, ...]]) -> list[_Entry]:
        compatible = [
            entry
            for entry in self._queue
            if entry.request.batching_key == key
        ]
        compatible.sort(key=self._scheduling_key_locked)
        selected = compatible[: self._config.max_batch_size]
        selected_ids = {id(entry) for entry in selected}
        self._queue = deque(
            entry for entry in self._queue if id(entry) not in selected_ids
        )
        return selected

    def _select_anchor_locked(self) -> _Entry:
        return min(self._queue, key=self._scheduling_key_locked)

    def _scheduling_key_locked(self, entry: _Entry) -> tuple[int, float]:
        # Higher effective priority first; ties broken by arrival order so
        # same-priority requests stay FIFO and aged requests eventually win.
        return (-self._effective_priority_locked(entry), entry.enqueued_at)

    def _effective_priority_locked(self, entry: _Entry) -> int:
        from .types import Priority

        priority = int(entry.request.priority)
        interval = self._config.aging_interval
        if interval is None or interval <= 0:
            return priority
        age = self._clock() - entry.enqueued_at
        boost = int(age // interval)
        return min(int(Priority.HIGH), priority + boost)

    def _execute(self, batch: Sequence[_Entry], model: Model) -> None:
        active: list[_Entry] = []
        with self._condition:
            self._batch_sequence += 1
            sequence = self._batch_sequence
        batch_id = f"batch-{sequence}"
        model_name = getattr(model, "name", type(model).__name__)
        for batch_index, entry in enumerate(batch):
            if entry.future.state == RequestState.CANCELLED:
                continue
            entry.future.set_attribution(
                BatchAttribution(
                    batch_id=batch_id,
                    model_name=model_name,
                    model_version=model.version,
                    batch_size=len(batch),
                    batch_index=batch_index,
                )
            )
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
        if replacement is None:
            return
        # Batches capture their model object at formation time, so an in-flight
        # batch keeps executing the old model even when the active model is
        # swapped. With a single batch slot we still wait for the batch to
        # finish (the original semantics); with concurrent slots we swap once
        # any batch using the old model is already in flight, which is what
        # makes redundant execution across a version boundary possible.
        if self._batch_in_flight and self._config.worker_count <= 1:
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
