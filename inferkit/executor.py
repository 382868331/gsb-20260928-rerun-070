"""动态合批执行器。

设计要点：

* 单队列 + 单调度线程负责合批，线程池负责实际计算；
* 只有模型名、显式版本解析结果、形状签名三者一致的请求才会同批；
* 批次形成瞬间固定模型对象（版本），之后即使重载也继续用旧版本跑完；
* 结果严格按批内位置拆分，取消/超时的请求只影响自身，不会错配他人结果。
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from typing import Any, Dict, List, Optional, Tuple

from .errors import (
    InferenceError,
    RequestTimeoutError,
    UnknownModelError,
)
from .model import LocalModel
from .registry import ModelRegistry
from .request import InferenceRequest, InferenceResponse, LifecycleState


@dataclass
class BatchingConfig:
    max_batch_size: int = 8
    max_wait_ms: float = 20.0
    max_pending: int = 64
    max_workers: int = 4
    schedule_tick_ms: float = 2.0


@dataclass
class ExecutorStats:
    submitted: int = 0
    executed: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped_cancelled: int = 0
    skipped_timeout: int = 0
    batches_run: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _bump(self, **fields: int) -> None:
        with self._lock:
            for key, value in fields.items():
                setattr(self, key, getattr(self, key) + value)

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return {
                "submitted": self.submitted,
                "executed": self.executed,
                "succeeded": self.succeeded,
                "failed": self.failed,
                "skipped_cancelled": self.skipped_cancelled,
                "skipped_timeout": self.skipped_timeout,
                "batches_run": self.batches_run,
            }


class DynamicBatchingExecutor:
    def __init__(
        self,
        registry: ModelRegistry,
        config: Optional[BatchingConfig] = None,
    ) -> None:
        self.registry = registry
        self.config = config or BatchingConfig()
        # 排队上限以“等待中 + 在途”总数为准（见 submit 中的计数），
        # 队列本身只承担调度线程与提交线程之间的交接。
        self._queue: "Queue[Tuple[str, Any, InferenceRequest]]" = Queue()
        self._workers = ThreadPoolExecutor(
            max_workers=self.config.max_workers,
            thread_name_prefix="infer-worker",
        )
        self._wake = threading.Event()
        self._stop = threading.Event()
        self.stats = ExecutorStats()
        self._pending = 0
        self._pending_lock = threading.Lock()
        self._scheduler = threading.Thread(
            target=self._schedule_loop,
            name="infer-batcher",
            daemon=True,
        )
        self._scheduler.start()

    @property
    def pending(self) -> int:
        return self._pending

    def submit(self, token: Any, request: InferenceRequest) -> None:
        """入队；调用方负责已完成模型名存在性与默认版本解析。"""
        with self._pending_lock:
            if self._pending >= self.config.max_pending:
                raise Full(
                    f"模型 {request.model_name!r} 待处理请求已达上限 "
                    f"{self.config.max_pending}"
                )
            self._pending += 1
        self._queue.put_nowait((request.model_name, token, request))
        self.stats._bump(submitted=1)
        self._wake.set()

    def _drain_queue(self) -> List[Tuple[str, Any, InferenceRequest]]:
        items: List[Tuple[str, Any, InferenceRequest]] = []
        while True:
            try:
                items.append(self._queue.get_nowait())
            except Empty:
                break
        return items

    @staticmethod
    def _resolve_version(
        request: InferenceRequest,
        registry: ModelRegistry,
    ) -> Tuple[Optional[LocalModel], Optional[str], Optional[BaseException]]:
        try:
            model = registry.resolve(request.model_name, request.model_version)
        except UnknownModelError as exc:
            return None, None, exc
        return model, model.version, None

    @staticmethod
    def _group_key(request: InferenceRequest, version: str) -> Tuple:
        return (request.model_name, version, tuple(
            (name, value.shape, value.dtype.str)
            for name, value in sorted(request.inputs.items())
        ))

    def _schedule_loop(self) -> None:
        waiting: Dict[Tuple, List[Tuple[str, Any, InferenceRequest]]] = {}
        first_seen: Dict[Tuple, float] = {}
        tick = max(0.0005, self.config.schedule_tick_ms / 1000.0)
        wait_window = self.config.max_wait_ms / 1000.0
        max_batch = self.config.max_batch_size

        while not self._stop.is_set():
            self._wake.wait(timeout=tick)
            self._wake.clear()

            for item in self._drain_queue():
                _, _, request = item
                _, version, resolve_error = self._resolve_version(request, self.registry)
                if resolve_error is not None:
                    self._reject(item, resolve_error)
                    continue
                assert version is not None
                key = self._group_key(request, version)
                waiting.setdefault(key, []).append(item)
                first_seen.setdefault(key, time.monotonic())

            now = time.monotonic()
            for key in list(waiting.keys()):
                group = waiting[key]
                due = now - first_seen[key] >= wait_window
                while group and (due or len(group) >= max_batch):
                    take = min(max_batch, len(group))
                    batch_items = group[:take]
                    del group[:take]
                    self._dispatch(key, batch_items)
                    if not group:
                        break
                    first_seen[key] = time.monotonic()
                    due = False
                if not group:
                    del waiting[key]
                    first_seen.pop(key, None)

        for group in list(waiting.values()):
            for item in group:
                self._reject(item, InferenceError("执行器已关闭，请求未执行"))

    def _reject(self, item: Tuple[str, Any, InferenceRequest], exc: BaseException) -> None:
        _, _, request = item
        request.fail(exc)
        with self._pending_lock:
            self._pending -= 1

    def _dispatch(
        self,
        key: Tuple,
        items: List[Tuple[str, Any, InferenceRequest]],
    ) -> None:
        # 批次形成瞬间解析并固定模型对象：在途批次始终使用完整旧版本。
        model_name, version, _signature = key
        model = self.registry.resolve(model_name, version)
        live: List[InferenceRequest] = []
        for _, _, request in items:
            if request.cancelled:
                self.stats._bump(skipped_cancelled=1)
                with self._pending_lock:
                    self._pending -= 1
                continue
            if request.expired:
                request.fail(RequestTimeoutError(f"请求 {request.request_id} 排队超时"))
                self.stats._bump(skipped_timeout=1)
                with self._pending_lock:
                    self._pending -= 1
                continue
            request.bind_version(version)
            request.transition(LifecycleState.BATCHED)
            live.append(request)
        if not live:
            return
        self._workers.submit(self._run_batch, model, live)

    def _run_batch(self, model: LocalModel, requests: List[InferenceRequest]) -> None:
        batch_size = len(requests)
        started = time.monotonic()
        outputs: Optional[Dict[str, np.ndarray]] = None
        run_error: Optional[BaseException] = None
        try:
            for request in requests:
                model.validate_inputs(request.inputs)
            batched_inputs = model.stack_inputs([req.inputs for req in requests])
            for request in requests:
                request.transition(LifecycleState.RUNNING)
            outputs = model.predict(batched_inputs)
            for name, value in outputs.items():
                if value.shape[0] != batch_size:
                    raise InferenceError(
                        f"模型 {model.name}:{model.version} 输出 {name!r} 的 batch 维 "
                        f"为 {value.shape[0]}，与批大小 {batch_size} 不一致"
                    )
        except BaseException as exc:  # 计算失败必须通知批内每个请求
            run_error = exc

        elapsed_ms = (time.monotonic() - started) * 1000.0
        self.stats._bump(executed=batch_size, batches_run=1)

        for index, request in enumerate(requests):
            with self._pending_lock:
                self._pending -= 1
            if request.cancelled:
                # 结果已算出但调用方已取消：只丢弃该位置，不影响其他人的下标对应。
                self.stats._bump(skipped_cancelled=1)
                continue
            remaining = request.remaining
            if remaining is not None and remaining <= 0:
                request.fail(RequestTimeoutError(f"请求 {request.request_id} 执行超时"))
                self.stats._bump(skipped_timeout=1)
                continue
            if run_error is not None:
                request.fail(run_error)
                self.stats._bump(failed=1)
                continue
            assert outputs is not None
            single_outputs = {name: value[index] for name, value in outputs.items()}
            response = InferenceResponse(
                request_id=request.request_id,
                outputs=single_outputs,
                model_name=model.name,
                model_version=model.version,
                batch_size=batch_size,
                elapsed_ms=elapsed_ms,
            )
            if request.deliver(response):
                self.stats._bump(succeeded=1)
            else:
                self.stats._bump(skipped_cancelled=1)

    def shutdown(self, wait: bool = True) -> None:
        self._stop.set()
        self._wake.set()
        self._scheduler.join(timeout=2.0)
        self._workers.shutdown(wait=wait, cancel_futures=not wait)
