"""推理服务门面：串联注册、入队上限、超时与取消。"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from .errors import (
    InferenceError,
    QueueFullError,
    RequestCancelledError,
    RequestTimeoutError,
    ReloadError,
)
from .executor import BatchingConfig, DynamicBatchingExecutor
from .model import LocalModel
from .registry import ModelRegistry
from .request import InferenceRequest, InferenceResponse


@dataclass
class ServiceConfig:
    default_timeout: Optional[float] = 5.0
    batching: BatchingConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.batching is None:
            self.batching = BatchingConfig()


class InferenceHandle:
    """单次提交返回的句柄，可等待结果或取消。"""

    def __init__(self, request: InferenceRequest, future: Future) -> None:
        self.request = request
        self._future = future

    def result(self, timeout: Optional[float] = None) -> InferenceResponse:
        """阻塞等待结果。

        ``timeout`` 为 ``None`` 时使用请求自身的剩余时间；超时抛出
        :class:`RequestTimeoutError` 并把请求标记为超时。
        """
        wait_timeout = timeout
        if wait_timeout is None:
            remaining = self.request.remaining
            wait_timeout = remaining if remaining is not None else None
        try:
            return self._future.result(timeout=wait_timeout)
        except RequestCancelledError:
            raise
        except InferenceError:
            raise
        except TimeoutError:
            self.request.fail(RequestTimeoutError(
                f"请求 {self.request.request_id} 等待结果超时"
            ))
            raise RequestTimeoutError(
                f"请求 {self.request.request_id} 等待结果超时"
            )

    def cancel(self) -> bool:
        """取消请求；已交付结果的请求取消无效。"""
        return self.request.cancel()

    @property
    def done(self) -> bool:
        return self._future.done()


class InferenceService:
    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        config: Optional[ServiceConfig] = None,
    ) -> None:
        self.registry = registry or ModelRegistry()
        self.config = config or ServiceConfig()
        self.executor = DynamicBatchingExecutor(
            self.registry, self.config.batching
        )
        self._inflight = 0
        self._lock = threading.Lock()

    def register_model(self, model: LocalModel, make_default: bool = True) -> None:
        self.registry.register(model, make_default=make_default)

    def submit(self, request: InferenceRequest) -> InferenceHandle:
        """提交请求。

        入队前完成默认版本解析与形状校验；队列已满立即抛出
        :class:`QueueFullError`，不阻塞调用方。
        """
        if request.timeout is None and self.config.default_timeout is not None:
            request.timeout = self.config.default_timeout
        if request.expired:
            raise RequestTimeoutError(f"请求 {request.request_id} 提交时已超时")
        # 提前解析默认版本并校验形状，使提交线程直接收到参数错误。
        model = self.registry.resolve(request.model_name, request.model_version)
        if request.model_version is None:
            request.model_version = model.version
        model.validate_inputs(request.inputs)

        future: Future = Future()
        request.bind_future(future)
        token = object()
        try:
            self.executor.submit(token, request)
        except Exception as exc:  # queue.Full
            raise QueueFullError(
                f"模型 {request.model_name!r} 等待队列已满（上限 "
                f"{self.config.batching.max_pending}），请求 {request.request_id} 被拒绝"
            ) from exc
        with self._lock:
            self._inflight += 1
        future.add_done_callback(lambda _f: self._release())
        return InferenceHandle(request, future)

    def _release(self) -> None:
        with self._lock:
            self._inflight -= 1

    def infer(
        self,
        model_name: str,
        inputs: Dict[str, Any],
        *,
        version: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> InferenceResponse:
        """同步推理的便捷方法。"""
        request = InferenceRequest(
            model_name=model_name,
            inputs=inputs,
            model_version=version,
            timeout=timeout,
        )
        return self.submit(request).result()

    def reload_model(
        self,
        name: str,
        factory: Callable[[str], LocalModel],
        new_version: str,
        *,
        close_old: bool = False,
    ) -> LocalModel:
        """用 ``factory(new_version)`` 构造新版本并原子切换。

        构造失败时旧版本完整保留，异常包装为 :class:`ReloadError`。
        已经形成的批次继续持有旧模型对象，用完整旧版本执行完毕。
        """
        try:
            new_model = factory(new_version)
        except Exception as exc:
            raise ReloadError(
                f"模型 {name!r} 重载到 {new_version!r} 失败，保留旧版本: {exc}"
            ) from exc
        if new_model.name != name or new_model.version != new_version:
            new_model.close()
            raise ReloadError(
                f"重载模型名/版本不匹配: 期望 {name}:{new_version}，"
                f"得到 {new_model.name}:{new_model.version}"
            )
        old_model, old_version = self.registry.swap_default(name, new_model)
        if close_old and old_model is not None:
            try:
                old_model.close()
            except Exception:
                pass
        return new_model

    def wait_drained(self, timeout: Optional[float] = 5.0) -> bool:
        """等待执行器内部待处理请求清零（主要供测试使用）。"""
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.executor.pending > 0:
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.002)
        return True

    @property
    def inflight(self) -> int:
        with self._lock:
            return self._inflight

    def stats(self) -> Dict[str, Any]:
        return {
            "inflight": self.inflight,
            "executor": self.executor.stats.snapshot(),
        }

    def shutdown(self, wait: bool = True) -> None:
        self.executor.shutdown(wait=wait)
        self.registry.shutdown()
