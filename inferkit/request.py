"""推理请求与响应，以及贯穿全流程的生命周期状态。"""

from __future__ import annotations

import enum
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .errors import RequestCancelledError


class LifecycleState(enum.Enum):
    """请求生命周期状态。

    PENDING -> QUEUED -> BATCHED -> RUNNING -> DELIVERED
    任意中间状态都可能转入 CANCELLED / TIMED_OUT / FAILED。
    """

    PENDING = "pending"
    QUEUED = "queued"
    BATCHED = "batched"
    RUNNING = "running"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    FAILED = "failed"


_TERMINAL_STATES = {
    LifecycleState.DELIVERED,
    LifecycleState.CANCELLED,
    LifecycleState.TIMED_OUT,
    LifecycleState.FAILED,
}


@dataclass
class InferenceRequest:
    """一次推理调用的完整上下文。

    Parameters
    ----------
    model_name:
        目标模型名。
    inputs:
        命名输入张量。
    model_version:
        显式指定版本；为 ``None`` 时使用当前默认版本。
    timeout:
        总超时时间（秒），``None`` 表示不限制。
    request_id:
        自定义请求 ID；默认生成 UUID。
    """

    model_name: str
    inputs: Dict[str, np.ndarray]
    model_version: Optional[str] = None
    timeout: Optional[float] = None
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.monotonic)
    state: LifecycleState = field(default=LifecycleState.PENDING, init=False)
    metadata: Dict[str, Any] = field(default_factory=dict, init=False)

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _cancel_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _future: Optional["Any"] = field(default=None, init=False, repr=False)
    _bound_version: Optional[str] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.inputs = {
            name: np.asarray(value)
            for name, value in self.inputs.items()
        }

    @property
    def deadline(self) -> Optional[float]:
        if self.timeout is None:
            return None
        return self.created_at + self.timeout

    @property
    def remaining(self) -> Optional[float]:
        """距离截止时间还剩多少秒；已过期返回负数，不限时返回 ``None``。"""
        deadline = self.deadline
        if deadline is None:
            return None
        return deadline - time.monotonic()

    @property
    def expired(self) -> bool:
        remaining = self.remaining
        return remaining is not None and remaining <= 0

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    @property
    def bound_version(self) -> Optional[str]:
        """批次形成时为本请求固定的模型版本。"""
        return self._bound_version

    def bind_future(self, future: "Any") -> None:
        self._future = future

    def transition(self, new_state: LifecycleState) -> bool:
        """尝试进行状态迁移；已终结的请求不再接受迁移。"""
        with self._lock:
            if self.state in _TERMINAL_STATES:
                return False
            self.state = new_state
            return True

    def bind_version(self, version: str) -> None:
        """批次形成时固定执行版本（旧批次继续用旧版本）。"""
        with self._lock:
            self._bound_version = version

    def cancel(self) -> bool:
        """请求取消。

        取消后结果不会正常交付：尚未执行则直接跳过，正在执行则在结果拆分
        阶段剔除该请求，其他请求的结果映射不受影响。
        """
        with self._lock:
            if self.state in _TERMINAL_STATES:
                return False
            self.state = LifecycleState.CANCELLED
            self._cancel_event.set()
        if self._future is not None and not self._future.done():
            self._future.set_exception(RequestCancelledError(self.request_id))
        return True

    def deliver(self, response: "InferenceResponse") -> bool:
        """正常交付结果；已取消/超时的请求不会被覆盖。"""
        if not self.transition(LifecycleState.DELIVERED):
            return False
        if self._future is not None and not self._future.done():
            self._future.set_result(response)
        return True

    def fail(self, exc: BaseException) -> bool:
        """交付执行异常；已终结（如已取消）的请求保持原结果语义。"""
        from .errors import RequestTimeoutError

        state = (
            LifecycleState.TIMED_OUT
            if isinstance(exc, RequestTimeoutError)
            else LifecycleState.FAILED
        )
        if not self.transition(state):
            return False
        if self._future is not None and not self._future.done():
            self._future.set_exception(exc)
        return True


@dataclass
class InferenceResponse:
    """推理结果。"""

    request_id: str
    outputs: Dict[str, np.ndarray]
    model_name: str
    model_version: str
    batch_size: int
    elapsed_ms: float
