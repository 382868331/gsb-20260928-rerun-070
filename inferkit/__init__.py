"""inferkit: 轻量级模型推理运行框架。

首轮能力：

* :class:`~inferkit.request.InferenceRequest` /
  :class:`~inferkit.request.InferenceResponse` 描述统一请求生命周期；
* :class:`~inferkit.model.LocalModel` 是自带计算逻辑的本地小模型基类；
* :class:`~inferkit.registry.ModelRegistry` 管理模型版本，切换对在途批次原子可见；
* :class:`~inferkit.executor.DynamicBatchingExecutor` 提供按形状分组的动态合批；
* :class:`~inferkit.service.InferenceService` 组合排队上限、超时、取消与串行对照。
"""

from .errors import (
    InferenceError,
    QueueFullError,
    RequestCancelledError,
    RequestTimeoutError,
    ReloadError,
    ShapeMismatchError,
    UnknownModelError,
)
from .model import LocalModel, TensorSpec, TinyLinear, TinyMLP, serial_forward
from .registry import ModelRegistry
from .request import InferenceRequest, InferenceResponse
from .executor import DynamicBatchingExecutor, BatchingConfig
from .service import InferenceService, ServiceConfig

__all__ = [
    "InferenceError",
    "UnknownModelError",
    "ShapeMismatchError",
    "ReloadError",
    "QueueFullError",
    "RequestCancelledError",
    "RequestTimeoutError",
    "InferenceRequest",
    "InferenceResponse",
    "LocalModel",
    "TensorSpec",
    "TinyLinear",
    "TinyMLP",
    "serial_forward",
    "ModelRegistry",
    "DynamicBatchingExecutor",
    "BatchingConfig",
    "InferenceService",
    "ServiceConfig",
]
