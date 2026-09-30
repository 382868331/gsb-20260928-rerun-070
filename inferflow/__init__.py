"""inferflow: a minimal model inference serving framework.

The first milestone ships:

* request / result lifecycle objects,
* pure-Python built-in models that can be evaluated without downloads,
* a single-worker dynamic batching executor,
* safe model version reload while batches are in flight.
"""

from .errors import (
    EngineStoppedError,
    IncompatibleInputError,
    ModelReloadError,
    QueueFullError,
    RequestCancelledError,
    RequestTimeoutError,
)
from .executor import DynamicBatchingExecutor, ExecutorConfig
from .models import AddConstantModel, LinearModel, Model
from .types import (
    BatchAttribution,
    PredictionFuture,
    Priority,
    Request,
    RequestState,
)

__all__ = [
    "AddConstantModel",
    "BatchAttribution",
    "DynamicBatchingExecutor",
    "EngineStoppedError",
    "ExecutorConfig",
    "IncompatibleInputError",
    "LinearModel",
    "Model",
    "ModelReloadError",
    "PredictionFuture",
    "Priority",
    "QueueFullError",
    "Request",
    "RequestCancelledError",
    "RequestState",
    "RequestTimeoutError",
]
