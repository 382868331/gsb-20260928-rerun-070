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
    VersionContractError,
)
from .executor import DynamicBatchingExecutor, ExecutorConfig
from .groups import (
    CandidateReport,
    DeliveryRecord,
    GroupReport,
    GroupStore,
    HedgeBudgetReport,
    HedgeConfig,
    ItemReport,
    RequestGroup,
)
from .models import AddConstantModel, LinearModel, Model
from .types import (
    BatchAttribution,
    CandidateState,
    GroupState,
    HedgeDelivery,
    ItemState,
    PredictionFuture,
    Priority,
    Request,
    RequestState,
    VersionContract,
)

__all__ = [
    "AddConstantModel",
    "BatchAttribution",
    "CandidateReport",
    "CandidateState",
    "DeliveryRecord",
    "DynamicBatchingExecutor",
    "EngineStoppedError",
    "ExecutorConfig",
    "GroupReport",
    "GroupState",
    "GroupStore",
    "HedgeBudgetReport",
    "HedgeConfig",
    "HedgeDelivery",
    "ItemReport",
    "ItemState",
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
    "RequestGroup",
    "VersionContract",
    "VersionContractError",
]
