class InferflowError(Exception):
    """Base class for all framework errors."""


class IncompatibleInputError(InferflowError):
    """Raised when an input does not match a model signature."""


class QueueFullError(InferflowError):
    """Raised by ``submit`` when the bounded waiting queue is full."""


class RequestTimeoutError(InferflowError):
    """Delivered to a request that exceeded its queue/execution timeout."""


class RequestCancelledError(InferflowError):
    """Delivered when a request is cancelled before its result is returned."""


class EngineStoppedError(InferflowError):
    """Raised for requests rejected or drained during executor shutdown."""


class ModelReloadError(InferflowError):
    """Raised when loading a replacement model fails."""


class VersionContractError(InferflowError):
    """Raised when a retry cannot honor the group's model-version contract."""
