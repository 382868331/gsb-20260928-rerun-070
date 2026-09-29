"""框架内统一的异常类型。"""


class InferenceError(Exception):
    """所有框架异常的基类。"""


class UnknownModelError(InferenceError):
    """请求了未注册的模型或版本。"""


class ShapeMismatchError(InferenceError):
    """输入形状/数据类型与模型声明不兼容。"""


class ReloadError(InferenceError):
    """模型新版本构造失败；旧版本保持不变。"""


class QueueFullError(InferenceError):
    """等待队列达到上限，请求被立即拒绝。"""


class RequestCancelledError(InferenceError):
    """请求在执行前或等待结果时被取消。"""


class RequestTimeoutError(InferenceError):
    """请求在截止时间前未拿到结果。"""
