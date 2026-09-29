"""本地可直接计算的小模型抽象与两个内置实现。

这些模型只依赖 numpy，权重在构造时随机或固定生成，无需下载任何大模型。
后续接入真实推理后端时，只需继承 :class:`LocalModel` 并实现
:meth:`LocalModel.predict`，请求生命周期与动态合批逻辑完全复用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import numpy as np

from .errors import ShapeMismatchError


@dataclass(frozen=True)
class TensorSpec:
    """单个输入张量的规格（单条样本，不含合批维度）。"""

    shape: Tuple[int, ...]
    dtype: np.dtype

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", np.dtype(self.dtype))


class LocalModel:
    """本地小模型基类。

    子类通过 :attr:`input_specs` 声明可接受的单样本输入形状与 dtype，
    并实现 :meth:`predict`（输入/输出均为带 batch 维的 numpy 数组）。
    """

    def __init__(
        self,
        name: str,
        version: str,
        input_specs: Mapping[str, TensorSpec],
    ) -> None:
        self.name = name
        self.version = version
        self.input_specs: Dict[str, TensorSpec] = dict(input_specs)

    def validate_inputs(self, inputs: Mapping[str, np.ndarray]) -> None:
        """校验单条请求的输入与模型声明兼容。"""
        if set(inputs) != set(self.input_specs):
            expected = sorted(self.input_specs)
            got = sorted(inputs)
            raise ShapeMismatchError(
                f"模型 {self.name}:{self.version} 输入名不匹配，期望 {expected}，实际 {got}"
            )
        for tensor_name, spec in self.input_specs.items():
            array = np.asarray(inputs[tensor_name])
            if array.shape != spec.shape:
                raise ShapeMismatchError(
                    f"模型 {self.name}:{self.version} 输入 {tensor_name!r} 形状不兼容，"
                    f"期望 {spec.shape}，实际 {array.shape}"
                )
            if array.dtype != spec.dtype:
                raise ShapeMismatchError(
                    f"模型 {self.name}:{self.version} 输入 {tensor_name!r} dtype 不兼容，"
                    f"期望 {spec.dtype}，实际 {array.dtype}"
                )

    def batch_signature(self, inputs: Mapping[str, np.ndarray]) -> Tuple:
        """返回用于判断能否同批的形状签名（形状+dtype 完全一致才可混批）。"""
        return tuple(
            (tensor_name, inputs[tensor_name].shape, inputs[tensor_name].dtype.str)
            for tensor_name in sorted(inputs)
        )

    def stack_inputs(self, request_inputs):
        """把多条形状一致的单样本输入沿第 0 维堆叠为批量输入。"""
        names = list(self.input_specs)
        return {
            tensor_name: np.stack(
                [np.asarray(item[tensor_name]) for item in request_inputs], axis=0
            )
            for tensor_name in names
        }

    def predict(self, batched_inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """前向计算。输入和输出的第 0 维都是 batch 维。"""
        raise NotImplementedError

    def close(self) -> None:
        """释放资源的钩子，默认无操作。"""

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<{type(self).__name__} {self.name}:{self.version}>"


class TinyLinear(LocalModel):
    """``y = x W + b`` 的小型线性模型。

    不同版本通过权重区分，便于验证“旧批次用旧版本、新批次用新版本”。
    """

    def __init__(
        self,
        in_features: int = 4,
        out_features: int = 3,
        version: str = "v1",
        seed: int = 0,
        name: str = "tiny_linear",
        dtype=np.float32,
    ) -> None:
        super().__init__(
            name=name,
            version=version,
            input_specs={"x": TensorSpec((in_features,), np.dtype(dtype))},
        )
        rng = np.random.default_rng(seed)
        # v1 与其它版本的权重明显不同，保证版本切换可观测。
        scale = 1.0 if version == "v1" else 2.0
        self.weight = (rng.standard_normal((in_features, out_features)) * scale).astype(dtype)
        self.bias = (np.arange(out_features, dtype=dtype) * scale)

    def predict(self, batched_inputs):
        x = batched_inputs["x"]
        return {"y": x @ self.weight + self.bias}


class TinyMLP(LocalModel):
    """两层 tanh 小网络，输出单个 logit。"""

    def __init__(
        self,
        in_features: int = 4,
        hidden_features: int = 5,
        version: str = "v1",
        seed: int = 42,
        name: str = "tiny_mlp",
        dtype=np.float32,
    ) -> None:
        super().__init__(
            name=name,
            version=version,
            input_specs={"x": TensorSpec((in_features,), np.dtype(dtype))},
        )
        rng = np.random.default_rng(seed + (0 if version == "v1" else 1))
        self.w1 = rng.standard_normal((in_features, hidden_features)).astype(dtype) * 0.5
        self.b1 = np.zeros(hidden_features, dtype=dtype)
        self.w2 = rng.standard_normal((hidden_features, 1)).astype(dtype) * 0.5
        self.b2 = np.zeros(1, dtype=dtype)

    def predict(self, batched_inputs):
        x = batched_inputs["x"]
        hidden = np.tanh(x @ self.w1 + self.b1)
        logits = hidden @ self.w2 + self.b2
        return {"logits": logits}


def serial_forward(
    model: LocalModel,
    single_inputs: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """逐条串行（batch=1）执行一次前向，作为合批结果的对照基准。"""
    model.validate_inputs(single_inputs)
    batched = {
        tensor_name: value[None, ...]
        for tensor_name, value in single_inputs.items()
    }
    outputs = model.predict(batched)
    return {name: value[0] for name, value in outputs.items()}
