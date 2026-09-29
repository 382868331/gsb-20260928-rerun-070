# inferkit — 模型推理运行框架（首轮：本地小模型动态合批）

纯 Python（numpy）实现的轻量推理运行框架，自带可直接计算的小模型，
无需下载任何大模型。后续接入真实推理后端时，只需实现
`LocalModel.predict`，请求生命周期、动态合批、版本切换、取消与超时
机制全部复用。

## 架构

```
InferenceRequest（生命周期：pending→queued→batched→running→delivered）
        │ submit（校验形状/版本，队列满立即拒绝）
        ▼
DynamicBatchingExecutor
   ├─ 调度线程：按 (模型, 版本, 形状签名) 分组合批
   │            批次形成瞬间固定模型对象（旧批次跑完整旧版本）
   └─ 工作线程：堆叠输入 -> model.predict -> 按下标拆分结果
        ▲
ModelRegistry（多版本注册 / 默认版本原子切换 / 重载失败保留旧版）
```

## 关键语义

- **形状不兼容不混批**：合批分组键包含每个输入的形状与 dtype，不同请求
  绝不会出现在同一批次；提交时形状不符直接抛 `ShapeMismatchError`。
- **取消只影响自身**：结果严格按批内下标拆分；取消的请求从交付中移除，
  其他人的 `request_id ↔ 输出` 对应关系不变。
- **版本切换**：批次形成时固定版本；切换前的在途批次用完整旧版本执行，
  切换后的新批次用新版本。
- **排队上限**：待处理（等待 + 在途）请求达到 `max_pending` 时
  `submit` 立即抛 `QueueFullError`，不阻塞。
- **超时**：请求带 `timeout`，排队/执行/等待结果各阶段都会检查截止时间，
  超时抛 `RequestTimeoutError`。
- **重载失败**：`factory` 构造异常包装为 `ReloadError`，旧版本保持可用。
- **串行对照**：`serial_forward(model, inputs)` 以 batch=1 执行，
  供结果正确性核对（测试中全部合批输出都与串行逐条对照）。

## 快速开始

```bash
python examples/quickstart.py
python -m pytest tests/ -q
```

```python
import numpy as np
from inferkit import InferenceService, TinyLinear

service = InferenceService()
service.register_model(TinyLinear(version="v1"))
resp = service.infer("tiny_linear", {"x": np.zeros(4, dtype=np.float32)})
print(resp.outputs["y"], resp.model_version)
```

## 扩展新模型

继承 `LocalModel`，声明输入规格并实现 `predict`（输入/输出第 0 维为
batch 维）即可被合批调度：

```python
from inferkit import LocalModel, TensorSpec

class MyModel(LocalModel):
    def __init__(self, version):
        super().__init__("my_model", version,
                         {"x": TensorSpec((16,), np.dtype(np.float32))})

    def predict(self, batched_inputs):
        return {"y": batched_inputs["x"].sum(axis=1, keepdims=True)}
```
