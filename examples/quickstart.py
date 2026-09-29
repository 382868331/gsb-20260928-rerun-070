"""最小可运行示例：注册内置小模型 -> 动态合批推理 -> 版本热切换。"""

import numpy as np

from inferkit import (
    BatchingConfig,
    InferenceRequest,
    InferenceService,
    ServiceConfig,
    TinyLinear,
    serial_forward,
)


def main() -> None:
    service = InferenceService(config=ServiceConfig(
        default_timeout=2.0,
        batching=BatchingConfig(max_batch_size=4, max_wait_ms=20),
    ))
    service.register_model(TinyLinear(in_features=4, out_features=3, version="v1"))

    x = np.array([1.0, -2.0, 3.0, 0.5], dtype=np.float32)
    response = service.infer("tiny_linear", {"x": x})
    print("合批结果:", response.outputs["y"])
    print("串行对照:", serial_forward(TinyLinear(version="v1"), {"x": x})["y"])

    # 热切换到新版本：在途批次继续用旧版本，之后形成的批次用新版本。
    service.reload_model(
        "tiny_linear",
        lambda version: TinyLinear(in_features=4, out_features=3, version=version),
        "v2",
    )
    response_v2 = service.infer("tiny_linear", {"x": x})
    print("新版本结果:", response_v2.outputs["y"], "version =", response_v2.model_version)

    # 取消语义。
    handle = service.submit(InferenceRequest("tiny_linear", {"x": x}))
    handle.cancel()

    service.shutdown()


if __name__ == "__main__":
    main()
