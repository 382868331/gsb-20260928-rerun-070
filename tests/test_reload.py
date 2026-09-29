"""模型版本切换与重载失败语义。"""

import threading
import time

import numpy as np
import pytest

from inferkit import (
    InferenceRequest,
    InferenceService,
    LocalModel,
    ReloadError,
    ServiceConfig,
    BatchingConfig,
    TensorSpec,
)


class GatedLinear(LocalModel):
    def __init__(self, version, gate, name="gated"):
        super().__init__(name, version, {"x": TensorSpec((4,), np.dtype(np.float32))})
        self.gate = gate
        self.scale = 1.0 if version == "v1" else 5.0

    def predict(self, batched_inputs):
        self.gate.wait(timeout=5)
        return {"y": batched_inputs["x"] * self.scale}


def _service(gate):
    config = ServiceConfig(
        default_timeout=10,
        batching=BatchingConfig(max_batch_size=4, max_wait_ms=10,
                               max_pending=32, max_workers=2),
    )
    svc = InferenceService(config=config)
    svc.register_model(GatedLinear("v1", gate))
    return svc


def test_inflight_batch_uses_old_version_new_batch_uses_new():
    gate = threading.Event()
    svc = _service(gate)
    try:
        old_xs = [np.full(4, 1.0, dtype=np.float32) for _ in range(2)]
        old_handles = [svc.submit(InferenceRequest("gated", {"x": x})) for x in old_xs]
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not all(
            h.request.state.value == "running" for h in old_handles
        ):
            time.sleep(0.002)
        assert all(h.request.bound_version == "v1" for h in old_handles)

        # 在途期间完成切换。
        svc.reload_model("gated", lambda v: GatedLinear(v, gate), "v2")
        assert svc.registry.current_version("gated") == "v2"

        new_xs = [np.full(4, 2.0, dtype=np.float32) for _ in range(2)]
        new_handles = [svc.submit(InferenceRequest("gated", {"x": x})) for x in new_xs]
        gate.set()

        old_responses = [h.result() for h in old_handles]
        new_responses = [h.result() for h in new_handles]
        assert all(r.model_version == "v1" for r in old_responses)
        assert all(r.model_version == "v2" for r in new_responses)
        # 旧版本 scale=1，新版本 scale=5。
        for r in old_responses:
            np.testing.assert_allclose(r.outputs["y"], np.ones(4, np.float32))
        for r in new_responses:
            np.testing.assert_allclose(r.outputs["y"], np.full(4, 10.0, np.float32))
    finally:
        gate.set()
        svc.shutdown()


def test_reload_failure_keeps_old_version(service, sample_inputs):
    def broken_factory(version):
        raise RuntimeError("weights file corrupted")

    before = service.registry.current_version("tiny_linear")
    with pytest.raises(ReloadError) as exc_info:
        service.reload_model("tiny_linear", broken_factory, "v3")
    assert "v3" in str(exc_info.value)
    assert service.registry.current_version("tiny_linear") == before
    resp = service.infer("tiny_linear", sample_inputs)
    assert resp.model_version == before


def test_reload_failure_during_model_init(service, sample_inputs):
    from inferkit import TinyLinear

    call_count = {"n": 0}

    def flaky_factory(version):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ValueError("boom")
        return TinyLinear(version=version)

    with pytest.raises(ReloadError):
        service.reload_model("tiny_linear", flaky_factory, "v3")
    assert service.registry.current_version("tiny_linear") == "v1"
    # 恢复后再次重载成功。
    service.reload_model("tiny_linear", flaky_factory, "v3")
    assert service.registry.current_version("tiny_linear") == "v3"
    assert service.infer("tiny_linear", sample_inputs).model_version == "v3"
