import threading
import time

import numpy as np
import pytest

from inferkit import (
    InferenceRequest,
    QueueFullError,
    RequestCancelledError,
    ShapeMismatchError,
    UnknownModelError,
    serial_forward,
    TinyLinear,
)


def test_infer_matches_serial(service, sample_inputs):
    resp = service.infer("tiny_linear", sample_inputs)
    expected = serial_forward(TinyLinear(version="v1"), sample_inputs)["y"]
    np.testing.assert_allclose(resp.outputs["y"], expected, rtol=1e-6)
    assert resp.model_version == "v1"
    assert resp.batch_size >= 1


def test_explicit_version(service, sample_inputs):
    resp_v1 = service.infer("tiny_linear", sample_inputs, version="v1")
    resp_v2 = service.infer("tiny_linear", sample_inputs, version="v2")
    assert resp_v1.model_version == "v1"
    assert resp_v2.model_version == "v2"
    assert not np.allclose(resp_v1.outputs["y"], resp_v2.outputs["y"])


def test_shape_incompatible_rejected(service):
    with pytest.raises(ShapeMismatchError):
        service.infer("tiny_linear", {"x": np.zeros(3, dtype=np.float32)})  # 4 != 3
    with pytest.raises(ShapeMismatchError):
        service.infer("tiny_linear", {"x": np.zeros(4, dtype=np.float64)})
    with pytest.raises(ShapeMismatchError):
        service.infer("tiny_linear", {"wrong": np.zeros(4, dtype=np.float32)})


def test_unknown_model(service, sample_inputs):
    with pytest.raises(UnknownModelError):
        service.infer("nope", sample_inputs)
    with pytest.raises(UnknownModelError):
        service.infer("tiny_linear", sample_inputs, version="v9")


def test_dynamic_batching_results_align(service):
    # 同形状请求被合批，结果必须与串行逐条对照，且 request_id 不错位。
    xs = [np.full(4, i, dtype=np.float32) for i in range(7)]
    handles = [service.submit(InferenceRequest("tiny_linear", {"x": x})) for x in xs]
    responses = [h.result() for h in handles]
    model = TinyLinear(version="v1")
    assert [r.request_id for r in responses] == [h.request.request_id for h in handles]
    for x, resp in zip(xs, responses):
        expected = serial_forward(model, {"x": x})["y"]
        np.testing.assert_allclose(resp.outputs["y"], expected, rtol=1e-5, atol=1e-6)
    assert service.stats()["executor"]["batches_run"] >= 2


def test_incompatible_shapes_never_mixed(service):
    # 不同形状各自成组，结果仍然各自正确。
    h1 = service.submit(InferenceRequest("tiny_linear", {"x": np.zeros(4, np.float32)}))
    h2 = service.submit(InferenceRequest("tiny_mlp", {"x": np.ones(4, np.float32)}))
    r1 = h1.result()
    r2 = h2.result()
    assert "y" in r1.outputs
    assert "logits" in r2.outputs
    assert r1.request_id == h1.request.request_id
    assert r2.request_id == h2.request.request_id


def test_queue_full_rejects_immediately():
    from inferkit import InferenceService, ServiceConfig, BatchingConfig

    class SlowLinear(TinyLinear):
        def predict(self, batched_inputs):
            import time as _t
            _t.sleep(0.3)
            return super().predict(batched_inputs)

    config = ServiceConfig(
        default_timeout=10,
        batching=BatchingConfig(
            max_batch_size=4, max_wait_ms=500, max_pending=4, max_workers=1
        ),
    )
    svc = InferenceService(config=config)
    svc.register_model(SlowLinear(version="v1"))
    try:
        # 第一批 4 个进入 worker；再多提交必须在队列满时立即抛错。
        handles = []
        for _ in range(4):
            handles.append(svc.submit(InferenceRequest(
                "tiny_linear", {"x": np.zeros(4, np.float32)}
            )))
        with pytest.raises(QueueFullError):
            for _ in range(8):
                svc.submit(InferenceRequest(
                    "tiny_linear", {"x": np.zeros(4, np.float32)}
                ))
        for h in handles:
            h.result()
    finally:
        svc.shutdown()


def test_request_timeout_waiting():
    from inferkit import InferenceService, ServiceConfig, BatchingConfig

    class SlowLinear(TinyLinear):
        def predict(self, batched_inputs):
            import time as _t
            _t.sleep(0.4)
            return super().predict(batched_inputs)

    config = ServiceConfig(
        default_timeout=None,
        batching=BatchingConfig(
            max_batch_size=4, max_wait_ms=10, max_pending=32, max_workers=1
        ),
    )
    svc = InferenceService(config=config)
    svc.register_model(SlowLinear(version="v1"))
    try:
        svc.submit(InferenceRequest(
            "tiny_linear", {"x": np.zeros(4, np.float32)}, timeout=5
        ))  # 占住 worker
        late = svc.submit(InferenceRequest(
            "tiny_linear", {"x": np.zeros(4, np.float32)}, timeout=0.05
        ))
        with pytest.raises(Exception):
            late.result()
        assert late.request.cancelled or late.request.state.value in ("timed_out", "failed")
    finally:
        svc.shutdown()


def test_cancel_removes_from_delivery_without_scrambling(service):
    from inferkit import InferenceService, ServiceConfig, BatchingConfig, LocalModel, TensorSpec

    class GatedLinear(LocalModel):
        def __init__(self, version, gate):
            super().__init__("gated", version,
                             {"x": TensorSpec((4,), np.dtype(np.float32))})
            self.gate = gate
            self.scale = 1.0 if version == "v1" else 5.0

        def predict(self, batched_inputs):
            self.gate.wait(timeout=5)
            return {"y": batched_inputs["x"] * self.scale}

    gate = threading.Event()
    config = ServiceConfig(
        default_timeout=10,
        batching=BatchingConfig(max_batch_size=4, max_wait_ms=10, max_pending=16,
                               max_workers=2),
    )
    svc = InferenceService(config=config)
    svc.register_model(GatedLinear("v1", gate))
    try:
        xs = [np.full(4, i, dtype=np.float32) for i in range(4)]
        handles = [svc.submit(InferenceRequest("gated", {"x": x})) for x in xs]
        # 等批次真正进入在途执行阶段，再取消中间两个请求。
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not all(
            h.request.state.value == "running" for h in handles
        ):
            time.sleep(0.002)
        assert all(h.request.state.value == "running" for h in handles)
        assert handles[1].cancel()
        assert handles[3].cancel()
        gate.set()

        outcomes = {}
        for i, handle in enumerate(handles):
            try:
                outcomes[i] = ("ok", handle.result())
            except RequestCancelledError:
                outcomes[i] = ("cancelled", None)
        for i in (1, 3):
            assert outcomes[i][0] == "cancelled"
        for i in (0, 2):
            status, resp = outcomes[i]
            assert status == "ok"
            np.testing.assert_allclose(resp.outputs["y"], xs[i], rtol=1e-6)
            assert resp.request_id == handles[i].request.request_id
    finally:
        gate.set()
        svc.shutdown()


def test_cancel_before_execution():
    # 批次尚未到形成时间前取消：请求不出现在任何结果交付中。
    from inferkit import InferenceService, ServiceConfig, BatchingConfig

    config = ServiceConfig(
        default_timeout=10,
        batching=BatchingConfig(max_batch_size=4, max_wait_ms=500, max_pending=8),
    )
    svc = InferenceService(config=config)
    svc.register_model(TinyLinear(version="v1"))
    try:
        handle = svc.submit(InferenceRequest(
            "tiny_linear", {"x": np.zeros(4, np.float32)}
        ))
        assert handle.cancel()
        with pytest.raises(RequestCancelledError):
            handle.result()
        svc.wait_drained()
        stats = svc.stats()["executor"]
        assert stats["skipped_cancelled"] >= 1
        assert stats["succeeded"] == 0
    finally:
        svc.shutdown()
