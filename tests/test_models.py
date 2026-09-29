"""合批结果必须与逐条串行结果一致。"""

import numpy as np

from inferkit import TinyLinear, TinyMLP, serial_forward


def test_serial_forward_matches_direct():
    model = TinyLinear(version="v1")
    x = np.array([1.0, -2.0, 3.0, 0.5], dtype=np.float32)
    out = serial_forward(model, {"x": x})
    expected = (x @ model.weight + model.bias)
    np.testing.assert_allclose(out["y"], expected, rtol=1e-6)


def test_batched_equals_serial_paths():
    model = TinyMLP(version="v1")
    batch = np.linspace(-1, 1, 24, dtype=np.float32).reshape(6, 4)
    batched_out = model.predict({"x": batch})["logits"].reshape(-1)
    for i in range(batch.shape[0]):
        serial_out = serial_forward(model, {"x": batch[i]})["logits"].reshape(-1)
        np.testing.assert_allclose(batched_out[i], serial_out, rtol=1e-5, atol=1e-6)


def test_versions_have_different_weights():
    v1 = TinyLinear(version="v1")
    v2 = TinyLinear(version="v2")
    assert not np.allclose(v1.weight, v2.weight)
