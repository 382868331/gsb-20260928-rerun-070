import numpy as np
import pytest

from inferkit import (
    InferenceService,
    ServiceConfig,
    TinyLinear,
    TinyMLP,
    BatchingConfig,
)


@pytest.fixture
def service():
    config = ServiceConfig(
        default_timeout=5.0,
        batching=BatchingConfig(
            max_batch_size=4,
            max_wait_ms=20,
            max_pending=8,
            max_workers=2,
        ),
    )
    svc = InferenceService(config=config)
    svc.register_model(TinyLinear(version="v1"))
    svc.register_model(TinyLinear(version="v2"), make_default=False)
    svc.register_model(TinyMLP(version="v1"))
    yield svc
    svc.shutdown()


@pytest.fixture
def sample_inputs():
    return {"x": np.arange(4, dtype=np.float32)}
