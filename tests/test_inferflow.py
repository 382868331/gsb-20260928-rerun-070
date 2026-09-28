from __future__ import annotations

import threading
import time
import unittest

from inferflow import (
    AddConstantModel,
    DynamicBatchingExecutor,
    ExecutorConfig,
    IncompatibleInputError,
    LinearModel,
    ModelReloadError,
    QueueFullError,
    Request,
    RequestCancelledError,
    RequestState,
    RequestTimeoutError,
)


def linear_sample(values: tuple[float, ...]) -> Request:
    return Request(data=values, shape=(len(values),), dtype=float)


class GateModel:
    def __init__(self, version="gate-v1", output=None, shape=(4,)):
        self.name = "gate"
        self.version = version
        self.shape = shape
        self.output = output or (version,)
        self.entered = threading.Event()
        self.release = threading.Event()
        self.batches: list[list[tuple]] = []
        self._lock = threading.Lock()

    def check(self, request_shape, dtype):
        if tuple(request_shape) != self.shape:
            raise IncompatibleInputError("shape mismatch")

    def predict(self, samples):
        self.entered.set()
        self.release.wait(2)
        with self._lock:
            self.batches.append(list(samples))
        return [self.output + (index,) for index, _ in enumerate(samples)]


class RecordingLinearModel(LinearModel):
    def __init__(self):
        super().__init__()
        self.batches = []

    def predict(self, samples):
        self.batches.append(list(samples))
        return super().predict(samples)


class ExecutorTests(unittest.TestCase):
    def test_builtin_model_matches_serial_results_and_batches(self):
        model = RecordingLinearModel()
        config = ExecutorConfig(max_batch_size=4, max_wait_time=0.08)
        samples = [
            (0.1, 0.2, 0.3, 0.4),
            (1.0, 2.0, 3.0, 4.0),
            (-0.5, 0.5, 1.5, 2.5),
        ]
        with DynamicBatchingExecutor(model, config) as executor:
            futures = [executor.submit(linear_sample(sample)) for sample in samples]
            batched = [future.result(1) for future in futures]

        self.assertEqual(batched, [model.predict_one(sample) for sample in samples])
        self.assertEqual(len(model.batches), 1)
        self.assertEqual(len(model.batches[0]), 3)

    def test_incompatible_shapes_run_in_separate_batches(self):
        model = AddConstantModel(input_shape=(3,))
        config = ExecutorConfig(max_batch_size=4, max_wait_time=0.02)
        with DynamicBatchingExecutor(model, config) as executor:
            good = executor.submit(Request(data=(1.0, 2.0, 3.0), shape=(3,)))
            bad = executor.submit(Request(data=(1.0, 2.0), shape=(2,)))
            self.assertEqual(good.result(1), (2.0, 3.0, 4.0))
            with self.assertRaises(IncompatibleInputError):
                bad.result(1)
            self.assertEqual(good.state, RequestState.FINISHED)
            self.assertEqual(bad.state, RequestState.FAILED)

    def test_cancel_removes_only_cancelled_result(self):
        model = GateModel()
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001)
        with DynamicBatchingExecutor(model, config) as executor:
            first = executor.submit(linear_sample((1, 2, 3, 4)), timeout=None)
            self.assertTrue(model.entered.wait(1))

            second = executor.submit(linear_sample((5, 6, 7, 8)), timeout=None)
            third = executor.submit(linear_sample((9, 10, 11, 12)), timeout=None)
            time.sleep(0.03)
            self.assertTrue(executor.cancel(second.request_id))

            model.release.set()
            self.assertEqual(first.result(1), ("gate-v1", 0))
            self.assertEqual(third.result(1), ("gate-v1", 0))
            with self.assertRaises(RequestCancelledError):
                second.result(1)
            self.assertEqual(second.state, RequestState.CANCELLED)

    def test_running_cancellation_suppresses_delivery_without_shifting_outputs(self):
        model = GateModel()
        config = ExecutorConfig(max_batch_size=2, max_wait_time=0.05)
        with DynamicBatchingExecutor(model, config) as executor:
            first = executor.submit(linear_sample((1, 2, 3, 4)), timeout=None)
            second = executor.submit(linear_sample((5, 6, 7, 8)), timeout=None)
            self.assertTrue(model.entered.wait(1))
            self.assertTrue(executor.cancel(first.request_id))
            model.release.set()
            self.assertEqual(second.result(1), ("gate-v1", 1))
            with self.assertRaises(RequestCancelledError):
                first.result(1)

    def test_queue_limit_is_enforced(self):
        model = GateModel()
        config = ExecutorConfig(
            max_queue_size=1,
            max_batch_size=1,
            max_wait_time=0.001,
        )
        with DynamicBatchingExecutor(model, config) as executor:
            executor.submit(linear_sample((1, 2, 3, 4)), timeout=None)
            self.assertTrue(model.entered.wait(1))
            executor.submit(linear_sample((5, 6, 7, 8)), timeout=None)
            with self.assertRaises(QueueFullError):
                executor.submit(linear_sample((9, 10, 11, 12)), timeout=None)
            model.release.set()

    def test_request_timeout_is_delivered(self):
        model = GateModel()
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001)
        with DynamicBatchingExecutor(model, config) as executor:
            future = executor.submit(linear_sample((1, 2, 3, 4)), timeout=0.03)
            self.assertTrue(model.entered.wait(1))
            with self.assertRaises(RequestTimeoutError):
                future.result(1)
            self.assertEqual(future.state, RequestState.TIMED_OUT)
            model.release.set()

    def test_reload_failure_keeps_old_model(self):
        model = GateModel(version="old")
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001)
        with DynamicBatchingExecutor(model, config) as executor:
            def fail_loader():
                raise RuntimeError("bad weights")

            with self.assertRaises(ModelReloadError):
                executor.reload_model(fail_loader)
            self.assertIs(executor.model, model)
            model.release.set()

    def test_in_flight_batch_uses_old_version_new_batch_uses_new_version(self):
        old = GateModel(version="old")
        new = GateModel(version="new")
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001)
        with DynamicBatchingExecutor(old, config) as executor:
            first = executor.submit(linear_sample((1, 2, 3, 4)), timeout=None)
            self.assertTrue(old.entered.wait(1))

            reloaded = threading.Event()

            def reload():
                executor.reload_model(new, timeout=2)
                reloaded.set()

            threading.Thread(target=reload, daemon=True).start()
            time.sleep(0.05)
            second = executor.submit(linear_sample((5, 6, 7, 8)), timeout=None)
            old.release.set()

            self.assertTrue(reloaded.wait(1))
            self.assertIs(executor.model, new)
            self.assertTrue(new.entered.wait(1))
            new.release.set()
            self.assertEqual(first.result(1), ("old", 0))
            self.assertEqual(second.result(1), ("new", 0))


if __name__ == "__main__":
    unittest.main()
