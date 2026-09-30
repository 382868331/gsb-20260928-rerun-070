from __future__ import annotations

import threading
import time
import unittest

from inferflow import (
    AddConstantModel,
    DynamicBatchingExecutor,
    EngineStoppedError,
    ExecutorConfig,
    ExecutionInfo,
    IncompatibleInputError,
    LinearModel,
    ModelReloadError,
    Priority,
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


class PartialFailModel(GateModel):
    """Fails validation for every second request while accepting the rest."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._checks = 0

    def check(self, request_shape, dtype):
        super().check(request_shape, dtype)
        self._checks += 1
        if self._checks % 2 == 0:
            raise IncompatibleInputError("rejected sample")


class ExecutorTests(unittest.TestCase):
    def test_finished_future_reports_batch_and_model_attribution(self):
        model = LinearModel(version="linear-v42")
        config = ExecutorConfig(max_batch_size=4, max_wait_time=0.05)
        with DynamicBatchingExecutor(model, config) as executor:
            first = executor.submit(linear_sample((1.0, 2.0, 3.0, 4.0)))
            second = executor.submit(linear_sample((5.0, 6.0, 7.0, 8.0)))
            first.result(1)
            second.result(1)

        self.assertEqual(first.info.state, RequestState.FINISHED)
        self.assertEqual(second.info.state, RequestState.FINISHED)
        self.assertEqual(first.info.batch_id, 0)
        self.assertEqual(second.info.batch_id, 0)
        self.assertEqual(first.info.model_version, "linear-v42")
        self.assertEqual(second.info.model_version, "linear-v42")
        self.assertEqual(first.info.model_name, model.name)
        self.assertEqual(first.info.batch_index, 0)
        self.assertEqual(second.info.batch_index, 1)
        self.assertTrue(first.info.bound)

    def test_queued_info_is_unbound_and_cancel_is_traceable_without_batch(self):
        model = GateModel(version="old")
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001)
        with DynamicBatchingExecutor(model, config) as executor:
            running = executor.submit(linear_sample((1, 2, 3, 4)), timeout=None)
            self.assertTrue(model.entered.wait(1))
            queued = executor.submit(linear_sample((5, 6, 7, 8)), timeout=None)
            time.sleep(0.02)

            pending = executor.info(queued.request_id)
            self.assertEqual(pending.state, RequestState.QUEUED)
            self.assertFalse(pending.bound)
            self.assertIsNone(pending.batch_id)
            self.assertIsNone(pending.model_version)

            self.assertTrue(executor.cancel(queued.request_id))
            cancelled_info = executor.info(queued.request_id)
            self.assertEqual(cancelled_info.state, RequestState.CANCELLED)
            self.assertIsNone(cancelled_info.batch_id)
            self.assertIsNone(cancelled_info.model_version)
            model.release.set()
            running.result(1)

    def test_terminal_attribution_is_stable_across_repeated_reads(self):
        old = GateModel(version="old")
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001)
        executor = DynamicBatchingExecutor(old, config).start()
        future = executor.submit(linear_sample((1, 2, 3, 4)), timeout=None)
        self.assertTrue(old.entered.wait(1))
        executor.stop(timeout=0.05)

        self.assertTrue(future.done())
        infos = [future.info for _ in range(5)]
        self.assertTrue(all(info == infos[0] for info in infos))
        self.assertEqual(infos[0].state, RequestState.FAILED)
        self.assertEqual(infos[0].batch_id, 0)
        self.assertEqual(infos[0].model_version, "old")
        queried = executor.info(future.request_id)
        self.assertEqual(queried, infos[0])
        old.release.set()

    def test_partial_batch_failure_keeps_shared_batch_identity(self):
        model = PartialFailModel(version="partial")
        config = ExecutorConfig(max_batch_size=2, max_wait_time=0.02)
        with DynamicBatchingExecutor(model, config) as executor:
            first = executor.submit(linear_sample((1, 2, 3, 4)), timeout=None)
            second = executor.submit(linear_sample((5, 6, 7, 8)), timeout=None)
            self.assertTrue(model.entered.wait(1))
            model.release.set()
            first.result(1)
            with self.assertRaises(IncompatibleInputError):
                second.result(1)

        self.assertEqual(first.info.state, RequestState.FINISHED)
        self.assertEqual(second.info.state, RequestState.FAILED)
        self.assertIsNotNone(first.info.batch_id)
        self.assertEqual(first.info.batch_id, second.info.batch_id)
        self.assertEqual(first.info.model_version, "partial")
        self.assertEqual(second.info.model_version, "partial")
        self.assertEqual(first.info.batch_index, 0)
        self.assertEqual(second.info.batch_index, 1)

    def test_reload_attributes_in_flight_batch_to_old_and_new_traffic_to_new(self):
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
            self.assertFalse(reloaded.is_set())
            # The first request is already bound to the old model batch.
            first_info = executor.info(first.request_id)
            self.assertEqual(first_info.model_version, "old")
            self.assertIsNotNone(first_info.batch_id)

            second = executor.submit(linear_sample((5, 6, 7, 8)), timeout=None)
            # Still queued when the old batch is in flight: unbound, so it is
            # eligible for the replacement model.
            pending = executor.info(second.request_id)
            self.assertEqual(pending.state, RequestState.QUEUED)
            self.assertIsNone(pending.model_version)

            old.release.set()
            self.assertTrue(reloaded.wait(1))
            self.assertTrue(new.entered.wait(1))
            new.release.set()
            first.result(1)
            second.result(1)

        self.assertEqual(first.info.model_version, "old")
        self.assertEqual(second.info.model_version, "new")
        self.assertEqual(first.info.batch_id, 0)
        self.assertEqual(second.info.batch_id, 1)

    def test_failed_reload_does_not_change_in_flight_attribution(self):
        old = GateModel(version="old")
        failed_new = GateModel(version="would-be-new")
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001)
        with DynamicBatchingExecutor(old, config) as executor:
            first = executor.submit(linear_sample((1, 2, 3, 4)), timeout=None)
            self.assertTrue(old.entered.wait(1))

            reloaded = threading.Event()

            def reload():
                try:
                    executor.reload_model(failed_new, timeout=0.05)
                except ModelReloadError:
                    reloaded.set()

            threading.Thread(target=reload, daemon=True).start()
            self.assertTrue(reloaded.wait(1))
            self.assertIs(executor.model, old)
            # Attribution captured by the executing batch is untouched.
            self.assertEqual(
                executor.info(first.request_id).model_version, "old"
            )
            old.release.set()
            first.result(1)
            self.assertEqual(first.info.model_version, "old")

    def test_cancel_racing_completion_keeps_single_consistent_terminal_info(self):
        model = GateModel(version="race")
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001)
        with DynamicBatchingExecutor(model, config) as executor:
            future = executor.submit(linear_sample((1, 2, 3, 4)), timeout=None)
            self.assertTrue(model.entered.wait(1))
            model.release.set()
            future.result(1)
            # Completion won; a late cancel cannot rewrite the terminal info.
            self.assertFalse(executor.cancel(future.request_id))

        info = future.info
        self.assertEqual(info.state, RequestState.FINISHED)
        self.assertEqual(info.model_version, "race")
        self.assertEqual(future.info, info)
        states = {snap.state for snap in future.history}
        self.assertNotIn(RequestState.CANCELLED, states)

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

    def test_high_priority_served_before_queued_normal(self):
        model = GateModel()
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001)
        with DynamicBatchingExecutor(model, config) as executor:
            blocker = executor.submit(linear_sample((0, 0, 0, 0)), timeout=None)
            self.assertTrue(model.entered.wait(1))
            normal = executor.submit(linear_sample((1, 1, 1, 1)), timeout=None)
            high = executor.submit(
                Request(data=(2, 2, 2, 2), shape=(4,), priority=Priority.HIGH),
                timeout=None,
            )
            time.sleep(0.03)
            model.release.set()
            blocker.result(1)
            high.result(1)
            normal.result(1)
        served = [batch[0] for batch in model.batches]
        self.assertEqual(served, [(0, 0, 0, 0), (2, 2, 2, 2), (1, 1, 1, 1)])

    def test_aging_prevents_starvation_under_continuous_high_priority(self):
        model = GateModel()
        config = ExecutorConfig(
            max_batch_size=1, max_wait_time=0.001, aging_interval=0.05
        )
        with DynamicBatchingExecutor(model, config) as executor:
            blocker = executor.submit(linear_sample((0, 0, 0, 0)), timeout=None)
            self.assertTrue(model.entered.wait(1))
            normal = executor.submit(linear_sample((1, 1, 1, 1)), timeout=None)
            time.sleep(0.12)  # normal ages up to effective HIGH
            highs = [
                executor.submit(
                    Request(
                        data=(float(i),) * 4, shape=(4,), priority=Priority.HIGH
                    ),
                    timeout=None,
                )
                for i in range(2, 5)
            ]
            time.sleep(0.03)
            model.release.set()
            blocker.result(1)
            normal.result(1)
            for future in highs:
                future.result(1)
        served = [batch[0] for batch in model.batches]
        self.assertEqual(served[0], (0, 0, 0, 0))
        # Aged normal request wins the tie against newer HIGH requests.
        self.assertEqual(served[1], (1, 1, 1, 1))
        self.assertEqual(normal.state, RequestState.FINISHED)

    def test_stop_fails_queued_and_in_flight_requests(self):
        model = GateModel()
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001)
        executor = DynamicBatchingExecutor(model, config).start()
        running = executor.submit(linear_sample((1, 2, 3, 4)), timeout=None)
        self.assertTrue(model.entered.wait(1))
        queued = executor.submit(linear_sample((5, 6, 7, 8)), timeout=None)
        time.sleep(0.02)
        executor.stop(timeout=0.05)
        for future in (running, queued):
            self.assertTrue(future.done())
            self.assertEqual(future.state, RequestState.FAILED)
            with self.assertRaises(EngineStoppedError):
                future.result(0.1)
        model.release.set()

    def test_queue_full_rejection_reports_capacity(self):
        model = GateModel()
        config = ExecutorConfig(
            max_queue_size=1, max_batch_size=1, max_wait_time=0.001
        )
        with DynamicBatchingExecutor(model, config) as executor:
            executor.submit(linear_sample((1, 2, 3, 4)), timeout=None)
            self.assertTrue(model.entered.wait(1))
            executor.submit(linear_sample((5, 6, 7, 8)), timeout=None)
            with self.assertRaises(QueueFullError) as ctx:
                executor.submit(linear_sample((9, 10, 11, 12)), timeout=None)
            self.assertIn("1/1", str(ctx.exception))
            model.release.set()

    def test_reload_failure_preserves_batches_and_terminal_states(self):
        old = GateModel(version="old")
        config = ExecutorConfig(max_batch_size=2, max_wait_time=0.01)
        with DynamicBatchingExecutor(old, config) as executor:
            def fail_loader():
                raise RuntimeError("corrupt weights")

            with self.assertRaises(ModelReloadError) as ctx:
                executor.reload_model(fail_loader)
            self.assertIn("corrupt weights", str(ctx.exception))
            self.assertIs(executor.model, old)

            first = executor.submit(linear_sample((1, 2, 3, 4)), timeout=None)
            second = executor.submit(linear_sample((5, 6, 7, 8)), timeout=None)
            self.assertTrue(old.entered.wait(1))
            old.release.set()
            self.assertEqual(first.result(1), ("old", 0))
            self.assertEqual(second.result(1), ("old", 1))
            self.assertEqual(first.state, RequestState.FINISHED)
            self.assertEqual(second.state, RequestState.FINISHED)
        self.assertEqual(old.batches, [[(1, 2, 3, 4), (5, 6, 7, 8)]])


if __name__ == "__main__":
    unittest.main()
