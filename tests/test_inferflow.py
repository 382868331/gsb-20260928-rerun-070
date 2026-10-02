from __future__ import annotations

import threading
import time
import tempfile
import unittest
from pathlib import Path

from inferflow import (
    AddConstantModel,
    DynamicBatchingExecutor,
    EngineStoppedError,
    ExecutorConfig,
    IncompatibleInputError,
    LinearModel,
    ModelReloadError,
    Priority,
    QueueFullError,
    Request,
    RequestCancelledError,
    RequestState,
    RequestTimeoutError,
    GroupState,
    GroupStore,
    ItemState,
    VersionContractError,
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


class VersionedShapeModel:
    def __init__(self, version):
        self.name = "versioned-shape"
        self.version = version

    def check(self, request_shape, dtype):
        if self.version == "old" and tuple(request_shape) != (3,):
            raise IncompatibleInputError("old version only accepts shape (3,)")

    def predict(self, samples):
        return [(self.version, tuple(sample)) for sample in samples]


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


class AttributionTests(unittest.TestCase):
    def test_finished_requests_share_batch_identity_and_version(self):
        model = RecordingLinearModel()
        config = ExecutorConfig(max_batch_size=4, max_wait_time=0.05)
        samples = [(0.1, 0.2, 0.3, 0.4), (1.0, 2.0, 3.0, 4.0)]
        with DynamicBatchingExecutor(model, config) as executor:
            futures = [executor.submit(linear_sample(sample)) for sample in samples]
            for future in futures:
                future.result(1)
        attributions = [future.attribution for future in futures]
        self.assertTrue(all(item is not None for item in attributions))
        self.assertEqual(
            {item.batch_id for item in attributions}, {attributions[0].batch_id}
        )
        for index, attribution in enumerate(attributions):
            self.assertEqual(attribution.model_name, model.name)
            self.assertEqual(attribution.model_version, model.version)
            self.assertEqual(attribution.batch_size, 2)
            self.assertEqual(attribution.batch_index, index)
        for future in futures:
            self.assertEqual(
                future.history,
                (RequestState.QUEUED, RequestState.RUNNING, RequestState.FINISHED),
            )


class RequestGroupTests(unittest.TestCase):
    def test_same_version_group_reports_each_item_and_pins_version(self):
        model = AddConstantModel(input_shape=(3,), version="old")
        config = ExecutorConfig(max_batch_size=2, max_wait_time=0.02)
        with DynamicBatchingExecutor(model, config) as executor:
            group = executor.submit_group(
                [
                    Request((1.0, 2.0, 3.0), (3,)),
                    Request((4.0, 5.0, 6.0), (3,)),
                ]
            )
            report = group.wait(1)

        self.assertEqual(report.state, GroupState.EXECUTED)
        self.assertTrue(report.complete)
        self.assertTrue(report.version_conformant)
        self.assertEqual(report.pinned_version, "old")
        self.assertEqual(report.executed_versions, ("old",))
        self.assertEqual([item.state for item in report.items], [ItemState.EXECUTED] * 2)
        self.assertTrue(all(item.attribution.model_version == "old" for item in report.items))

    def test_current_version_contract_allows_reload_boundary_split(self):
        old = GateModel(version="old")
        new = GateModel(version="new")
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001)
        with DynamicBatchingExecutor(old, config) as executor:
            group = executor.submit_group(
                [
                    linear_sample((1, 2, 3, 4)),
                    linear_sample((5, 6, 7, 8)),
                ],
                same_version=False,
            )
            self.assertTrue(old.entered.wait(1))
            reloaded = threading.Event()

            def reload():
                executor.reload_model(new, timeout=2)
                reloaded.set()

            threading.Thread(target=reload, daemon=True).start()
            time.sleep(0.03)
            old.release.set()
            self.assertTrue(new.entered.wait(1))
            new.release.set()
            report = group.wait(2)

        self.assertTrue(reloaded.is_set())
        self.assertEqual(report.state, GroupState.EXECUTED)
        self.assertTrue(report.version_conformant)
        self.assertEqual(report.executed_versions, ("old", "new"))
        self.assertEqual(report.items[0].attribution.model_version, "old")
        self.assertEqual(report.items[1].attribution.model_version, "new")

    def test_pinned_retry_is_rejected_then_accepted_after_explicit_confirmation(self):
        old = VersionedShapeModel("old")
        new = VersionedShapeModel("new")
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.01)
        with DynamicBatchingExecutor(old, config) as executor:
            group = executor.submit_group(
                [
                    Request((1.0, 2.0, 3.0), (3,)),
                    Request((4.0, 5.0), (2,)),
                ]
            )
            failed_report = group.wait(1)
            self.assertEqual(failed_report.state, GroupState.FAILED)
            executor.reload_model(new, timeout=1)

            with self.assertRaises(VersionContractError):
                group.retry_unfinished()
            rejected = group.refresh()
            self.assertEqual(rejected.state, GroupState.FAILED)
            self.assertEqual(rejected.items[0].state, ItemState.EXECUTED)
            self.assertEqual(rejected.items[0].result, ("old", (1.0, 2.0, 3.0)))
            self.assertEqual(rejected.items[0].attribution.model_version, "old")
            self.assertEqual(rejected.items[1].state, ItemState.FAILED)

            confirmed = group.retry_unfinished(allow_version_switch=True)
            self.assertEqual(confirmed.items[0].state, ItemState.EXECUTED)
            self.assertEqual(confirmed.items[1].state, ItemState.WAITING)
            final_report = group.wait(1)

        self.assertEqual(final_report.state, GroupState.EXECUTED)
        self.assertEqual(final_report.items[1].result, ("new", (4.0, 5.0)))
        self.assertEqual(final_report.items[1].attribution.model_version, "new")
        self.assertEqual(
            [attempt.model_version for attempt in final_report.items[1].attempts],
            ["old", "new"],
        )

    def test_cancel_remaining_preserves_completed_and_failed_records(self):
        class CancelGateModel:
            def __init__(self):
                self.name = "versioned-shape"
                self.version = "old"
                self.entered = threading.Event()
                self.release = threading.Event()
                self.predictions = 0

            def check(self, request_shape, dtype):
                if tuple(request_shape) != (3,):
                    raise IncompatibleInputError("only shape (3,) succeeds")

            def predict(self, samples):
                self.predictions += 1
                if self.predictions == 2:
                    self.entered.set()
                    self.release.wait(2)
                return [(self.version, tuple(sample)) for sample in samples]

        old = CancelGateModel()
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001)
        with DynamicBatchingExecutor(old, config) as executor:
            group = executor.submit_group(
                [
                    Request((1.0, 2.0, 3.0), (3,)),
                    Request((4.0, 5.0), (2,)),
                    Request((7.0, 8.0, 9.0), (3,)),
                ]
            )
            self.assertTrue(old.entered.wait(1))
            report = group.cancel_remaining()
            old.release.set()

        self.assertEqual(report.state, GroupState.FAILED)
        self.assertEqual([item.state for item in report.items], [
            ItemState.EXECUTED,
            ItemState.FAILED,
            ItemState.CANCELLED,
        ])
        self.assertEqual(report.items[0].attribution.model_version, "old")
        self.assertIsNotNone(report.items[2].attribution)

    def test_stop_marks_waiting_group_failed_with_stable_attribution(self):
        model = GateModel(version="old")
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001)
        executor = DynamicBatchingExecutor(model, config).start()
        group = executor.submit_group(
            [
                linear_sample((1, 2, 3, 4)),
                linear_sample((5, 6, 7, 8)),
            ]
        )
        self.assertTrue(model.entered.wait(1))
        executor.stop(timeout=0.05)
        report = group.refresh()
        self.assertEqual([item.state for item in report.items], [ItemState.FAILED] * 2)
        self.assertEqual(report.state, GroupState.FAILED)
        self.assertEqual(report.items[0].attribution.model_version, "old")
        self.assertIsNone(report.items[1].attribution)
        model.release.set()

    def test_saved_failed_group_reopens_without_completion_contradiction(self):
        old = VersionedShapeModel("old")
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.01)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "group.json"
            with DynamicBatchingExecutor(old, config) as executor:
                group = executor.submit_group(
                    [
                        Request((1.0, 2.0, 3.0), (3,)),
                        Request((4.0, 5.0), (2,)),
                    ]
                )
                group.wait(1)
                saved = group.save(path)
                self.assertEqual(saved.state, GroupState.FAILED)

            reopened = GroupStore().load(path)
            first_read = reopened.report()
            second_read = reopened.report()

        self.assertFalse(first_read.complete)
        self.assertEqual(first_read, second_read)
        self.assertEqual(first_read.state, GroupState.FAILED)
        self.assertEqual(first_read.items[0].state, ItemState.EXECUTED)
        self.assertEqual(first_read.items[1].state, ItemState.FAILED)
        self.assertIsNotNone(first_read.items[1].error)

    def test_cancelled_queued_request_has_no_attribution(self):
        model = GateModel()
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001)
        with DynamicBatchingExecutor(model, config) as executor:
            blocker = executor.submit(linear_sample((1, 2, 3, 4)), timeout=None)
            self.assertTrue(model.entered.wait(1))
            queued = executor.submit(linear_sample((5, 6, 7, 8)), timeout=None)
            time.sleep(0.03)
            self.assertTrue(executor.cancel(queued.request_id))
            model.release.set()
            blocker.result(1)
        self.assertIsNone(queued.attribution)
        self.assertEqual(queued.history, (RequestState.QUEUED, RequestState.CANCELLED))
        self.assertIsNotNone(blocker.attribution)

    def test_reload_boundary_splits_attribution_between_versions(self):
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
            self.assertTrue(new.entered.wait(1))
            new.release.set()
            first.result(1)
            second.result(1)
        self.assertEqual(first.attribution.model_version, "old")
        self.assertEqual(second.attribution.model_version, "new")
        self.assertNotEqual(first.attribution.batch_id, second.attribution.batch_id)

    def test_failed_reload_keeps_in_flight_attribution(self):
        old = GateModel(version="old")
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001)
        with DynamicBatchingExecutor(old, config) as executor:
            first = executor.submit(linear_sample((1, 2, 3, 4)), timeout=None)
            self.assertTrue(old.entered.wait(1))

            def fail_loader():
                raise RuntimeError("corrupt weights")

            with self.assertRaises(ModelReloadError):
                executor.reload_model(fail_loader)
            old.release.set()
            first.result(1)
            second = executor.submit(linear_sample((5, 6, 7, 8)), timeout=None)
            second.result(1)
        self.assertEqual(first.attribution.model_version, "old")
        self.assertEqual(second.attribution.model_version, "old")

    def test_partial_batch_failure_keeps_shared_batch_identity(self):
        model = GateModel()
        config = ExecutorConfig(max_batch_size=2, max_wait_time=0.01)
        with DynamicBatchingExecutor(model, config) as executor:
            timed_out = executor.submit(linear_sample((1, 2, 3, 4)), timeout=0.05)
            finished = executor.submit(linear_sample((5, 6, 7, 8)), timeout=None)
            self.assertTrue(model.entered.wait(1))
            time.sleep(0.1)
            model.release.set()
            with self.assertRaises(RequestTimeoutError):
                timed_out.result(1)
            finished.result(1)
        self.assertEqual(timed_out.state, RequestState.TIMED_OUT)
        self.assertEqual(finished.state, RequestState.FINISHED)
        self.assertIsNotNone(timed_out.attribution)
        self.assertIsNotNone(finished.attribution)
        self.assertEqual(timed_out.attribution.batch_id, finished.attribution.batch_id)
        self.assertEqual(timed_out.attribution.batch_index, 0)
        self.assertEqual(finished.attribution.batch_index, 1)

    def test_stop_leaves_stable_records(self):
        model = GateModel()
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001)
        executor = DynamicBatchingExecutor(model, config).start()
        running = executor.submit(linear_sample((1, 2, 3, 4)), timeout=None)
        self.assertTrue(model.entered.wait(1))
        queued = executor.submit(linear_sample((5, 6, 7, 8)), timeout=None)
        time.sleep(0.02)
        executor.stop(timeout=0.05)
        model.release.set()
        self.assertEqual(running.state, RequestState.FAILED)
        self.assertIsNotNone(running.attribution)
        self.assertEqual(running.attribution.model_version, model.version)
        self.assertEqual(queued.state, RequestState.FAILED)
        self.assertIsNone(queued.attribution)
        for future in (running, queued):
            self.assertEqual(future.state, future.state)
            self.assertEqual(future.attribution, future.attribution)
            self.assertEqual(future.history, future.history)

    def test_repeated_reads_never_contradict(self):
        model = GateModel()
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001)
        with DynamicBatchingExecutor(model, config) as executor:
            future = executor.submit(linear_sample((1, 2, 3, 4)), timeout=None)
            self.assertTrue(model.entered.wait(1))
            model.release.set()
            value = future.result(1)
            first_read = (future.state, future.attribution, future.result(1))
            second_read = (future.state, future.attribution, future.result(1))
        self.assertEqual(first_read, second_read)
        self.assertFalse(future.cancel())
        self.assertEqual(future.state, RequestState.FINISHED)
        self.assertEqual(
            future.history,
            (RequestState.QUEUED, RequestState.RUNNING, RequestState.FINISHED),
        )


if __name__ == "__main__":
    unittest.main()
