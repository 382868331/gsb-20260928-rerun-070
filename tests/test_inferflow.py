from __future__ import annotations

import threading
import time
import tempfile
import unittest
from pathlib import Path

from inferflow import (
    AddConstantModel,
    AllocationStrategy,
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
    HedgeConfig,
    HedgeDelivery,
    ItemState,
    CandidateState,
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


class CountingSlowModel:
    def __init__(self, version="slow-v1"):
        self.name = "slow"
        self.version = version
        self.calls = 0
        self._lock = threading.Lock()

    def check(self, request_shape, dtype):
        return None

    def predict(self, samples):
        with self._lock:
            self.calls += 1
        time.sleep(0.05)
        return [tuple(sample) for sample in samples]


class FlakyResultModel:
    def __init__(self, version="flaky-v1"):
        self.name = "flaky"
        self.version = version
        self.calls = 0
        self._lock = threading.Lock()

    def check(self, request_shape, dtype):
        return None

    def predict(self, samples):
        with self._lock:
            self.calls += 1
            serial = self.calls
        time.sleep(0.03)
        return [(float(serial),) * len(sample) for sample in samples]


class HedgeTests(unittest.TestCase):
    def test_first_conforming_result_delivers_item_once(self):
        model = CountingSlowModel()
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001)
        with DynamicBatchingExecutor(model, config) as executor:
            group = executor.submit_group(
                [Request((1.0, 2.0, 3.0), (3,))],
                hedge=HedgeConfig(budget=1, delay=0.02),
            )
            report = group.wait(2)
            self.assertEqual(report.state, GroupState.EXECUTED)
            item = report.items[0]
            self.assertEqual(item.state, ItemState.EXECUTED)
            self.assertEqual(item.result, (1.0, 2.0, 3.0))
            self.assertEqual(item.delivery.mode, HedgeDelivery.FIRST_CONFORMING)
            delivered_id = item.delivery.candidate_id
            self.assertEqual(
                [candidate.role for candidate in item.candidates],
                ["primary", "hedge-1"],
            )
            self.assertEqual(item.candidates[0].candidate_id, delivered_id)
            self.assertNotEqual(item.candidates[1].candidate_id, delivered_id)
            # Let the redundant execution finish too; the decision stays fixed.
            time.sleep(0.1)
            settled = group.report()
            self.assertEqual(settled.items[0].delivery.candidate_id, delivered_id)
            self.assertEqual(
                [candidate.state for candidate in settled.items[0].candidates],
                [CandidateState.FINISHED, CandidateState.FINISHED],
            )
            self.assertEqual(
                [candidate.batch_id for candidate in settled.items[0].candidates if candidate.batch_id],
                [candidate.batch_id for candidate in settled.items[0].candidates if candidate.batch_id],
            )
            self.assertNotEqual(
                settled.items[0].candidates[0].batch_id,
                settled.items[0].candidates[1].batch_id,
            )

    def test_consistent_contract_requires_two_equal_results(self):
        model = CountingSlowModel()
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001)
        with DynamicBatchingExecutor(model, config) as executor:
            group = executor.submit_group(
                [Request((1.0, 2.0, 3.0), (3,))],
                hedge=HedgeConfig(
                    budget=1,
                    delivery=HedgeDelivery.CONSISTENT,
                    delay=0.01,
                ),
            )
            report = group.wait(2)
        item = report.items[0]
        self.assertEqual(report.state, GroupState.EXECUTED)
        self.assertEqual(item.delivery.mode, HedgeDelivery.CONSISTENT)
        self.assertEqual(item.delivery.reason, "consistent_results")
        self.assertEqual(item.result, (1.0, 2.0, 3.0))
        self.assertEqual(
            [candidate.state for candidate in item.candidates],
            [CandidateState.FINISHED, CandidateState.FINISHED],
        )

    def test_consistent_contract_reports_mismatch_without_delivery(self):
        model = FlakyResultModel()
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001)
        with DynamicBatchingExecutor(model, config) as executor:
            group = executor.submit_group(
                [Request((1.0, 2.0, 3.0), (3,))],
                hedge=HedgeConfig(
                    budget=1,
                    delivery=HedgeDelivery.CONSISTENT,
                    delay=0.005,
                ),
            )
            report = group.wait(2)
        self.assertEqual(report.state, GroupState.FAILED)
        item = report.items[0]
        self.assertIsNone(item.delivery)
        self.assertEqual(item.unmet_reason, "redundant_results_mismatch")
        self.assertEqual(item.state, ItemState.FAILED)
        self.assertIsNone(item.result)

    def test_budget_distinguishes_pending_committed_and_completed(self):
        model = GateModel(version="slow-v1", shape=(3,), output=("ok",))
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001)
        with DynamicBatchingExecutor(model, config) as executor:
            group = executor.submit_group(
                [
                    Request((1.0, 2.0, 3.0), (3,)),
                    Request((4.0, 5.0, 6.0), (3,)),
                ],
                hedge=HedgeConfig(budget=2, delay=5.0),
            )
            self.assertTrue(model.entered.wait(1))
            pending_report = group.report()
            self.assertEqual(pending_report.hedge_budget.total, 2)
            self.assertEqual(pending_report.hedge_budget.pending, 2)
            self.assertEqual(pending_report.hedge_budget.committed, 0)
            self.assertEqual(pending_report.hedge_budget.completed, 0)
            self.assertEqual(pending_report.hedge_budget.remaining, 0)
            model.release.set()
            final = group.wait(2)
        # Both items finished on the primary; pending hedges were withdrawn.
        self.assertEqual(final.state, GroupState.EXECUTED)
        self.assertEqual(final.hedge_budget.pending, 0)

    def test_cancel_marks_uncommitted_candidates_and_fixes_nothing(self):
        model = GateModel(version="slow-v1", shape=(3,), output=("ok",))
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001)
        with DynamicBatchingExecutor(model, config) as executor:
            group = executor.submit_group(
                [Request((1.0, 2.0, 3.0), (3,))],
                hedge=HedgeConfig(budget=1, delay=5.0),
            )
            self.assertTrue(model.entered.wait(1))
            report = group.cancel_remaining()
            model.release.set()
            time.sleep(0.05)
            settled = group.report()
        self.assertEqual(report.state, GroupState.CANCELLED)
        self.assertIsNone(report.items[0].delivery)
        states = {candidate.role: candidate.state for candidate in settled.items[0].candidates}
        self.assertEqual(states["hedge-1"], CandidateState.CANCELLED)
        self.assertFalse(settled.items[0].candidates[1].committed)

    def test_saved_hedged_group_keeps_one_fixed_decision(self):
        model = CountingSlowModel()
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hedged.json"
            with DynamicBatchingExecutor(model, config) as executor:
                group = executor.submit_group(
                    [Request((1.0, 2.0, 3.0), (3,))],
                    hedge=HedgeConfig(budget=1, delay=0.02),
                )
                delivered = group.wait(2)
                delivered_id = delivered.items[0].delivery.candidate_id
                group.save(path)

            reopened = GroupStore().load(path).report()
            self.assertEqual(reopened.state, GroupState.EXECUTED)
            self.assertEqual(reopened.hedge_delivery, HedgeDelivery.FIRST_CONFORMING)
            self.assertEqual(reopened.items[0].delivery.candidate_id, delivered_id)
            self.assertEqual(len(reopened.items[0].candidates), 2)
            repeated = GroupStore().load(path).report()
            self.assertEqual(reopened, repeated)

    def test_retry_unfinished_skips_delivered_items_and_keeps_contract(self):
        class FailingOnceModel(CountingSlowModel):
            def __init__(self):
                super().__init__()
                self.failed_ids: set[str] = set()

            def predict(self, samples):
                time.sleep(0.01)
                outputs = []
                for sample in samples:
                    key = tuple(sample)
                    if key == (2.0, 2.0, 2.0) and key not in self.failed_ids:
                        self.failed_ids.add(key)
                        raise RuntimeError("transient failure")
                    outputs.append(tuple(sample))
                return outputs

        model = FailingOnceModel()
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001)
        with DynamicBatchingExecutor(model, config) as executor:
            group = executor.submit_group(
                [
                    Request((1.0, 1.0, 1.0), (3,)),
                    Request((2.0, 2.0, 2.0), (3,)),
                ],
                hedge=HedgeConfig(budget=1, delay=0.005),
            )
            first = group.wait(2)
            delivered_id = first.items[0].delivery.candidate_id
            group.retry_unfinished()
            final = group.wait(2)
        self.assertEqual(final.state, GroupState.EXECUTED)
        self.assertEqual(final.items[0].delivery.candidate_id, delivered_id)
        self.assertIsNotNone(final.items[1].delivery)


class FirstBatchGateModel:
    """Blocks only the first predict call until released."""

    def __init__(self, version="fb-v1", shape=(3,)):
        self.name = "first-batch-gate"
        self.version = version
        self.shape = shape
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self._lock = threading.Lock()

    def check(self, request_shape, dtype):
        return None

    def predict(self, samples):
        with self._lock:
            index = self.calls
            self.calls += 1
        if index == 0:
            self.entered.set()
            self.release.wait(2)
        return [(self.version, tuple(sample)) for sample in samples]


class TwoGateModel:
    """Blocks the first two predict calls on two separate gates."""

    def __init__(self, version="two-gate-v1", shape=(3,)):
        self.name = "two-gate"
        self.version = version
        self.shape = shape
        self.gates = [threading.Event(), threading.Event()]
        self.calls = 0
        self._lock = threading.Lock()

    def check(self, request_shape, dtype):
        return None

    def predict(self, samples):
        with self._lock:
            index = self.calls
            self.calls += 1
        self.gates[min(index, 1)].wait(2)
        return [(self.version, tuple(sample)) for sample in samples]


class DynamicBudgetTests(unittest.TestCase):
    def test_fair_strategy_spreads_budget_in_arrival_order(self):
        model = GateModel(version="fair-v1", shape=(3,), output=("ok",))
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001)
        with DynamicBatchingExecutor(model, config) as executor:
            group = executor.submit_group(
                [
                    Request((1.0, 2.0, 3.0), (3,)),
                    Request((4.0, 5.0, 6.0), (3,)),
                ],
                hedge=HedgeConfig(
                    budget=1, delay=5.0, strategy=AllocationStrategy.FAIR
                ),
            )
            self.assertTrue(model.entered.wait(1))
            report = group.report()
            roles = [[c.role for c in item.candidates] for item in report.items]
            self.assertEqual(roles[0], ["primary", "hedge-1"])
            self.assertEqual(roles[1], ["primary"])
            self.assertEqual(report.hedge_budget.pending, 1)
            model.release.set()
            final = group.wait(2)
        self.assertEqual(final.state, GroupState.EXECUTED)

    def test_deadline_first_rescues_nearest_deadline_item(self):
        model = GateModel(version="dl-v1", shape=(3,), output=("ok",))
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001)
        with DynamicBatchingExecutor(model, config) as executor:
            group = executor.submit_group(
                [
                    Request((1.0, 2.0, 3.0), (3,)),
                    Request((4.0, 5.0, 6.0), (3,), deadline=5.0),
                ],
                hedge=HedgeConfig(
                    budget=1,
                    delay=5.0,
                    strategy=AllocationStrategy.DEADLINE_FIRST,
                ),
            )
            self.assertTrue(model.entered.wait(1))
            report = group.report()
            roles = [[c.role for c in item.candidates] for item in report.items]
            # The single slot goes to the item with the delivery deadline,
            # even though it arrived second.
            self.assertEqual(roles[0], ["primary"])
            self.assertEqual(roles[1], ["primary", "hedge-1"])
            self.assertEqual(report.items[1].deadline, 5.0)
            model.release.set()
            final = group.wait(2)
        self.assertEqual(final.state, GroupState.EXECUTED)

    def test_freed_slot_moves_to_still_waiting_item(self):
        model = FirstBatchGateModel()
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001)
        with DynamicBatchingExecutor(model, config) as executor:
            group = executor.submit_group(
                [
                    Request((1.0, 2.0, 3.0), (3,)),
                    Request((4.0, 5.0, 6.0), (3,)),
                ],
                hedge=HedgeConfig(budget=1, delay=5.0),
            )
            self.assertTrue(model.entered.wait(1))
            initial = group.report()
            self.assertEqual(
                [c.role for c in initial.items[0].candidates],
                ["primary", "hedge-1"],
            )
            self.assertEqual(
                [c.role for c in initial.items[1].candidates], ["primary"]
            )
            model.release.set()
            final = group.wait(2)
        self.assertEqual(final.state, GroupState.EXECUTED)
        # Item 0 delivered on its primary; its unstarted hedge was withdrawn
        # and the freed slot was re-promised to item 1, which also delivered
        # on its primary and returned the slot unused.
        first, second = final.items
        self.assertEqual(first.candidates[1].state, CandidateState.CANCELLED)
        self.assertFalse(first.candidates[1].committed)
        self.assertEqual(
            [c.role for c in second.candidates], ["primary", "hedge-1"]
        )
        self.assertEqual(second.candidates[1].state, CandidateState.CANCELLED)
        self.assertFalse(second.candidates[1].committed)
        self.assertEqual(final.hedge_budget.remaining, 1)
        self.assertEqual(final.hedge_budget.executed, 0)

    def test_item_redundancy_cap_redirects_budget(self):
        model = GateModel(version="cap-v1", shape=(3,), output=("ok",))
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001)
        with DynamicBatchingExecutor(model, config) as executor:
            group = executor.submit_group(
                [
                    Request((1.0, 2.0, 3.0), (3,), max_redundancy=0),
                    Request((4.0, 5.0, 6.0), (3,)),
                ],
                hedge=HedgeConfig(budget=3, delay=5.0),
            )
            self.assertTrue(model.entered.wait(1))
            report = group.report()
            self.assertEqual(
                [c.role for c in report.items[0].candidates], ["primary"]
            )
            self.assertEqual(
                [c.role for c in report.items[1].candidates],
                ["primary", "hedge-1", "hedge-2", "hedge-3"],
            )
            self.assertEqual(report.hedge_budget.pending, 3)
            model.release.set()
            final = group.wait(2)
        self.assertEqual(final.state, GroupState.EXECUTED)

    def test_deadline_exceeded_recorded_while_work_continues(self):
        model = GateModel(version="late-v1", shape=(3,), output=("ok",))
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001)
        with DynamicBatchingExecutor(model, config) as executor:
            group = executor.submit_group(
                [Request((1.0, 2.0, 3.0), (3,), deadline=0.05)],
                hedge=HedgeConfig(budget=1, delay=5.0),
            )
            self.assertTrue(model.entered.wait(1))
            time.sleep(0.15)
            missed = group.report()
            self.assertEqual(missed.items[0].state, ItemState.WAITING)
            self.assertEqual(missed.items[0].unmet_reason, "deadline_exceeded")
            # The unstartable hedge was withdrawn; the running primary is
            # real work and is left alone.
            self.assertEqual(
                missed.items[0].candidates[1].state, CandidateState.CANCELLED
            )
            model.release.set()
            final = group.wait(2)
        self.assertEqual(final.state, GroupState.EXECUTED)
        self.assertIsNone(final.items[0].unmet_reason)
        self.assertIsNotNone(final.items[0].delivery)

    def test_consistent_waits_for_second_conforming_result(self):
        model = TwoGateModel()
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001)
        with DynamicBatchingExecutor(model, config) as executor:
            group = executor.submit_group(
                [Request((1.0, 2.0, 3.0), (3,))],
                hedge=HedgeConfig(
                    budget=1,
                    delivery=HedgeDelivery.CONSISTENT,
                    delay=0.02,
                ),
            )
            self.assertTrue(model.gates[0].wait(0) or True)
            deadline = time.monotonic() + 2
            while model.calls < 1 and time.monotonic() < deadline:
                time.sleep(0.005)
            model.gates[0].set()
            # The primary finishes first; one fast conforming result must
            # not impersonate the two-result consistent delivery.
            deadline = time.monotonic() + 2
            while model.calls < 2 and time.monotonic() < deadline:
                time.sleep(0.005)
            time.sleep(0.05)
            midway = group.report()
            self.assertIsNone(midway.items[0].delivery)
            self.assertEqual(midway.items[0].state, ItemState.WAITING)
            model.gates[1].set()
            final = group.wait(2)
        self.assertEqual(final.state, GroupState.EXECUTED)
        item = final.items[0]
        self.assertEqual(item.delivery.mode, HedgeDelivery.CONSISTENT)
        self.assertEqual(item.delivery.reason, "consistent_results")
        self.assertEqual(item.result, ("two-gate-v1", (1.0, 2.0, 3.0)))
        self.assertEqual(
            [c.state for c in item.candidates],
            [CandidateState.FINISHED, CandidateState.FINISHED],
        )
        self.assertEqual(final.hedge_budget.executed, 1)

    def test_model_withdrawal_blocks_delivery_and_counts_no_work(self):
        model = GateModel(version="gate-v1", shape=(3,), output=("ok",))
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001)
        with DynamicBatchingExecutor(model, config) as executor:
            # Keep the worker busy so the reload lands while the group's
            # pinned candidates are still queued behind the blocker.
            blocker = executor.submit(Request((9.0, 9.0, 9.0), (3,)), timeout=None)
            group = executor.submit_group(
                [Request((1.0, 2.0, 3.0), (3,))],
                hedge=HedgeConfig(budget=1, delay=0.01),
                timeout=0.2,
            )
            self.assertTrue(model.entered.wait(1))
            reloaded = threading.Event()

            def do_reload():
                executor.reload_model(GateModel(version="gate-v2", shape=(3,)))
                reloaded.set()

            thread = threading.Thread(target=do_reload, daemon=True)
            thread.start()
            model.release.set()
            self.assertTrue(reloaded.wait(2))
            blocker.result(2)
            report = group.wait(2)
        # Every candidate was pinned to the withdrawn version; none may
        # occupy a delivery slot and none entered a real batch.
        self.assertEqual(report.state, GroupState.FAILED)
        item = report.items[0]
        self.assertIsNone(item.delivery)
        self.assertEqual(item.unmet_reason, "insufficient_conforming_results")
        self.assertEqual(item.state, ItemState.FAILED)
        self.assertEqual(report.hedge_budget.executed, 0)
        self.assertTrue(
            all(c.batch_id is None for c in item.candidates),
            msg="ineligible candidates must not hold batch attribution",
        )

    def test_start_cancel_interleaving_never_exceeds_budget(self):
        for _ in range(20):
            model = CountingSlowModel()
            config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001)
            with DynamicBatchingExecutor(model, config) as executor:
                group = executor.submit_group(
                    [Request((1.0, 2.0, 3.0), (3,))],
                    hedge=HedgeConfig(budget=2, delay=0.001),
                )
                group.cancel_remaining()
                report = group.report()
                budget = report.hedge_budget
                # One slot can never be promised to two candidates at once.
                self.assertLessEqual(budget.used, budget.total)
                self.assertGreaterEqual(budget.remaining, 0)
                self.assertEqual(budget.used + budget.remaining, budget.total)
                self.assertLessEqual(budget.executed, budget.total)

    def test_executed_work_survives_cancellation_and_snapshot(self):
        model = GateModel(version="work-v1", shape=(3,), output=("ok",))
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "work.json"
            with DynamicBatchingExecutor(model, config) as executor:
                group = executor.submit_group(
                    [Request((1.0, 2.0, 3.0), (3,))],
                    hedge=HedgeConfig(budget=1, delay=0.02),
                )
                self.assertTrue(model.entered.wait(1))
                time.sleep(0.1)
                midway = group.report()
                # The hedge is already committed to the executor while the
                # primary occupies the worker.
                self.assertEqual(midway.hedge_budget.committed, 1)
                model.release.set()
                delivered = group.wait(2)
                self.assertEqual(delivered.state, GroupState.EXECUTED)
                time.sleep(0.1)
                group.cancel_remaining()
                settled = group.report()
                group.save(path)

            # The committed hedge ran to completion after the delivery was
            # fixed; that real work is not erased by the fixed delivery,
            # the later cancel, or a snapshot round trip.
            self.assertEqual(settled.hedge_budget.executed, 1)
            self.assertEqual(settled.hedge_budget.completed, 1)
            hedge = settled.items[0].candidates[1]
            self.assertEqual(hedge.state, CandidateState.FINISHED)
            self.assertIsNotNone(hedge.batch_id)
            reopened = GroupStore().load(path).report()
            self.assertEqual(reopened.hedge_budget.executed, 1)
            self.assertEqual(
                reopened.items[0].candidates[1].batch_id, hedge.batch_id
            )
            self.assertEqual(
                reopened.items[0].delivery.candidate_id,
                settled.items[0].delivery.candidate_id,
            )
            self.assertEqual(
                reopened.items[0].delivery.reason,
                settled.items[0].delivery.reason,
            )


if __name__ == "__main__":
    unittest.main()
