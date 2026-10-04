from __future__ import annotations

import threading
import time
import tempfile
import unittest
from pathlib import Path

from inferflow import (
    DeliveryMode,
    DynamicBatchingExecutor,
    ExecutorConfig,
    GroupState,
    GroupStore,
    HedgePolicy,
    IncompatibleInputError,
    ItemState,
    Request,
)


def sample(values=(1.0, 2.0, 3.0, 4.0)):
    return Request(data=tuple(values), shape=(len(values),), dtype=float)


class GateModel:
    def __init__(self, version="gate-v1", output=None, shape=(4,)):
        self.name = "gate"
        self.version = version
        self.shape = shape
        self.output = output or (version,)
        self.entered = threading.Event()
        self.release = threading.Event()
        self.batches = []
        self._lock = threading.Lock()

    def check(self, request_shape, dtype):
        if tuple(request_shape) != self.shape:
            raise IncompatibleInputError("shape mismatch")

    def predict(self, samples):
        self.entered.set()
        self.release.wait(5)
        with self._lock:
            self.batches.append(list(samples))
        return [self.output + (index,) for index, _ in enumerate(samples)]


class PrimarySlowModel:
    """Primary (first batch) blocks; the redundant candidate returns fast."""

    def __init__(self, version="gate-v1", shape=(4,)):
        self.name = "gate"
        self.version = version
        self.shape = shape
        self.entered = threading.Event()
        self.release = threading.Event()
        self._lock = threading.Lock()
        self.batches = []

    def check(self, request_shape, dtype):
        if tuple(request_shape) != self.shape:
            raise IncompatibleInputError("shape mismatch")

    def predict(self, samples):
        with self._lock:
            is_primary = not self.entered.is_set()
            if is_primary:
                self.entered.set()
        if is_primary:
            self.release.wait(5)
            return [("primary-result",) for _ in samples]
        self.batches.append(list(samples))
        return [("hedge-result",) for _ in samples]


class ConstantGateModel:
    def __init__(self, value="same", version="const-v1", shape=(4,)):
        self.name = "const-gate"
        self.version = version
        self.shape = shape
        self.value = value
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self._lock = threading.Lock()

    def check(self, request_shape, dtype):
        if tuple(request_shape) != self.shape:
            raise IncompatibleInputError("shape mismatch")

    def predict(self, samples):
        with self._lock:
            self.calls += 1
            call_index = self.calls
        if call_index == 1:
            self.entered.set()
            self.release.wait(5)
        return [(self.value,) for _ in samples]


class FlakyThenOkModel:
    """First execution of the first seen sample fails; later ones succeed."""

    def __init__(self, version="flaky-v1", shape=(4,)):
        self.name = "flaky"
        self.version = version
        self.shape = shape
        self._seen = 0
        self._lock = threading.Lock()

    def check(self, request_shape, dtype):
        if tuple(request_shape) != self.shape:
            raise IncompatibleInputError("shape mismatch")

    def predict(self, samples):
        with self._lock:
            self._seen += 1
            call = self._seen
        if call == 1:
            raise RuntimeError("transient failure")
        return [("recovered",) for _ in samples]


class HedgeTests(unittest.TestCase):
    def test_first_conformant_adopts_fast_hedge_once_without_double_counting(self):
        model = PrimarySlowModel()
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001, worker_count=2)
        with DynamicBatchingExecutor(model, config) as executor:
            group = executor.submit_group(
                [sample()],
                hedge=HedgePolicy(hedge_after=0.05),
            )
            self.assertTrue(model.entered.wait(1))  # primary enters gate
            time.sleep(0.15)
            report = group.wait(2)
            model.release.set()

        self.assertEqual(report.state, GroupState.EXECUTED)
        self.assertEqual(len(report.items), 1)
        item = report.items[0]
        self.assertTrue(item.delivered)
        self.assertEqual(item.adopted_candidate_id, f"{item.item_id}:hedge")
        self.assertEqual(item.result, ("hedge-result",))
        candidate_ids = [candidate.candidate_id for candidate in item.candidates]
        self.assertEqual(candidate_ids, [f"{item.item_id}:primary", f"{item.item_id}:hedge"])
        adopted = [candidate for candidate in item.candidates if candidate.adopted]
        self.assertEqual(len(adopted), 1)
        primary = item.candidates[0]
        self.assertFalse(primary.adopted)
        self.assertTrue(primary.late)
        self.assertNotEqual(
            primary.attribution.batch_id, item.candidates[1].attribution.batch_id
        )
        self.assertEqual(report.redundancy.committed_extra, 1)
        self.assertEqual(report.redundancy.completed_extra, 1)
        self.assertEqual(report.redundancy.pending_extra, 0)
        self.assertEqual(report.redundancy.late_executions, 1)
        self.assertEqual(report.fixed_deliveries, (item.item_id,))
        # The item is still one delivered item, not two.
        self.assertEqual(len([item for item in report.items if item.delivered]), 1)

    def test_version_ineligible_fast_hedge_is_not_silently_adopted(self):
        old = GateModel(version="old", output=("old",))
        new = GateModel(version="new", output=("new",))
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001, worker_count=2)
        with DynamicBatchingExecutor(old, config) as executor:
            group = executor.submit_group(
                [sample()],
                same_version=True,
                hedge=HedgePolicy(hedge_after=0.05),
            )
            self.assertTrue(old.entered.wait(1))  # primary blocks on old
            executor.reload_model(new, timeout=2)
            # The redundant candidate commits and runs on the new version.
            self.assertTrue(new.entered.wait(2))
            new.release.set()
            waiting = group.wait(0.3)
            item = waiting.items[0]
            self.assertFalse(item.delivered)
            self.assertIn("candidate_version_not_conformant", item.unmet_conditions)
            self.assertEqual(item.state, ItemState.WAITING)
            old.release.set()
            report = group.wait(2)

        self.assertEqual(report.state, GroupState.EXECUTED)
        delivered_item = report.items[0]
        self.assertEqual(delivered_item.adopted_candidate_id, f"{delivered_item.item_id}:primary")
        self.assertEqual(delivered_item.result, ("old", 0))
        hedge_report = delivered_item.candidates[1]
        self.assertFalse(hedge_report.adopted)
        self.assertFalse(hedge_report.conformant)

    def test_compare_mode_requires_two_agreeing_results(self):
        model = ConstantGateModel(value="same")
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001, worker_count=2)
        with DynamicBatchingExecutor(model, config) as executor:
            group = executor.submit_group(
                [sample()],
                hedge=HedgePolicy(hedge_after=0.05, delivery=DeliveryMode.COMPARE),
            )
            self.assertTrue(model.entered.wait(1))
            pending = group.refresh()
            self.assertFalse(pending.items[0].delivered)
            self.assertIn("insufficient_conformant_results", pending.items[0].unmet_conditions)
            time.sleep(0.12)
            model.release.set()
            report = group.wait(2)

        self.assertEqual(report.state, GroupState.EXECUTED)
        item = report.items[0]
        self.assertEqual(item.result, ("same",))
        states = [candidate.state.value for candidate in item.candidates]
        self.assertEqual(states, ["finished", "finished"])
        self.assertTrue(item.candidates[0].adopted)
        self.assertTrue(item.candidates[1].verified)

    def test_compare_disagreement_is_reported_not_delivered(self):
        class DisagreeGate:
            def __init__(self):
                self.name = "disagree"
                self.version = "v1"
                self.entered = threading.Event()
                self.release = threading.Event()
                self._call = 0
                self._lock = threading.Lock()

            def check(self, request_shape, dtype):
                if tuple(request_shape) != (4,):
                    raise IncompatibleInputError("shape mismatch")

            def predict(self, samples):
                with self._lock:
                    self._call += 1
                    call = self._call
                if call == 1:
                    self.entered.set()
                    self.release.wait(5)
                    value = "A"
                else:
                    value = "B"
                return [(value,) for _ in samples]

        model = DisagreeGate()
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001, worker_count=2)
        with DynamicBatchingExecutor(model, config) as executor:
            group = executor.submit_group(
                [sample()],
                hedge=HedgePolicy(hedge_after=0.05, delivery=DeliveryMode.COMPARE),
            )
            self.assertTrue(model.entered.wait(1))
            time.sleep(0.12)
            model.release.set()
            report = group.wait(2)

        self.assertEqual(report.state, GroupState.FAILED)
        item = report.items[0]
        self.assertFalse(item.delivered)
        self.assertIn("results_disagree", item.unmet_conditions)

    def test_budget_zero_keeps_pending_candidate_unstarted(self):
        model = GateModel()
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001, worker_count=2)
        with DynamicBatchingExecutor(model, config) as executor:
            group = executor.submit_group(
                [sample()],
                hedge=HedgePolicy(hedge_after=0.02, max_extra_executions=0),
            )
            self.assertTrue(model.entered.wait(1))
            time.sleep(0.1)
            waiting = group.refresh()
            self.assertEqual(waiting.redundancy.committed_extra, 0)
            self.assertEqual(waiting.redundancy.pending_extra, 1)
            self.assertEqual(waiting.redundancy.extra_budget, 0)
            model.release.set()
            report = group.wait(2)

        self.assertEqual(report.state, GroupState.EXECUTED)
        item = report.items[0]
        self.assertEqual(item.candidates[1].state.value, "cancelled")
        self.assertEqual(item.candidates[1].attribution, None)

    def test_cancel_concurrent_with_hedge_marks_committed_work_irrevocable(self):
        model = ConstantGateModel(value="same")
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001, worker_count=2)
        with DynamicBatchingExecutor(model, config) as executor:
            group = executor.submit_group(
                [sample()],
                hedge=HedgePolicy(hedge_after=0.05, delivery=DeliveryMode.COMPARE),
            )
            self.assertTrue(model.entered.wait(1))
            time.sleep(0.12)  # hedge committed and entered a second batch
            snapshot = group.refresh()
            self.assertEqual(snapshot.redundancy.committed_extra, 1)
            report = group.cancel_remaining()
            model.release.set()
            time.sleep(0.05)

        self.assertEqual(report.state, GroupState.CANCELLED)
        self.assertGreaterEqual(report.irrevocable_executions, 1)
        self.assertEqual(report.fixed_deliveries, ())

    def test_late_candidate_result_after_adoption_is_recorded_but_ignored(self):
        model = PrimarySlowModel()
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001, worker_count=2)
        with DynamicBatchingExecutor(model, config) as executor:
            group = executor.submit_group(
                [sample()],
                hedge=HedgePolicy(hedge_after=0.05),
            )
            self.assertTrue(model.entered.wait(1))
            report = group.wait(2)  # hedge wins while primary still runs
            self.assertEqual(report.state, GroupState.EXECUTED)
            adopted = report.items[0].adopted_candidate_id
            model.release.set()  # primary now finishes, too late
            time.sleep(0.1)
            again = group.report()
        self.assertEqual(again.state, GroupState.EXECUTED)
        self.assertEqual(again.items[0].adopted_candidate_id, adopted)
        self.assertEqual(again.fixed_deliveries, (again.items[0].item_id,))
        primary = again.items[0].candidates[0]
        self.assertTrue(primary.late)
        self.assertFalse(primary.adopted)
        # The superseded primary is either cancelled or (if it already entered
        # a batch) finishes afterwards; in both cases its record stays without
        # changing the one adoption.
        self.assertIn(primary.state.value, ("cancelled", "finished"))
        if primary.state.value == "finished":
            self.assertIsNotNone(primary.result)
        self.assertEqual(again.redundancy.late_executions, 1)

    def test_stop_during_hedging_reports_unmet_and_committed_work(self):
        model = ConstantGateModel(value="same")
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001, worker_count=2)
        executor = DynamicBatchingExecutor(model, config).start()
        group = executor.submit_group(
            [sample()],
            hedge=HedgePolicy(hedge_after=0.05, delivery=DeliveryMode.COMPARE),
        )
        self.assertTrue(model.entered.wait(1))
        time.sleep(0.12)
        executor.stop(timeout=0.1)
        model.release.set()
        report = group.report()
        self.assertEqual(report.state, GroupState.FAILED)
        self.assertFalse(report.items[0].delivered)
        self.assertGreaterEqual(report.redundancy.committed_extra, 1)
        self.assertIn("candidate_failed", report.unmet_conditions)

    def test_saved_hedged_group_reopens_with_fixed_adoption_and_queries_stable(self):
        model = GateModel()
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001, worker_count=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hedged.json"
            with DynamicBatchingExecutor(model, config) as executor:
                group = executor.submit_group(
                    [sample()],
                    hedge=HedgePolicy(hedge_after=0.05),
                )
                self.assertTrue(model.entered.wait(1))
                time.sleep(0.12)
                model.release.set()
                report = group.wait(2)
                self.assertEqual(report.state, GroupState.EXECUTED)
                group.save(path)

            reopened = GroupStore().load(path)
            first = reopened.report()
            second = reopened.report()
            self.assertEqual(first, second)
            self.assertEqual(first.state, GroupState.EXECUTED)
            self.assertEqual(first.fixed_deliveries, (first.items[0].item_id,))
            self.assertTrue(first.items[0].delivered)
            self.assertEqual(len(first.items[0].candidates), 2)

    def test_group_budget_is_shared_and_second_item_keeps_pending_candidate(self):
        class SlowFirstModel:
            def __init__(self):
                self.name = "slow-first"
                self.version = "v1"
                self.entered = threading.Event()
                self.release = threading.Event()
                self._call = 0
                self._lock = threading.Lock()

            def check(self, request_shape, dtype):
                if tuple(request_shape) != (4,):
                    raise IncompatibleInputError("bad shape")

            def predict(self, samples):
                with self._lock:
                    self._call += 1
                    first_call = self._call == 1
                if first_call:
                    self.entered.set()
                    self.release.wait(5)
                    return [("slow",) for _ in samples]
                return [("fast",) for _ in samples]

        model = SlowFirstModel()
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.005, worker_count=3)
        with DynamicBatchingExecutor(model, config) as executor:
            group = executor.submit_group(
                [sample((1.0, 2.0, 3.0, 4.0)), sample((5.0, 6.0, 7.0, 8.0))],
                hedge=HedgePolicy(hedge_after=0.05, max_extra_executions=1),
            )
            self.assertTrue(model.entered.wait(1))
            # Second item finishes normally and needs no redundancy.
            time.sleep(0.15)
            second_delivered = group.refresh().items[1].delivered
            self.assertTrue(second_delivered)
            time.sleep(0.2)
            report = group.wait(2)
            model.release.set()

        self.assertEqual(report.state, GroupState.EXECUTED)
        self.assertEqual(report.redundancy.committed_extra, 1)
        self.assertEqual(report.redundancy.pending_extra, 0)
        adopted_hedges = [
            item.item_id for item in report.items if item.adopted_candidate_id.endswith(":hedge")
        ]
        self.assertEqual(len(adopted_hedges), 1)

    def test_primary_failure_is_rescued_by_immediate_hedge(self):
        model = FlakyThenOkModel()
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.001, worker_count=2)
        with DynamicBatchingExecutor(model, config) as executor:
            group = executor.submit_group(
                [sample()],
                hedge=HedgePolicy(hedge_after=5.0),
            )
            report = group.wait(2)

        self.assertEqual(report.state, GroupState.EXECUTED)
        item = report.items[0]
        self.assertTrue(item.delivered)
        self.assertEqual(item.result, ("recovered",))
        self.assertEqual(item.adopted_candidate_id, f"{item.item_id}:hedge")
        self.assertEqual(item.candidates[0].state.value, "failed")
        self.assertEqual(item.candidates[1].start_condition, "hedge_after_primary_failure")

    def test_retry_unfinished_only_reruns_undelivered_items(self):
        # Second item's first primary execution fails; a retry recovers it
        # while the already-delivered first item keeps its single adoption.
        class OneFailureModel:
            def __init__(self):
                self.name = "one-failure"
                self.version = "v1"
                self._call = 0
                self._lock = threading.Lock()

            def check(self, request_shape, dtype):
                if tuple(request_shape) != (4,):
                    raise IncompatibleInputError("bad shape")

            def predict(self, samples):
                with self._lock:
                    self._call += 1
                    call = self._call
                # Second overall batch (the failing item's primary) errors.
                if call == 2:
                    raise RuntimeError("boom")
                return [("ok",) for _ in samples]

        model = OneFailureModel()
        config = ExecutorConfig(max_batch_size=1, max_wait_time=0.005, worker_count=1)
        with DynamicBatchingExecutor(model, config) as executor:
            first = sample((1.0, 2.0, 3.0, 4.0))
            second = sample((5.0, 6.0, 7.0, 8.0))
            group = executor.submit_group(
                [first, second],
                same_version=False,
            )
            failed_report = group.wait(2)
            self.assertEqual(failed_report.state, GroupState.FAILED)
            self.assertTrue(failed_report.items[0].delivered)
            self.assertFalse(failed_report.items[1].delivered)
            first_attribution = failed_report.items[0].attribution
            retried = group.retry_unfinished()
            final_report = group.wait(2)

        self.assertEqual(final_report.state, GroupState.EXECUTED)
        self.assertTrue(all(item.delivered for item in final_report.items))
        # Confirmed item keeps its identity and original adoption decision.
        self.assertEqual(final_report.items[0].attribution, first_attribution)
        self.assertEqual(final_report.items[0].adopted_candidate_id, f"{first.request_id}:primary")
        self.assertTrue(final_report.items[1].delivered)


if __name__ == "__main__":
    unittest.main()
