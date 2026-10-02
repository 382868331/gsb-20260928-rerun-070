from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest

from inferflow import (
    AddConstantModel,
    DynamicBatchingExecutor,
    ExecutorConfig,
    GroupConclusion,
    IncompatibleInputError,
    JsonGroupStore,
    Request,
    RequestGroup,
    RequestState,
    VersionConflictError,
    VersionPolicy,
)


def sample(values) -> Request:
    return Request(data=tuple(values), shape=(len(values),), dtype=float)


class BlockingAddModel(AddConstantModel):
    """Blocks inside predict until released, one release event per call."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.entered = threading.Event()
        self.releases: list[threading.Event] = []
        self._call = 0
        self._lock = threading.Lock()

    def predict(self, samples):
        with self._lock:
            event = threading.Event()
            self.releases.append(event)
            self._call += 1
        self.entered.set()
        event.wait(2)
        return super().predict(samples)

    def release(self, index: int) -> None:
        self.releases[index].set()


class FlakyAddModel(AddConstantModel):
    """Fails the first predict call, then behaves like AddConstantModel."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = 0

    def predict(self, samples):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient backend failure")
        return super().predict(samples)


class GroupContractTests(unittest.TestCase):
    def make_executor(self, model, **overrides):
        config = ExecutorConfig(
            max_batch_size=overrides.get("max_batch_size", 8),
            max_wait_time=overrides.get("max_wait_time", 0.02),
            default_timeout=overrides.get("default_timeout", 5.0),
        )
        return DynamicBatchingExecutor(model, config).start()

    def test_pinned_group_completes_with_uniform_version(self):
        model = AddConstantModel(input_shape=(3,), version="add-v1")
        executor = self.make_executor(model)
        try:
            group = RequestGroup(
                executor, [sample((1, 2, 3)), sample((4, 5, 6))], VersionPolicy.PINNED
            )
            report = self.wait_terminal(group)
            self.assertEqual(report.conclusion, GroupConclusion.COMPLETED)
            self.assertTrue(report.contract_satisfied)
            self.assertEqual(report.versions_observed, ("add-v1",))
            self.assertEqual(report.pinned_version, "add-v1")
            for item in report.items:
                self.assertEqual(item.state, RequestState.FINISHED)
                self.assertIsNotNone(item.attribution)
                self.assertIsNone(item.explanation)
            self.assertEqual(report.items[0].result, (2.0, 3.0, 4.0))
            # Repeated queries are stable.
            self.assertEqual(group.report(), report)
        finally:
            executor.stop()

    def test_pinned_group_detects_version_split(self):
        model_v1 = BlockingAddModel(input_shape=(3,), version="add-v1")
        model_v2 = AddConstantModel(input_shape=(3,), version="add-v2")
        executor = self.make_executor(model_v1, max_batch_size=1, max_wait_time=0.005)
        try:
            group = RequestGroup(
                executor, [sample((1, 2, 3)), sample((4, 5, 6))], VersionPolicy.PINNED
            )
            # First item is inside predict on v1; queue the reload so the
            # second item can only run on v2.
            self.assertTrue(model_v1.entered.wait(2))
            reload_error = []

            def reload():
                try:
                    executor.reload_model(model_v2)
                except Exception as exc:  # pragma: no cover - diagnostic
                    reload_error.append(exc)

            reloader = threading.Thread(target=reload)
            reloader.start()
            time.sleep(0.05)
            model_v1.release(0)
            reloader.join(2)
            self.assertFalse(reload_error)
            report = self.wait_terminal(group)
            self.assertEqual(report.conclusion, GroupConclusion.CONTRACT_VIOLATED)
            self.assertFalse(report.contract_satisfied)
            self.assertEqual(set(report.versions_observed), {"add-v1", "add-v2"})
            self.assertNotEqual(report.conclusion, GroupConclusion.COMPLETED)
        finally:
            executor.stop()

    def test_flexible_group_allows_mixed_versions(self):
        model_v1 = BlockingAddModel(input_shape=(3,), version="add-v1")
        model_v2 = AddConstantModel(input_shape=(3,), version="add-v2")
        executor = self.make_executor(model_v1, max_batch_size=1, max_wait_time=0.005)
        try:
            group = RequestGroup(
                executor,
                [sample((1, 2, 3)), sample((4, 5, 6))],
                VersionPolicy.FLEXIBLE,
            )
            self.assertTrue(model_v1.entered.wait(2))
            reload_error = []

            def reload():
                try:
                    executor.reload_model(model_v2)
                except Exception as exc:  # pragma: no cover - diagnostic
                    reload_error.append(exc)

            reloader = threading.Thread(target=reload)
            reloader.start()
            time.sleep(0.05)
            model_v1.release(0)
            reloader.join(2)
            self.assertFalse(reload_error)
            report = self.wait_terminal(group)
            self.assertEqual(report.conclusion, GroupConclusion.COMPLETED)
            self.assertTrue(report.contract_satisfied)
            self.assertEqual(set(report.versions_observed), {"add-v1", "add-v2"})
        finally:
            executor.stop()

    def test_resubmit_only_unfinished_keeps_finished_identity(self):
        model = FlakyAddModel(input_shape=(3,), version="add-v1")
        executor = self.make_executor(model, max_batch_size=1, max_wait_time=0.005)
        try:
            group = RequestGroup(
                executor, [sample((1, 2, 3)), sample((4, 5, 6))], VersionPolicy.PINNED
            )
            report = self.wait_terminal(group)
            self.assertEqual(report.conclusion, GroupConclusion.COMPLETED_PARTIAL)
            failed = [item for item in report.items if item.state is RequestState.FAILED]
            finished = [
                item for item in report.items if item.state is RequestState.FINISHED
            ]
            self.assertEqual(len(failed), 1)
            self.assertEqual(len(finished), 1)
            self.assertIn("transient backend failure", failed[0].explanation)
            kept_attribution = finished[0].attribution

            resubmitted = group.resubmit_unfinished()
            self.assertEqual(resubmitted, [failed[0].index])
            report = self.wait_terminal(group)
            self.assertEqual(report.conclusion, GroupConclusion.COMPLETED)
            self.assertTrue(report.contract_satisfied)
            again_finished = report.items[finished[0].index]
            self.assertEqual(again_finished.attribution, kept_attribution)
            self.assertEqual(again_finished.attempts, 1)
            self.assertEqual(report.items[failed[0].index].attempts, 2)
        finally:
            executor.stop()

    def test_resubmit_rejected_when_pinned_version_unavailable(self):
        model_v1 = FlakyAddModel(input_shape=(3,), version="add-v1")
        model_v2 = AddConstantModel(input_shape=(3,), version="add-v2")
        executor = self.make_executor(model_v1, max_batch_size=1, max_wait_time=0.005)
        try:
            group = RequestGroup(
                executor, [sample((1, 2, 3)), sample((4, 5, 6))], VersionPolicy.PINNED
            )
            report = self.wait_terminal(group)
            self.assertEqual(report.conclusion, GroupConclusion.COMPLETED_PARTIAL)
            executor.reload_model(model_v2)

            with self.assertRaises(VersionConflictError):
                group.resubmit_unfinished()

            # Explicit reconfirmation amends the contract and proceeds.
            resubmitted = group.resubmit_unfinished(confirm_version_change=True)
            self.assertEqual(len(resubmitted), 1)
            report = self.wait_terminal(group)
            self.assertEqual(report.conclusion, GroupConclusion.COMPLETED)
            self.assertTrue(report.contract_amended)
            self.assertEqual(set(report.versions_observed), {"add-v1", "add-v2"})
        finally:
            executor.stop()

    def test_cancel_remaining_keeps_finished_results(self):
        model = BlockingAddModel(input_shape=(3,), version="add-v1")
        executor = self.make_executor(model, max_batch_size=1, max_wait_time=0.01)
        try:
            group = RequestGroup(
                executor,
                [sample((1, 2, 3)), sample((4, 5, 6)), sample((7, 8, 9))],
                VersionPolicy.PINNED,
            )
            self.assertTrue(model.entered.wait(2))
            model.release(0)
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                if any(
                    item.state is RequestState.FINISHED for item in group.report().items
                ):
                    break
                time.sleep(0.005)
            group.cancel_remaining()
            report = self.wait_terminal(group)
            self.assertEqual(report.conclusion, GroupConclusion.CANCELLED)
            finished = [
                item for item in report.items if item.state is RequestState.FINISHED
            ]
            cancelled = [
                item for item in report.items if item.state is RequestState.CANCELLED
            ]
            self.assertEqual(len(finished), 1)
            self.assertEqual(len(cancelled), 2)
            self.assertIsNotNone(finished[0].attribution)
            self.assertEqual(finished[0].result, (2.0, 3.0, 4.0))
            for item in cancelled:
                self.assertIsNotNone(item.explanation)
        finally:
            executor.stop()

    def test_stop_gives_every_item_an_explained_terminal_state(self):
        entered = threading.Event()
        release = threading.Event()

        class SlowModel:
            name = "slow"
            version = "slow-v1"

            def check(self, shape, dtype):
                return None

            def predict(self, samples):
                entered.set()
                release.wait(2)
                return [tuple(s) for s in samples]

        executor = self.make_executor(
            SlowModel(), max_batch_size=1, max_wait_time=0.005, default_timeout=None
        )
        group = RequestGroup(
            executor, [sample((1, 2, 3)), sample((4, 5, 6))], VersionPolicy.PINNED
        )
        self.assertTrue(entered.wait(2))
        executor.stop(timeout=0.2)
        release.set()
        report = group.report()
        self.assertIn(
            report.conclusion,
            (GroupConclusion.COMPLETED_PARTIAL, GroupConclusion.COMPLETED),
        )
        self.assertNotEqual(report.conclusion, GroupConclusion.IN_PROGRESS)
        for item in report.items:
            self.assertIn(
                item.state,
                (RequestState.FINISHED, RequestState.FAILED, RequestState.CANCELLED),
            )
            if item.state is not RequestState.FINISHED:
                self.assertIsNotNone(item.explanation)
        # Stable under repeated reads after shutdown.
        self.assertEqual(group.report(), report)

    def wait_terminal(self, group, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            report = group.report()
            if report.conclusion is not GroupConclusion.IN_PROGRESS:
                return report
            time.sleep(0.01)
        self.fail("group did not reach a terminal conclusion")


class GroupStoreTests(unittest.TestCase):
    def test_saved_report_reopens_with_same_conclusion(self):
        model = AddConstantModel(input_shape=(3,), version="add-v1")
        executor = DynamicBatchingExecutor(
            model, ExecutorConfig(max_batch_size=4, max_wait_time=0.01)
        ).start()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "groups.json")
            store = JsonGroupStore(path)
            try:
                group = RequestGroup(
                    executor, [sample((1, 2, 3)), sample((4, 5, 6))], VersionPolicy.PINNED
                )
                deadline = time.monotonic() + 5
                while True:
                    report = group.report()
                    if report.conclusion is not GroupConclusion.IN_PROGRESS:
                        break
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.01)
                store.save(report)
            finally:
                executor.stop()

            reopened = store.load(group.group_id)
            self.assertEqual(reopened.conclusion, GroupConclusion.COMPLETED)
            self.assertEqual(reopened, report)
            self.assertEqual(store.load(group.group_id), reopened)

    def test_reopened_record_never_shows_complete_with_unexplained_failures(self):
        model = FlakyAddModel(input_shape=(3,), version="add-v1")
        executor = DynamicBatchingExecutor(
            model, ExecutorConfig(max_batch_size=1, max_wait_time=0.005)
        ).start()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "groups.json")
            store = JsonGroupStore(path)
            try:
                group = RequestGroup(
                    executor, [sample((1, 2, 3)), sample((4, 5, 6))], VersionPolicy.PINNED
                )
                deadline = time.monotonic() + 5
                while True:
                    report = group.report()
                    if report.conclusion is not GroupConclusion.IN_PROGRESS:
                        break
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.01)
                self.assertEqual(report.conclusion, GroupConclusion.COMPLETED_PARTIAL)
                record = report.to_dict()
                # Simulate a corrupted/legacy record that claims completion.
                record["conclusion"] = GroupConclusion.COMPLETED.value
                import json

                with open(path, "w", encoding="utf-8") as handle:
                    json.dump({report.group_id: record}, handle)
            finally:
                executor.stop()

            reopened = store.load(report.group_id)
            self.assertEqual(reopened.conclusion, GroupConclusion.COMPLETED_PARTIAL)
            failed = [
                item for item in reopened.items if item.state is RequestState.FAILED
            ]
            self.assertEqual(len(failed), 1)
            self.assertIsNotNone(failed[0].explanation)


if __name__ == "__main__":
    unittest.main()
