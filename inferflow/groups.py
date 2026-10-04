from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from .errors import EngineStoppedError, RequestCancelledError, VersionContractError
from .types import (
    BatchAttribution,
    CandidateState,
    DeliveryMode,
    GroupState,
    HedgePolicy,
    ItemState,
    PredictionFuture,
    Request,
    RequestState,
    VersionContract,
)

if TYPE_CHECKING:
    from .executor import DynamicBatchingExecutor

# Unmet delivery-condition tags.
UNMET_WAITING = "waiting_for_candidate"
UNMET_RESULTS_DISAGREE = "results_disagree"
UNMET_FAILURE = "candidate_failed"
UNMET_INSUFFICIENT = "insufficient_conformant_results"
UNMET_VERSION = "candidate_version_not_conformant"
UNMET_BUDGET_EXHAUSTED = "redundancy_budget_exhausted"

_TERMINAL_CANDIDATE = {
    CandidateState.FINISHED,
    CandidateState.FAILED,
    CandidateState.CANCELLED,
}


@dataclass(frozen=True)
class CandidateReport:
    """One redundant execution of a request item, with its own identity."""

    candidate_id: str
    label: str
    state: CandidateState
    model_version: Optional[str] = None
    start_condition: Optional[str] = None
    result: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    attribution: Optional[BatchAttribution] = None
    conformant: bool = True
    adopted: bool = False
    verified: bool = False
    late: bool = False


@dataclass(frozen=True)
class ItemReport:
    item_id: str
    request_id: str
    state: ItemState
    result: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    attribution: Optional[BatchAttribution] = None
    attempts: tuple[BatchAttribution, ...] = ()
    delivered: bool = False
    adopted_candidate_id: Optional[str] = None
    unmet_conditions: tuple[str, ...] = ()
    candidates: tuple[CandidateReport, ...] = ()


@dataclass(frozen=True)
class RedundancyReport:
    """Group-wide accounting for the shared redundancy budget."""

    delivery: DeliveryMode
    hedge_after: float
    extra_budget: int
    committed_extra: int
    pending_extra: int
    completed_extra: int
    late_executions: int


@dataclass(frozen=True)
class GroupReport:
    group_id: str
    contract: VersionContract
    pinned_version: Optional[str]
    state: GroupState
    version_conformant: bool
    executed_versions: tuple[str, ...]
    items: tuple[ItemReport, ...]
    unmet_conditions: tuple[str, ...] = ()
    redundancy: Optional[RedundancyReport] = None
    fixed_deliveries: tuple[str, ...] = ()
    irrevocable_executions: int = 0

    @property
    def complete(self) -> bool:
        return self.state is GroupState.EXECUTED


class _Candidate:
    def __init__(
        self,
        candidate_id: str,
        label: str,
        request: Request,
        future: Optional[PredictionFuture],
        state: CandidateState,
        start_condition: str,
    ):
        self.candidate_id = candidate_id
        self.label = label
        self.request = request
        self.future = future
        self.state = state
        self.start_condition = start_condition
        self.conformant = True
        self.adopted = False
        self.verified = False
        self.late = False
        self.finish_order: Optional[int] = None

    @property
    def committed(self) -> bool:
        return self.future is not None


class _GroupItem:
    def __init__(self, item_id: str):
        self.item_id = item_id
        self.candidates: list[_Candidate] = []
        self.primary_request: Optional[Request] = None
        self.delivered = False
        self.adopted_candidate_id: Optional[str] = None

    def primary(self) -> _Candidate:
        return self.candidates[0]


class RequestGroup:
    """A coordinated set of requests governed by one delivery agreement.

    When a :class:`HedgePolicy` is supplied, each item owns one primary
    execution plus a pending redundant candidate. Candidates are separate
    executions (each with its own model, start condition, and recorded
    result) but belong to the same request item; the item adopts exactly one
    result at most.
    """

    def __init__(
        self,
        executor: "DynamicBatchingExecutor",
        requests: list[Request],
        contract: VersionContract = VersionContract.SAME_VERSION,
        group_id: Optional[str] = None,
        pinned_version: Optional[str] = None,
        submit_timeout: Optional[float] = -1,
        hedge: Optional[HedgePolicy] = None,
    ):
        if not requests:
            raise ValueError("request group must contain at least one request")
        self._executor: Optional["DynamicBatchingExecutor"] = executor
        self._lock = threading.RLock()
        self.group_id = group_id or uuid.uuid4().hex
        self.contract = contract
        self.pinned_version = pinned_version
        if self.pinned_version is None and contract is VersionContract.SAME_VERSION:
            self.pinned_version = executor.model.version
        self._items: dict[str, _GroupItem] = {}
        self._saved_reports: dict[str, GroupReport] = {}
        self._submit_timeout = submit_timeout
        self.hedge_policy = hedge
        self._clock = getattr(executor, "_clock", time.monotonic)
        self._group_started_at = self._clock()
        self._committed_extra = 0
        self._finish_sequence = 0
        self._monitor_stop = threading.Event()
        self._monitor: Optional[threading.Thread] = None

        required_version = (
            self.pinned_version if contract is VersionContract.SAME_VERSION else None
        )
        for index, request in enumerate(requests):
            item_id = request.request_id
            item = _GroupItem(item_id)
            primary_request = self._prepare_request(request, required_version)
            item.primary_request = primary_request
            future = executor.submit(primary_request, submit_timeout)
            primary = _Candidate(
                candidate_id=f"{item_id}:primary",
                label="primary",
                request=primary_request,
                future=future,
                state=CandidateState.COMMITTED,
                start_condition="immediate",
            )
            item.candidates.append(primary)
            if hedge is not None:
                hedge_required = (
                    hedge.hedge_required_version
                    if hedge.hedge_required_version is not None
                    else None
                )
                hedge_request = self._prepare_request(request, hedge_required, suffix="hedge")
                hedge_candidate = _Candidate(
                    candidate_id=f"{item_id}:hedge",
                    label="hedge",
                    request=hedge_request,
                    future=None,
                    state=CandidateState.PENDING,
                    start_condition=f"after {hedge.hedge_after:g}s undelivered",
                )
                item.candidates.append(hedge_candidate)
            self._items[item_id] = item

        if hedge is not None:
            self._monitor = threading.Thread(
                target=self._run_monitor, name=f"hedge-monitor-{self.group_id[:8]}", daemon=True
            )
            self._monitor.start()

    @classmethod
    def from_report(cls, report: GroupReport) -> "RequestGroup":
        group = object.__new__(cls)
        group._executor = None
        group._lock = threading.RLock()
        group.group_id = report.group_id
        group.contract = report.contract
        group.pinned_version = report.pinned_version
        group._items = {}
        group._saved_reports = {"reopened": report}
        group._submit_timeout = -1
        group.hedge_policy = None
        group._clock = time.monotonic
        group._group_started_at = 0.0
        group._committed_extra = 0
        group._finish_sequence = 0
        group._monitor_stop = threading.Event()
        group._monitor = None
        return group

    def _prepare_request(
        self,
        request: Request,
        required_version: Optional[str],
        suffix: Optional[str] = None,
    ) -> Request:
        required = required_version
        if required is None and request.required_model_version is not None:
            required = request.required_model_version
        if (
            suffix is None
            and required is not None
            and request.required_model_version not in (None, required)
        ):
            raise VersionContractError(
                f"request requires model version {request.required_model_version!r}; "
                f"group pins {required!r}"
            )
        return Request(
            data=request.data,
            shape=request.shape,
            dtype=request.dtype,
            priority=request.priority,
            required_model_version=required,
            request_id=request.request_id
            if suffix is None
            else f"{request.request_id}:{suffix}",
        )
        if suffix is None:
            return request
        return Request(
            data=request.data,
            shape=request.shape,
            dtype=request.dtype,
            priority=request.priority,
            required_model_version=request.required_model_version,
            request_id=f"{request.request_id}:{suffix}",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def refresh(self) -> GroupReport:
        with self._lock:
            if not self._items:
                return self._saved_reports["reopened"]
            self._sync_locked()
            for item in self._items.values():
                self._evaluate_item_locked(item)
            return self._build_report_locked()

    def report(self) -> GroupReport:
        return self.refresh()

    def wait(self, timeout: Optional[float] = None) -> GroupReport:
        report = self.refresh()
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self._terminal(report):
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return report
                threading.Event().wait(min(0.01, remaining))
            else:
                threading.Event().wait(0.01)
            report = self.refresh()
        return report

    def cancel_remaining(self) -> GroupReport:
        with self._lock:
            if self._executor is None:
                raise EngineStoppedError("reopened group records cannot be modified")
            self._sync_locked()
            executor = self._executor
            for item in self._items.values():
                if item.delivered:
                    continue
                for candidate in item.candidates:
                    if candidate.state is CandidateState.PENDING:
                        candidate.state = CandidateState.CANCELLED
                        candidate.start_condition = candidate.start_condition + "; cancelled before start"
                    elif candidate.state is CandidateState.COMMITTED and not candidate.future.done():
                        candidate.future.cancel(
                            RequestCancelledError("remaining group request cancelled")
                        )
                        executor.cancel(candidate.request.request_id)
            return self.report()

    def retry_unfinished(self, allow_version_switch: bool = False) -> GroupReport:
        """Submit replacement executions for items with no adopted result."""
        with self._lock:
            if self._executor is None:
                raise EngineStoppedError("reopened group records cannot be modified")
            current_version = self._executor.model.version
            if self.contract is VersionContract.SAME_VERSION and current_version != self.pinned_version:
                if not allow_version_switch:
                    raise VersionContractError(
                        f"group requires model version {self.pinned_version!r}; "
                        f"current model is {current_version!r}"
                    )
                self.contract = VersionContract.CURRENT_VERSION
                self.pinned_version = None
            self._sync_locked()
            for item in self._items.values():
                if item.delivered:
                    continue
                if not any(
                    candidate.state in (CandidateState.FAILED, CandidateState.CANCELLED)
                    for candidate in item.candidates
                ):
                    continue
                self._resubmit_item_locked(item)
            return self.report()

    def save(self, path: str | Path) -> GroupReport:
        report = self.report()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = _report_to_jsonable(report)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with self._lock:
            self._saved_reports[str(path)] = report
        return report

    # ------------------------------------------------------------------
    # Redundancy scheduling
    # ------------------------------------------------------------------
    def _run_monitor(self) -> None:
        policy = self.hedge_policy
        while not self._monitor_stop.wait(0.005):
            with self._lock:
                if self._executor is None:
                    return
                now = self._clock()
                self._sync_locked()
                for item in self._items.values():
                    self._maybe_commit_hedge_locked(item, now)
                self._evaluate_all_locked()
                if all(item.delivered or self._item_is_terminal_locked(item) for item in self._items.values()):
                    self._monitor_stop.set()
                    return

    def _maybe_commit_hedge_locked(self, item: _GroupItem, now: float) -> None:
        policy = self.hedge_policy
        if policy is None or item.delivered:
            return
        hedge = next(
            (candidate for candidate in item.candidates if candidate.label == "hedge"),
            None,
        )
        if hedge is None or hedge.state is not CandidateState.PENDING:
            return
        primary = item.primary()
        if not primary.committed:
            return
        # A primary that already failed is rescued immediately regardless of
        # the timeout; a queued primary that has not even started is not the
        # tail latency the redundancy budget is meant to cover.
        primary_dead = primary.state in (CandidateState.FAILED,)
        primary_running = (
            primary.state is CandidateState.COMMITTED
            and primary.future is not None
            and primary.future.state in (RequestState.RUNNING, RequestState.FINISHED)
        )
        if primary_dead:
            condition = "hedge_after_primary_failure"
        elif primary_running and now - self._group_started_at >= policy.hedge_after:
            condition = "hedge_after_delay"
        else:
            return
        if self._committed_extra >= policy.max_extra_executions:
            return
        self._commit_candidate_locked(hedge, condition)

    def _commit_candidate_locked(self, candidate: _Candidate, condition: str) -> bool:
        executor = self._executor
        if executor is None:
            return False
        required_version = (
            self.pinned_version if self.contract is VersionContract.SAME_VERSION else None
        )
        if candidate.label == "hedge" and self.hedge_policy is not None:
            required_version = self.hedge_policy.hedge_required_version
        request = self._prepare_request(
            Request(
                data=candidate.request.data,
                shape=candidate.request.shape,
                dtype=candidate.request.dtype,
                priority=candidate.request.priority,
            ),
            required_version,
        )
        try:
            future = executor.submit(request, self._submit_timeout)
        except EngineStoppedError:
            candidate.state = CandidateState.FAILED
            candidate.start_condition = condition + "; executor closed"
            return False
        except Exception:
            candidate.state = CandidateState.FAILED
            candidate.start_condition = condition + "; rejected"
            return False
        candidate.request = request
        candidate.future = future
        candidate.state = CandidateState.COMMITTED
        candidate.start_condition = condition
        self._committed_extra += 1
        return True

    def _resubmit_item_locked(self, item: _GroupItem) -> None:
        executor = self._executor
        required_version = (
            self.pinned_version if self.contract is VersionContract.SAME_VERSION else None
        )
        base_request = item.primary_request
        request = self._prepare_request(
            Request(
                data=base_request.data,
                shape=base_request.shape,
                dtype=base_request.dtype,
                priority=base_request.priority,
            ),
            required_version,
        )
        future = executor.submit(request, self._submit_timeout)
        attempt_number = sum(1 for candidate in item.candidates if candidate.label == "primary-retry") + 1
        candidate = _Candidate(
            candidate_id=f"{item.item_id}:retry-{attempt_number}",
            label="primary-retry",
            request=request,
            future=future,
            state=CandidateState.COMMITTED,
            start_condition="retry_unfinished",
        )
        item.candidates.append(candidate)
        if self.hedge_policy is not None and self._committed_extra < self.hedge_policy.max_extra_executions:
            hedge_request = self._prepare_request(
                Request(
                    data=base_request.data,
                    shape=base_request.shape,
                    dtype=base_request.dtype,
                    priority=base_request.priority,
                ),
                required_version,
                suffix="hedge",
            )
            pending = _Candidate(
                candidate_id=f"{item.item_id}:hedge-{attempt_number}",
                label="hedge",
                request=hedge_request,
                future=None,
                state=CandidateState.PENDING,
                start_condition=f"after {self.hedge_policy.hedge_after:g}s undelivered",
            )
            item.candidates.append(pending)

    # ------------------------------------------------------------------
    # Candidate state synchronisation and the single adoption decision
    # ------------------------------------------------------------------
    def _sync_locked(self) -> None:
        for item in self._items.values():
            for candidate in item.candidates:
                future = candidate.future
                if future is None or candidate.state in _TERMINAL_CANDIDATE:
                    continue
                if not future.done():
                    continue
                self._finish_sequence += 1
                candidate.finish_order = self._finish_sequence
                candidate.conformant = self._candidate_conformant_locked(candidate)
                if future.state is RequestState.FINISHED:
                    candidate.state = CandidateState.FINISHED
                elif future.state is RequestState.CANCELLED:
                    candidate.state = CandidateState.CANCELLED
                else:
                    candidate.state = CandidateState.FAILED
                if item.delivered and candidate.candidate_id != item.adopted_candidate_id:
                    candidate.late = True

    def _mark_late_remaining_locked(self, item: _GroupItem) -> None:
        for candidate in item.candidates:
            if candidate.candidate_id != item.adopted_candidate_id and candidate.future is not None:
                candidate.late = True

    def _candidate_conformant_locked(self, candidate: _Candidate) -> bool:
        attribution = candidate.future.attribution
        if attribution is None:
            return False
        if self.contract is not VersionContract.SAME_VERSION:
            return True
        return attribution.model_version == self.pinned_version

    def _evaluate_all_locked(self) -> None:
        for item in self._items.values():
            self._evaluate_item_locked(item)

    def _evaluate_item_locked(self, item: _GroupItem) -> None:
        # A prior adoption is frozen: later candidate records never rewrite it.
        if item.delivered:
            adopted = next(
                candidate for candidate in item.candidates if candidate.candidate_id == item.adopted_candidate_id
            )
            for candidate in item.candidates:
                if candidate is adopted or candidate.adopted or candidate.verified:
                    continue
                if candidate.state in (CandidateState.PENDING, CandidateState.COMMITTED):
                    self._discard_candidate_locked(item, candidate)
            return

        policy = self.hedge_policy
        delivery = policy.delivery if policy is not None else DeliveryMode.FIRST_CONFORMANT
        conformant_finished = [
            candidate
            for candidate in item.candidates
            if candidate.state is CandidateState.FINISHED and candidate.conformant
        ]
        rank = {"primary": 0, "hedge": 1, "primary-retry": 2}
        conformant_finished.sort(
            key=lambda candidate: (
                candidate.finish_order if self.hedge_policy is None or self.hedge_policy.delivery is DeliveryMode.FIRST_CONFORMANT else rank.get(candidate.label, 3),
                candidate.finish_order,
            )
        )

        if delivery is DeliveryMode.FIRST_CONFORMANT:
            self._evaluate_first_locked(item, conformant_finished)
        else:
            self._evaluate_compare_locked(item, conformant_finished)

    def _evaluate_first_locked(
        self, item: _GroupItem, conformant_finished: list[_Candidate]
    ) -> None:
        if conformant_finished:
            winner = conformant_finished[0]
            self._adopt_locked(item, winner)
            return
        self._discard_redundant_when_resolved_locked(item)

    def _evaluate_compare_locked(
        self, item: _GroupItem, conformant_finished: list[_Candidate]
    ) -> None:
        if len(conformant_finished) >= 2:
            first, second = conformant_finished[0], conformant_finished[1]
            if first.future.result(0) == second.future.result(0):
                for candidate in conformant_finished[1:]:
                    candidate.verified = True
                self._adopt_locked(item, first)
                return
            # Divergent conformant results: the delivery condition can never
            # be satisfied. Any still-live execution is already committed work
            # that cannot be revoked; retire only the not-yet-started one.
            self._discard_redundant_when_resolved_locked(item, disagreement=True)
            return
        self._discard_redundant_when_resolved_locked(item)

    def _adopt_locked(self, item: _GroupItem, winner: _Candidate) -> None:
        winner.adopted = True
        item.delivered = True
        item.adopted_candidate_id = winner.candidate_id
        for candidate in item.candidates:
            if candidate is winner:
                continue
            if candidate.state in _TERMINAL_CANDIDATE:
                candidate.late = True
            self._discard_candidate_locked(item, candidate)

    def _discard_redundant_when_resolved_locked(
        self, item: _GroupItem, disagreement: bool = False
    ) -> None:
        # With no possible winner, retire pending / still-running candidates.
        if not self._item_failure_terminal_locked(item, disagreement):
            return
        for candidate in item.candidates:
            if candidate.state in (CandidateState.PENDING, CandidateState.COMMITTED):
                self._discard_candidate_locked(item, candidate)

    def _item_failure_terminal_locked(
        self, item: _GroupItem, disagreement: bool = False
    ) -> bool:
        policy = self.hedge_policy
        has_live = any(
            candidate.state in (CandidateState.PENDING, CandidateState.COMMITTED)
            for candidate in item.candidates
        )
        if disagreement and not has_live:
            return True
        if policy is None:
            latest = item.candidates[-1]
            return latest.state in (CandidateState.FAILED, CandidateState.CANCELLED)
        # A hedge can still rescue the item while any candidate is pending or
        # committed and the group still owns redundancy budget.
        if has_live:
            return False
        budget_exhausted = self._committed_extra >= policy.max_extra_executions
        hedge_pending = any(
            candidate.label == "hedge" and candidate.state is CandidateState.PENDING
            for candidate in item.candidates
        )
        return not hedge_pending or budget_exhausted

    def _discard_candidate_locked(self, item: _GroupItem, candidate: _Candidate) -> None:
        if candidate.state is CandidateState.PENDING:
            candidate.state = CandidateState.CANCELLED
            candidate.start_condition = candidate.start_condition + "; not started"
            return
        if candidate.state is CandidateState.COMMITTED:
            future = candidate.future
            executor = self._executor
            if future is not None and executor is not None and not future.done():
                future.cancel(
                    RequestCancelledError("redundant execution superseded before delivery")
                )
                executor.cancel(candidate.request.request_id)

    def _item_is_terminal_locked(self, item: _GroupItem) -> bool:
        if item.delivered:
            return True
        return self._item_failure_terminal_locked(item)

    @staticmethod
    def _terminal(report: GroupReport) -> bool:
        return report.state in (GroupState.EXECUTED, GroupState.FAILED, GroupState.CANCELLED)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def _build_report_locked(self) -> GroupReport:
        item_reports = tuple(self._item_report_locked(item) for item in self._items.values())

        delivered = [item for item in item_reports if item.delivered]
        versions = tuple(
            dict.fromkeys(
                item.attribution.model_version
                for item in delivered
                if item.attribution is not None
            )
        )
        version_conformant = True
        if self.contract is VersionContract.SAME_VERSION:
            expected = self.pinned_version
            version_conformant = all(
                item.attribution is not None and item.attribution.model_version == expected
                for item in delivered
            )

        has_failed = any(item.state is ItemState.FAILED for item in item_reports)
        has_cancelled = any(item.state is ItemState.CANCELLED for item in item_reports)
        has_waiting = any(item.state is ItemState.WAITING for item in item_reports)
        all_delivered = all(item.delivered for item in item_reports)

        if all_delivered and version_conformant:
            state = GroupState.EXECUTED
        elif has_failed or not version_conformant:
            state = GroupState.FAILED
        elif has_waiting:
            state = GroupState.WAITING
        elif has_cancelled:
            state = GroupState.CANCELLED
        else:
            state = GroupState.FAILED

        group_unmet = tuple(
            dict.fromkeys(
                condition
                for item in item_reports
                for condition in item.unmet_conditions
            )
        )

        fixed_deliveries = tuple(item.item_id for item in item_reports if item.delivered)
        irrevocable = sum(
            1
            for item in self._items.values()
            for candidate in item.candidates
            if candidate.state is CandidateState.CANCELLED
            and candidate.future is not None
            and candidate.future.attribution is not None
        )

        redundancy = None
        policy = self.hedge_policy
        if policy is not None:
            pending_extra = sum(
                1
                for item in self._items.values()
                for candidate in item.candidates
                if candidate.label == "hedge" and candidate.state is CandidateState.PENDING
            )
            completed_extra = sum(
                1
                for item in self._items.values()
                for candidate in item.candidates
                if candidate.label == "hedge"
                and candidate.future is not None
                and candidate.state in (CandidateState.FINISHED, CandidateState.FAILED, CandidateState.CANCELLED)
            )
            late_executions = sum(
                1
                for item in self._items.values()
                for candidate in item.candidates
                if candidate.late
            )
            redundancy = RedundancyReport(
                delivery=policy.delivery,
                hedge_after=policy.hedge_after,
                extra_budget=policy.max_extra_executions,
                committed_extra=self._committed_extra,
                pending_extra=pending_extra,
                completed_extra=completed_extra,
                late_executions=late_executions,
            )

        return GroupReport(
            group_id=self.group_id,
            contract=self.contract,
            pinned_version=self.pinned_version,
            state=state,
            version_conformant=version_conformant,
            executed_versions=versions,
            items=item_reports,
            unmet_conditions=group_unmet,
            redundancy=redundancy,
            fixed_deliveries=fixed_deliveries,
            irrevocable_executions=irrevocable,
        )

    def _item_report_locked(self, item: _GroupItem) -> ItemReport:
        candidate_reports = tuple(
            self._candidate_report_locked(candidate) for candidate in item.candidates
        )
        attempts = tuple(
            candidate.attribution
            for candidate in candidate_reports
            if candidate.attribution is not None
        )
        adopted = next(
            (candidate for candidate in candidate_reports if candidate.adopted),
            None,
        )

        if item.delivered and adopted is not None:
            return ItemReport(
                item_id=item.item_id,
                request_id=item.item_id,
                state=ItemState.EXECUTED,
                result=adopted.result,
                attribution=adopted.attribution,
                attempts=attempts,
                delivered=True,
                adopted_candidate_id=adopted.candidate_id,
                candidates=candidate_reports,
            )

        unmet = self._item_unmet_locked(item)
        latest = item.candidates[-1]
        primary_report = self._candidate_report_locked(latest)
        if self._item_failure_terminal_locked(item):
            any_failed = any(
                candidate.state is CandidateState.FAILED
                or (candidate.future is not None and not candidate.conformant and candidate.state is CandidateState.FINISHED)
                or UNMET_RESULTS_DISAGREE in unmet
                for candidate in item.candidates
            )
            state = ItemState.FAILED if any_failed else ItemState.CANCELLED
        else:
            state = ItemState.WAITING
        return ItemReport(
            item_id=item.item_id,
            request_id=item.item_id,
            state=state,
            result=None,
            error=primary_report.error,
            error_type=primary_report.error_type,
            attribution=primary_report.attribution,
            attempts=attempts,
            delivered=False,
            unmet_conditions=tuple(dict.fromkeys(unmet)),
            candidates=candidate_reports,
        )

    def _item_unmet_locked(self, item: _GroupItem) -> list[str]:
        unmet: list[str] = []
        policy = self.hedge_policy
        compare = policy is not None and policy.delivery is DeliveryMode.COMPARE
        conformant_finished = [
            candidate
            for candidate in item.candidates
            if candidate.state is CandidateState.FINISHED and candidate.conformant
        ]
        nonconformant_finished = [
            candidate
            for candidate in item.candidates
            if candidate.state is CandidateState.FINISHED and not candidate.conformant
        ]
        if nonconformant_finished:
            unmet.append(UNMET_VERSION)
        if compare and len(conformant_finished) >= 2:
            first, second = conformant_finished[0], conformant_finished[1]
            if first.future.result(0) != second.future.result(0):
                unmet.append(UNMET_RESULTS_DISAGREE)
        if any(candidate.state is CandidateState.FAILED for candidate in item.candidates):
            unmet.append(UNMET_FAILURE)

        terminal = self._item_failure_terminal_locked(item, disagreement=UNMET_RESULTS_DISAGREE in unmet)
        if not terminal:
            unmet.append(UNMET_WAITING)
            if compare and len(conformant_finished) < 2:
                unmet.append(UNMET_INSUFFICIENT)
            return unmet

        if compare and UNMET_RESULTS_DISAGREE not in unmet and len(conformant_finished) < 2:
            unmet.append(UNMET_INSUFFICIENT)
        if policy is not None and self._committed_extra >= policy.max_extra_executions:
            pending_hedge = any(
                candidate.label == "hedge" and candidate.state is CandidateState.PENDING
                for candidate in item.candidates
            )
            if pending_hedge:
                unmet.append(UNMET_BUDGET_EXHAUSTED)
        if not any(
            candidate.state is CandidateState.FINISHED
            for candidate in item.candidates
        ) and all(
            candidate.state is CandidateState.CANCELLED for candidate in item.candidates
        ) and UNMET_FAILURE not in unmet:
            pass
        return unmet

    def _candidate_report_locked(self, candidate: _Candidate) -> CandidateReport:
        future = candidate.future
        result = None
        error = None
        error_type = None
        attribution = None
        model_version = None
        if future is not None:
            attribution = future.attribution
            if attribution is not None:
                model_version = attribution.model_version
            if future.done():
                try:
                    result = future.result(0)
                except BaseException as exc:
                    error = str(exc)
                    error_type = type(exc).__name__
        return CandidateReport(
            candidate_id=candidate.candidate_id,
            label=candidate.label,
            state=candidate.state,
            model_version=model_version,
            start_condition=candidate.start_condition,
            result=result,
            error=error,
            error_type=error_type,
            attribution=attribution,
            conformant=candidate.conformant,
            adopted=candidate.adopted,
            verified=candidate.verified,
            late=candidate.late,
        )


class GroupStore:
    """JSON-backed group snapshot storage."""

    def save(self, path: str | Path, group: RequestGroup) -> GroupReport:
        return group.save(path)

    def load(self, path: str | Path) -> RequestGroup:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return RequestGroup.from_report(_report_from_jsonable(payload))


def _report_to_jsonable(report: GroupReport) -> dict[str, Any]:
    redundancy = None
    if report.redundancy is not None:
        redundancy = {
            "delivery": report.redundancy.delivery.value,
            "hedge_after": report.redundancy.hedge_after,
            "extra_budget": report.redundancy.extra_budget,
            "committed_extra": report.redundancy.committed_extra,
            "pending_extra": report.redundancy.pending_extra,
            "completed_extra": report.redundancy.completed_extra,
            "late_executions": report.redundancy.late_executions,
        }
    return {
        "group_id": report.group_id,
        "contract": report.contract.value,
        "pinned_version": report.pinned_version,
        "state": report.state.value,
        "version_conformant": report.version_conformant,
        "executed_versions": list(report.executed_versions),
        "unmet_conditions": list(report.unmet_conditions),
        "fixed_deliveries": list(report.fixed_deliveries),
        "irrevocable_executions": report.irrevocable_executions,
        "redundancy": redundancy,
        "items": [
            {
                "item_id": item.item_id,
                "request_id": item.request_id,
                "state": item.state.value,
                "result": item.result,
                "error": item.error,
                "error_type": item.error_type,
                "attribution": _attribution_to_jsonable(item.attribution),
                "attempts": [_attribution_to_jsonable(value) for value in item.attempts],
                "delivered": item.delivered,
                "adopted_candidate_id": item.adopted_candidate_id,
                "unmet_conditions": list(item.unmet_conditions),
                "candidates": [
                    {
                        "candidate_id": candidate.candidate_id,
                        "label": candidate.label,
                        "state": candidate.state.value,
                        "model_version": candidate.model_version,
                        "start_condition": candidate.start_condition,
                        "result": candidate.result,
                        "error": candidate.error,
                        "error_type": candidate.error_type,
                        "attribution": _attribution_to_jsonable(candidate.attribution),
                        "conformant": candidate.conformant,
                        "adopted": candidate.adopted,
                        "verified": candidate.verified,
                        "late": candidate.late,
                    }
                    for candidate in item.candidates
                ],
            }
            for item in report.items
        ],
    }


def _report_from_jsonable(payload: dict[str, Any]) -> GroupReport:
    items = tuple(_item_from_jsonable(item) for item in payload["items"])
    redundancy_payload = payload.get("redundancy")
    redundancy = None
    if redundancy_payload is not None:
        redundancy = RedundancyReport(
            delivery=DeliveryMode(redundancy_payload["delivery"]),
            hedge_after=redundancy_payload["hedge_after"],
            extra_budget=redundancy_payload["extra_budget"],
            committed_extra=redundancy_payload["committed_extra"],
            pending_extra=redundancy_payload["pending_extra"],
            completed_extra=redundancy_payload["completed_extra"],
            late_executions=redundancy_payload.get("late_executions", 0),
        )
    return GroupReport(
        group_id=payload["group_id"],
        contract=VersionContract(payload["contract"]),
        pinned_version=payload.get("pinned_version"),
        state=GroupState(payload["state"]),
        version_conformant=payload["version_conformant"],
        executed_versions=tuple(payload.get("executed_versions", [])),
        items=items,
        unmet_conditions=tuple(payload.get("unmet_conditions", [])),
        redundancy=redundancy,
        fixed_deliveries=tuple(payload.get("fixed_deliveries", [])),
        irrevocable_executions=payload.get("irrevocable_executions", 0),
    )


def _item_from_jsonable(payload: dict[str, Any]) -> ItemReport:
    candidates = tuple(
        CandidateReport(
            candidate_id=candidate["candidate_id"],
            label=candidate["label"],
            state=CandidateState(candidate["state"]),
            model_version=candidate.get("model_version"),
            start_condition=candidate.get("start_condition"),
            result=candidate.get("result"),
            error=candidate.get("error"),
            error_type=candidate.get("error_type"),
            attribution=_attribution_from_jsonable(candidate.get("attribution")),
            conformant=candidate.get("conformant", True),
            adopted=candidate.get("adopted", False),
            verified=candidate.get("verified", False),
            late=candidate.get("late", False),
        )
        for candidate in payload.get("candidates", [])
    )
    attempts = tuple(
        value
        for value in (
            _attribution_from_jsonable(attempt)
            for attempt in payload.get("attempts", [])
        )
        if value is not None
    )
    return ItemReport(
        item_id=payload["item_id"],
        request_id=payload["request_id"],
        state=ItemState(payload["state"]),
        result=payload.get("result"),
        error=payload.get("error"),
        error_type=payload.get("error_type"),
        attribution=_attribution_from_jsonable(payload.get("attribution")),
        attempts=attempts,
        delivered=payload.get("delivered", False),
        adopted_candidate_id=payload.get("adopted_candidate_id"),
        unmet_conditions=tuple(payload.get("unmet_conditions", [])),
        candidates=candidates,
    )


def _attribution_to_jsonable(value: Optional[BatchAttribution]) -> Optional[dict[str, Any]]:
    return None if value is None else asdict(value)


def _attribution_from_jsonable(value: Optional[dict[str, Any]]) -> Optional[BatchAttribution]:
    if value is None:
        return None
    return BatchAttribution(**value)
