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
    AllocationStrategy,
    BatchAttribution,
    CandidateState,
    GroupState,
    HedgeDelivery,
    ItemState,
    Request,
    RequestState,
    VersionContract,
)

if TYPE_CHECKING:
    from .executor import DynamicBatchingExecutor


@dataclass(frozen=True)
class HedgeConfig:
    """Redundant-execution policy shared by every item of a group.

    ``budget`` bounds the *extra* executions beyond each item's primary
    attempt for the whole group. ``delivery`` selects the delivery contract:
    the first conforming candidate, or two conforming candidates checked for
    equal content.

    ``strategy`` decides which undelivered item receives the next budget
    slot: ``FAIR`` spreads slots round-robin, ``DEADLINE_FIRST`` rescues the
    item whose caller-supplied deadline is closest. Slots are allocated
    dynamically: a slot held by a candidate that never started returns to
    the pool and can be promised to another item, while a committed or
    completed execution keeps its slot forever.
    """

    budget: int = 1
    delivery: HedgeDelivery = HedgeDelivery.FIRST_CONFORMING
    delay: Optional[float] = None
    max_hedges_per_item: Optional[int] = None
    strategy: AllocationStrategy = AllocationStrategy.FAIR


@dataclass(frozen=True)
class CandidateReport:
    """One redundant execution kept under a single request item."""

    candidate_id: str
    role: str
    state: CandidateState
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    batch_id: Optional[str] = None
    batch_index: Optional[int] = None
    batch_size: Optional[int] = None
    result: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    committed: bool = False
    completed: bool = False


@dataclass(frozen=True)
class DeliveryRecord:
    """The single, immutable adoption decision for one request item."""

    candidate_id: str
    mode: HedgeDelivery
    reason: str
    result: Any = None


@dataclass(frozen=True)
class HedgeBudgetReport:
    """Group-wide accounting of redundant execution work."""

    total: int
    pending: int
    committed: int
    completed: int
    # Hedge candidates that entered a real batch. This work is never
    # erased by later failure, cancellation, or a fixed delivery.
    executed: int = 0
    # Candidates that actually entered a batch; real work is never erased
    # by later failure or cancellation.
    executed: int = 0

    @property
    def used(self) -> int:
        return self.pending + self.committed + self.completed

    @property
    def remaining(self) -> int:
        return self.total - self.used


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
    candidates: tuple[CandidateReport, ...] = ()
    delivery: Optional[DeliveryRecord] = None
    unmet_reason: Optional[str] = None
    # Caller-supplied delivery deadline in seconds after group submission.
    deadline: Optional[float] = None
    deadline: Optional[float] = None


@dataclass(frozen=True)
class GroupReport:
    group_id: str
    contract: VersionContract
    pinned_version: Optional[str]
    state: GroupState
    version_conformant: bool
    executed_versions: tuple[str, ...]
    items: tuple[ItemReport, ...]
    hedge_delivery: Optional[HedgeDelivery] = None
    hedge_budget: Optional[HedgeBudgetReport] = None

    @property
    def complete(self) -> bool:
        return self.state is GroupState.EXECUTED


@dataclass
class _Candidate:
    candidate_id: str
    role: str
    request: Optional[Request] = None
    future: Any = None
    state: CandidateState = CandidateState.PENDING
    timer: Any = None
    error: Optional[BaseException] = None

    @property
    def committed(self) -> bool:
        return self.future is not None

    @property
    def completed_work(self) -> bool:
        return self.state in (CandidateState.FINISHED, CandidateState.FAILED)


@dataclass
class _GroupItem:
    item_id: str
    request: Request
    primary: _Candidate
    hedges: list[_Candidate] = field(default_factory=list)
    delivery: Optional[DeliveryRecord] = None
    unmet_reason: Optional[str] = None
    attempts: list = field(default_factory=list)
    deadline_at: Optional[float] = None

    @property
    def candidates(self) -> list[_Candidate]:
        return [self.primary, *self.hedges]

    @property
    def future(self):
        return self.primary.future

    @property
    def pending_hedges(self) -> list[_Candidate]:
        return [hedge for hedge in self.hedges if hedge.state is CandidateState.PENDING]

    @property
    def outstanding(self) -> list[_Candidate]:
        live = [
            candidate
            for candidate in self.candidates
            if candidate.committed and not candidate.future.done()
        ]
        return live + self.pending_hedges


class RequestGroup:
    """A coordinated set of futures governed by one version agreement.

    With a :class:`HedgeConfig` the group may start redundant executions for
    items that have not returned. Each redundant execution stays a separate
    candidate of the same item, while each item makes exactly one delivery
    decision and therefore counts once.
    """

    def __init__(
        self,
        executor: "DynamicBatchingExecutor",
        requests: list[Request],
        contract: VersionContract = VersionContract.SAME_VERSION,
        group_id: Optional[str] = None,
        pinned_version: Optional[str] = None,
        submit_timeout: Optional[float] = -1,
        hedge: Optional[HedgeConfig] = None,
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
        self._hedge = hedge
        self._hedge_clock = getattr(executor, "_clock", time.monotonic)
        self._submitted_at = self._hedge_clock()
        self._budget_pool = hedge.budget if hedge is not None else 0
        self._items: dict[str, _GroupItem] = {}
        self._saved_reports: dict[str, GroupReport] = {}
        self._submit_timeout = submit_timeout

        required_version = (
            self.pinned_version if contract is VersionContract.SAME_VERSION else None
        )
        for request in requests:
            submission = self._pin_request(request, required_version)
            primary = _Candidate(
                candidate_id=uuid.uuid4().hex,
                role="primary",
                request=submission,
            )
            item = _GroupItem(
                item_id=submission.request_id,
                request=submission,
                primary=primary,
                deadline_at=(
                    None
                    if submission.deadline is None
                    else self._submitted_at + submission.deadline
                ),
            )
            self._items[submission.request_id] = item
            self._commit_candidate(item, primary)

        if hedge is not None and hedge.budget > 0:
            self._schedule_hedges()

    def _pin_request(self, request: Request, required_version: Optional[str]) -> Request:
        if required_version is None:
            return request
        if request.required_model_version not in (None, required_version):
            raise VersionContractError(
                f"request requires model version {request.required_model_version!r}; "
                f"group pins {required_version!r}"
            )
        return Request(
            data=request.data,
            shape=request.shape,
            dtype=request.dtype,
            priority=request.priority,
            required_model_version=required_version,
            request_id=request.request_id,
            deadline=request.deadline,
            max_redundancy=request.max_redundancy,
        )

    def _new_hedge_request(self, template: Request) -> Request:
        required_version = (
            self.pinned_version if self.contract is VersionContract.SAME_VERSION else None
        )
        return Request(
            data=template.data,
            shape=template.shape,
            dtype=template.dtype,
            priority=template.priority,
            required_model_version=required_version,
            deadline=template.deadline,
            max_redundancy=template.max_redundancy,
        )

    # -- redundant execution -------------------------------------------------

    def _schedule_hedges(self) -> None:
        with self._lock:
            self._allocate_budget_locked()

    def _item_redundancy_cap(self, item: _GroupItem) -> Optional[int]:
        caps = []
        if self._hedge.max_hedges_per_item is not None:
            caps.append(self._hedge.max_hedges_per_item)
        if item.request.max_redundancy is not None:
            caps.append(item.request.max_redundancy)
        return min(caps) if caps else None

    def _item_deadline_passed(self, item: _GroupItem) -> bool:
        return item.deadline_at is not None and self._hedge_clock() >= item.deadline_at

    def _item_eligible_for_hedge(self, item: _GroupItem) -> bool:
        if item.delivery is not None:
            return False
        if self._item_deadline_passed(item):
            return False
        cap = self._item_redundancy_cap(item)
        if cap is not None and len(item.hedges) >= cap:
            return False
        if all(
            candidate.state is CandidateState.CANCELLED
            for candidate in item.candidates
        ):
            return False
        return True

    def _next_allocation_item(self) -> Optional[_GroupItem]:
        eligible = [
            item for item in self._items.values() if self._item_eligible_for_hedge(item)
        ]
        if not eligible:
            return None
        if self._hedge.strategy is AllocationStrategy.DEADLINE_FIRST:
            # Rescue the item closest to its delivery deadline first; items
            # without a deadline are served after every deadline-bound item.
            eligible.sort(
                key=lambda item: (
                    item.deadline_at is None,
                    item.deadline_at if item.deadline_at is not None else 0.0,
                    len(item.hedges),
                )
            )
        else:
            # Fair: the item with the fewest hedges so far is served next,
            # so a group-wide budget is spread evenly across items.
            eligible.sort(key=lambda item: len(item.hedges))
        return eligible[0]

    def _allocate_budget_locked(self) -> None:
        hedge = self._hedge
        if hedge is None or self._executor is None:
            return
        delay = hedge.delay if hedge.delay is not None else 0.05
        while self._budget_pool > 0:
            item = self._next_allocation_item()
            if item is None:
                break
            index = len(item.hedges)
            candidate = _Candidate(
                candidate_id=uuid.uuid4().hex,
                role=f"hedge-{index + 1}",
            )
            item.hedges.append(candidate)
            # The slot is promised to exactly this candidate; it returns to
            # the pool only if the candidate is withdrawn before starting.
            self._budget_pool -= 1
            item.unmet_reason = None
            timer = threading.Timer(
                delay * (index + 1),
                self._start_hedge,
                args=(item.item_id, candidate.candidate_id),
            )
            timer.daemon = True
            candidate.timer = timer
            timer.start()

    def _start_hedge(self, item_id: str, candidate_id: str) -> None:
        with self._lock:
            if self._executor is None:
                return
            item = self._items.get(item_id)
            if item is None:
                return
            candidate = next(
                (hedge for hedge in item.hedges if hedge.candidate_id == candidate_id),
                None,
            )
            if candidate is None or candidate.state is not CandidateState.PENDING:
                return
            # A fixed FIRST-mode delivery no longer needs this execution; a
            # candidate that never started can be withdrawn and frees budget.
            if item.delivery is not None:
                self._withdraw_pending(item, candidate)
                return
            if self._hedge.delivery is HedgeDelivery.CONSISTENT and item.delivery is not None:
                self._withdraw_pending(item, candidate)
                return
            try:
                self._commit_candidate(item, candidate)
            except EngineStoppedError:
                candidate.state = CandidateState.FAILED
                candidate.error = EngineStoppedError(
                    "executor stopped before redundant execution started"
                )
            self._evaluate_item(item)

    def _withdraw_pending(self, item: _GroupItem, candidate: _Candidate) -> None:
        if candidate.timer is not None:
            candidate.timer.cancel()
            candidate.timer = None
        if candidate.state is not CandidateState.PENDING:
            return
        candidate.state = CandidateState.CANCELLED
        # A candidate withdrawn before starting never consumed real work;
        # its budget slot returns to the pool for another item.
        self._budget_pool += 1

    def _commit_candidate(self, item: _GroupItem, candidate: _Candidate) -> None:
        if candidate.timer is not None:
            candidate.timer.cancel()
            candidate.timer = None
        request = candidate.request
        if request is None:
            request = self._new_hedge_request(item.request)
            candidate.request = request
        candidate.future = self._executor.submit(request, self._submit_timeout)
        candidate.state = CandidateState.COMMITTED
        item.attempts.append((request, candidate.future))
        candidate.future.add_done_callback(
            lambda _future, target_item=item.item_id: self._on_candidate_done(target_item)
        )

    def _on_candidate_done(self, item_id: str) -> None:
        with self._lock:
            item = self._items.get(item_id)
            if item is not None:
                self._evaluate_item(item)

    def _sync_candidate(self, candidate: _Candidate) -> None:
        future = candidate.future
        if future is None or not future.done():
            return
        if candidate.state in (
            CandidateState.FINISHED,
            CandidateState.FAILED,
            CandidateState.CANCELLED,
        ):
            return
        if future.state is RequestState.FINISHED:
            candidate.state = CandidateState.FINISHED
            candidate.error = None
        elif future.state is RequestState.CANCELLED:
            candidate.state = CandidateState.CANCELLED
        else:
            candidate.state = CandidateState.FAILED
            try:
                future.result(0)
            except BaseException as exc:  # noqa: BLE001 - recorded per candidate
                candidate.error = exc

    def _candidate_conforms(self, candidate: _Candidate) -> bool:
        future = candidate.future
        if future is None or future.state is not RequestState.FINISHED:
            return False
        attribution = future.attribution
        if attribution is None:
            return False
        if self.contract is VersionContract.SAME_VERSION:
            return attribution.model_version == self.pinned_version
        return True

    def _evaluate_item(self, item: _GroupItem) -> None:
        if self._hedge is None:
            return
        for candidate in item.candidates:
            self._sync_candidate(candidate)

        # A fixed adoption decision (or a fixed impossibility) is never
        # rewritten by execution records arriving afterwards.
        if item.delivery is not None:
            self._withdraw_obsolete_pending(item)
            self._allocate_budget_locked()
            return

        if self._item_deadline_passed(item):
            # The delivery deadline is gone: stop promising new budget to
            # this item, but never erase work already committed or done.
            self._withdraw_obsolete_pending(item)
            if item.unmet_reason is None:
                item.unmet_reason = "deadline_exceeded"

        conforming = [
            candidate
            for candidate in item.candidates
            if self._candidate_conforms(candidate)
        ]

        if self._hedge.delivery is HedgeDelivery.FIRST_CONFORMING:
            if conforming:
                chosen = conforming[0]
                item.delivery = DeliveryRecord(
                    candidate_id=chosen.candidate_id,
                    mode=HedgeDelivery.FIRST_CONFORMING,
                    reason="first_conforming_result",
                    result=chosen.future.result(0),
                )
                item.unmet_reason = None
                self._withdraw_obsolete_pending(item)
                self._allocate_budget_locked()
                return
        else:
            if len(conforming) >= 2:
                values = [candidate.future.result(0) for candidate in conforming]
                if all(value == values[0] for value in values[1:]):
                    chosen = conforming[0]
                    item.delivery = DeliveryRecord(
                        candidate_id=chosen.candidate_id,
                        mode=HedgeDelivery.CONSISTENT,
                        reason="consistent_results",
                        result=values[0],
                    )
                    item.unmet_reason = None
                    self._withdraw_obsolete_pending(item)
                    self._allocate_budget_locked()
                    return
                if not item.outstanding:
                    if item.unmet_reason is None:
                        item.unmet_reason = "redundant_results_mismatch"
                    self._withdraw_obsolete_pending(item)
                    self._allocate_budget_locked()
                    return
        if not item.outstanding and item.unmet_reason is None:
            # No delivery and nothing left to wait for: record why the
            # item could not be delivered instead of staying silent.
            if conforming:
                item.unmet_reason = "insufficient_conforming_results"
            elif any(
                candidate.state is CandidateState.FINISHED
                and candidate.future is not None
                and candidate.future.attribution is not None
                and not self._version_matches(candidate.future.attribution.model_version)
                for candidate in item.candidates
            ):
                item.unmet_reason = "model_version_ineligible"
            elif all(
                candidate.state is CandidateState.CANCELLED
                for candidate in item.candidates
            ):
                item.unmet_reason = "all_candidates_cancelled"
            else:
                item.unmet_reason = "insufficient_conforming_results"
        self._allocate_budget_locked()

    def _version_matches(self, version: str) -> bool:
        return (
            self.contract is not VersionContract.SAME_VERSION
            or version == self.pinned_version
        )

    def _withdraw_obsolete_pending(self, item: _GroupItem) -> None:
        for hedge in item.pending_hedges:
            self._withdraw_pending(item, hedge)

    @classmethod
    def from_report(cls, report: GroupReport) -> "RequestGroup":
        group = object.__new__(cls)
        group._executor = None
        group._lock = threading.RLock()
        group.group_id = report.group_id
        group.contract = report.contract
        group.pinned_version = report.pinned_version
        group._hedge = (
            HedgeConfig(
                budget=report.hedge_budget.total,
                delivery=report.hedge_delivery,
            )
            if report.hedge_delivery is not None
            else None
        )
        group._hedge_clock = time.monotonic
        group._items = {}
        group._saved_reports = {"reopened": report}
        group._submit_timeout = None
        group._budget_pool = 0
        group._submitted_at = group._hedge_clock()
        return group

    def refresh(self) -> GroupReport:
        with self._lock:
            if not self._items:
                return self._saved_reports["reopened"]
            if self._hedge is not None:
                for item in self._items.values():
                    self._evaluate_item(item)
            reports = tuple(self._item_report(item) for item in self._items.values())
            report = self._build_report(reports)
            return report

    def report(self) -> GroupReport:
        return self.refresh()

    def budget(self) -> Optional[HedgeBudgetReport]:
        return self.report().hedge_budget

    def wait(self, timeout: Optional[float] = None) -> GroupReport:
        report = self.refresh()
        deadline = None if timeout is None else self._hedge_clock() + timeout
        while not self._terminal(report):
            report = self.refresh()
            if self._terminal(report):
                break
            if deadline is not None:
                remaining = deadline - self._hedge_clock()
                if remaining <= 0:
                    return report
                threading.Event().wait(min(0.01, max(0.0, remaining)))
                continue
            threading.Event().wait(0.01)
        return report

    def cancel_remaining(self) -> GroupReport:
        with self._lock:
            if self._executor is None:
                raise EngineStoppedError("reopened group records cannot be modified")
            for item in self._items.values():
                if self._hedge is not None:
                    if item.delivery is not None:
                        continue
                    for candidate in item.candidates:
                        self._cancel_candidate(item, candidate)
                    self._evaluate_item(item)
                    continue
                if not item.future.done():
                    item.future.cancel(
                        RequestCancelledError("remaining group request cancelled")
                    )
                    self._executor.cancel(item.request.request_id)
            return self.report()

    def _cancel_candidate(self, item: _GroupItem, candidate: _Candidate) -> None:
        if candidate.state is CandidateState.PENDING:
            self._withdraw_pending(item, candidate)
            return
        if candidate.committed and not candidate.future.done():
            candidate.future.cancel(
                RequestCancelledError("redundant execution cancelled")
            )
            self._executor.cancel(candidate.request.request_id)

    def retry_unfinished(self, allow_version_switch: bool = False) -> GroupReport:
        """Submit replacement futures for failed or undelivered items only."""
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

            if self._hedge is None:
                self._retry_legacy()
            else:
                self._retry_hedged()
            return self.report()

    def _retry_legacy(self) -> None:
        for item in self._items.values():
            if item.future.state not in (RequestState.FAILED, RequestState.TIMED_OUT):
                continue
            request = self._new_hedge_request(item.request)
            if self.contract is not VersionContract.SAME_VERSION:
                request = Request(
                    data=request.data,
                    shape=request.shape,
                    dtype=request.dtype,
                    priority=request.priority,
                    deadline=item.request.deadline,
                    max_redundancy=item.request.max_redundancy,
                )
            item.request = request
            candidate = _Candidate(
                candidate_id=uuid.uuid4().hex,
                role=f"retry-{len(item.attempts)}",
                request=request,
            )
            item.primary = candidate
            item.hedges = []
            self._commit_candidate(item, candidate)

    def _retry_hedged(self) -> None:
        for item in self._items.values():
            if item.delivery is not None:
                # An adopted item is never re-run; its decision is fixed.
                continue
            request = self._new_hedge_request(item.request)
            item.request = request
            candidate = _Candidate(
                candidate_id=uuid.uuid4().hex,
                role="primary",
                request=request,
            )
            item.primary = candidate
            item.hedges = []
            item.unmet_reason = None
            self._commit_candidate(item, candidate)
        self._schedule_hedges()

    def save(self, path) -> GroupReport:
        report = self.report()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = _report_to_jsonable(report)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with self._lock:
            self._saved_reports[str(path)] = report
        return report

    def _item_report(self, item: _GroupItem) -> ItemReport:
        if self._hedge is None:
            return self._legacy_item_report(item)

        candidates = tuple(self._candidate_report(candidate) for candidate in item.candidates)
        delivered = item.delivery is not None
        future = item.primary.future
        state = ItemState.WAITING
        result = None
        error = None
        error_type = None
        attribution = None
        if delivered:
            state = ItemState.EXECUTED
            result = item.delivery.result
            chosen = next(
                candidate
                for candidate in item.candidates
                if candidate.candidate_id == item.delivery.candidate_id
            )
            attribution = chosen.future.attribution
        elif not item.outstanding:
            if any(
                candidate.state is CandidateState.CANCELLED
                and not candidate.committed
                for candidate in item.candidates
            ) and all(
                candidate.state is CandidateState.CANCELLED
                for candidate in item.candidates
            ):
                state = ItemState.CANCELLED
            elif item.unmet_reason == "all_candidates_cancelled":
                state = ItemState.CANCELLED
            elif all(
                not candidate.committed
                for candidate in item.candidates
            ):
                state = ItemState.CANCELLED
            else:
                state = ItemState.FAILED
                error = item.unmet_reason or "delivery conditions not satisfied"
                error_type = "DeliveryConditionError"
        attempts = tuple(
            attempt.attribution
            for _, attempt in item.attempts
            if attempt.attribution is not None
        )
        return ItemReport(
            item_id=item.item_id,
            request_id=item.item_id,
            state=state,
            result=result,
            error=error,
            error_type=error_type,
            attribution=attribution,
            attempts=attempts,
            candidates=candidates,
            delivery=item.delivery,
            unmet_reason=None if delivered else item.unmet_reason,
            deadline=item.request.deadline,
        )

    def _candidate_report(self, candidate: _Candidate) -> CandidateReport:
        future = candidate.future
        attribution = future.attribution if future is not None else None
        result = None
        error = None
        error_type = None
        if future is not None and future.done():
            try:
                result = future.result(0)
            except BaseException as exc:  # noqa: BLE001
                error = str(exc)
                error_type = type(exc).__name__
        return CandidateReport(
            candidate_id=candidate.candidate_id,
            role=candidate.role,
            state=candidate.state,
            model_name=attribution.model_name if attribution else None,
            model_version=attribution.model_version if attribution else None,
            batch_id=attribution.batch_id if attribution else None,
            batch_index=attribution.batch_index if attribution else None,
            batch_size=attribution.batch_size if attribution else None,
            result=result,
            error=error,
            error_type=error_type,
            committed=candidate.committed,
            completed=candidate.completed_work,
        )

    def _legacy_item_report(self, item: _GroupItem) -> ItemReport:
        future = item.primary.future
        state = _item_state(future.state)
        result = None
        error = None
        error_type = None
        if future.done():
            try:
                result = future.result(0)
            except BaseException as exc:  # noqa: BLE001
                error = str(exc)
                error_type = type(exc).__name__
        return ItemReport(
            item_id=item.item_id,
            request_id=item.request.request_id,
            state=state,
            result=result,
            error=error,
            error_type=error_type,
            attribution=future.attribution,
            attempts=tuple(
                attempt.attribution
                for _, attempt in item.attempts
                if attempt.attribution is not None
            ),
        )

    def _hedge_budget(self, reports: tuple[ItemReport, ...]) -> HedgeBudgetReport:
        pending = 0
        committed = 0
        completed = 0
        executed = 0
        for item in self._items.values():
            for hedge in item.hedges:
                if hedge.state is CandidateState.PENDING:
                    pending += 1
                elif hedge.completed_work:
                    completed += 1
                elif hedge.committed:
                    committed += 1
                # A candidate withdrawn before commit returned its slot to
                # the pool and accounts for no budget here.
                if hedge.future is not None and hedge.future.attribution is not None:
                    # The candidate entered a real batch; that work stands
                    # even if the candidate later fails or is cancelled.
                    executed += 1
        return HedgeBudgetReport(
            total=self._hedge.budget,
            pending=pending,
            committed=committed,
            completed=completed,
            executed=executed,
        )

    def _build_report(self, reports: tuple[ItemReport, ...]) -> GroupReport:
        versions = tuple(
            dict.fromkeys(
                item.attribution.model_version
                for item in reports
                if item.attribution is not None
            )
        )
        if self._hedge is not None:
            # Only the version of the uniquely adopted result matters.
            versions = tuple(
                dict.fromkeys(
                    report.attribution.model_version
                    for report in reports
                    if report.state is ItemState.EXECUTED and report.attribution is not None
                )
            )
        version_conformant = True
        if self.contract is VersionContract.SAME_VERSION:
            expected = self.pinned_version
            version_conformant = all(
                item.attribution is not None
                and item.attribution.model_version == expected
                for item in reports
                if item.state is ItemState.EXECUTED
            ) and (not versions or versions == (expected,))

        has_failed = any(item.state is ItemState.FAILED for item in reports)
        has_cancelled = any(item.state is ItemState.CANCELLED for item in reports)
        has_waiting = any(item.state is ItemState.WAITING for item in reports)
        all_executed = all(item.state is ItemState.EXECUTED for item in reports)

        if all_executed and version_conformant:
            state = GroupState.EXECUTED
        elif has_failed or not version_conformant:
            state = GroupState.FAILED
        elif has_waiting:
            state = GroupState.WAITING
        elif has_cancelled:
            state = GroupState.CANCELLED
        else:
            state = GroupState.FAILED

        return GroupReport(
            group_id=self.group_id,
            contract=self.contract,
            pinned_version=self.pinned_version,
            state=state,
            version_conformant=version_conformant,
            executed_versions=versions,
            items=reports,
            hedge_delivery=self._hedge.delivery if self._hedge is not None else None,
            hedge_budget=self._hedge_budget(reports) if self._hedge is not None else None,
        )

    @staticmethod
    def _terminal(report: GroupReport) -> bool:
        return report.state in (GroupState.EXECUTED, GroupState.FAILED, GroupState.CANCELLED)


class GroupStore:
    """JSON-backed group snapshot storage."""

    def save(self, path, group: RequestGroup) -> GroupReport:
        return group.save(path)

    def load(self, path) -> RequestGroup:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return RequestGroup.from_report(_report_from_jsonable(payload))


def _item_state(state: RequestState) -> ItemState:
    if state is RequestState.FINISHED:
        return ItemState.EXECUTED
    if state in (RequestState.FAILED, RequestState.TIMED_OUT):
        return ItemState.FAILED
    if state is RequestState.CANCELLED:
        return ItemState.CANCELLED
    return ItemState.WAITING


def _attribution_to_jsonable(value: Optional[BatchAttribution]) -> Optional[dict[str, Any]]:
    return None if value is None else asdict(value)


def _report_to_jsonable(report: GroupReport) -> dict[str, Any]:
    items = []
    for item in report.items:
        entry = {
            "item_id": item.item_id,
            "request_id": item.request_id,
            "state": item.state.value,
            "result": item.result,
            "error": item.error,
            "error_type": item.error_type,
            "attribution": _attribution_to_jsonable(item.attribution),
            "attempts": [_attribution_to_jsonable(value) for value in item.attempts],
            "deadline": item.deadline,
        }
        if report.hedge_delivery is not None:
            entry["unmet_reason"] = item.unmet_reason
            entry["delivery"] = (
                None
                if item.delivery is None
                else {
                    "candidate_id": item.delivery.candidate_id,
                    "mode": item.delivery.mode.value,
                    "reason": item.delivery.reason,
                    "result": item.delivery.result,
                }
            )
            entry["candidates"] = [
                {
                    "candidate_id": candidate.candidate_id,
                    "role": candidate.role,
                    "state": candidate.state.value,
                    "model_name": candidate.model_name,
                    "model_version": candidate.model_version,
                    "batch_id": candidate.batch_id,
                    "batch_index": candidate.batch_index,
                    "batch_size": candidate.batch_size,
                    "result": candidate.result,
                    "error": candidate.error,
                    "error_type": candidate.error_type,
                    "committed": candidate.committed,
                    "completed": candidate.completed,
                }
                for candidate in item.candidates
            ]
        items.append(entry)
    payload = {
        "group_id": report.group_id,
        "contract": report.contract.value,
        "pinned_version": report.pinned_version,
        "state": report.state.value,
        "version_conformant": report.version_conformant,
        "executed_versions": list(report.executed_versions),
        "items": items,
    }
    if report.hedge_delivery is not None:
        payload["hedge_delivery"] = report.hedge_delivery.value
        payload["hedge_budget"] = asdict(report.hedge_budget)
    return payload


def _attribution_from_jsonable(value: Optional[dict[str, Any]]) -> Optional[BatchAttribution]:
    if value is None:
        return None
    return BatchAttribution(**value)


def _report_from_jsonable(payload: dict[str, Any]) -> GroupReport:
    hedge_delivery = (
        HedgeDelivery(payload["hedge_delivery"]) if payload.get("hedge_delivery") else None
    )
    items = []
    for item in payload["items"]:
        candidates = ()
        delivery = None
        if hedge_delivery is not None:
            candidates = tuple(
                CandidateReport(
                    candidate_id=candidate["candidate_id"],
                    role=candidate["role"],
                    state=CandidateState(candidate["state"]),
                    model_name=candidate.get("model_name"),
                    model_version=candidate.get("model_version"),
                    batch_id=candidate.get("batch_id"),
                    batch_index=candidate.get("batch_index"),
                    batch_size=candidate.get("batch_size"),
                    result=candidate.get("result"),
                    error=candidate.get("error"),
                    error_type=candidate.get("error_type"),
                    committed=candidate.get("committed", False),
                    completed=candidate.get("completed", False),
                )
                for candidate in item.get("candidates", [])
            )
            saved_delivery = item.get("delivery")
            if saved_delivery is not None:
                delivery = DeliveryRecord(
                    candidate_id=saved_delivery["candidate_id"],
                    mode=HedgeDelivery(saved_delivery["mode"]),
                    reason=saved_delivery["reason"],
                    result=saved_delivery.get("result"),
                )
        items.append(
            ItemReport(
                item_id=item["item_id"],
                request_id=item["request_id"],
                state=ItemState(item["state"]),
                result=item.get("result"),
                error=item.get("error"),
                error_type=item.get("error_type"),
                attribution=_attribution_from_jsonable(item.get("attribution")),
                attempts=tuple(
                    value
                    for value in (
                        _attribution_from_jsonable(attempt)
                        for attempt in item.get("attempts", [])
                    )
                    if value is not None
                ),
                candidates=candidates,
                delivery=delivery,
                unmet_reason=item.get("unmet_reason"),
                deadline=item.get("deadline"),
            )
        )
    hedge_budget = None
    if payload.get("hedge_budget"):
        hedge_budget = HedgeBudgetReport(**payload["hedge_budget"])
    return GroupReport(
        group_id=payload["group_id"],
        contract=VersionContract(payload["contract"]),
        pinned_version=payload.get("pinned_version"),
        state=GroupState(payload["state"]),
        version_conformant=payload["version_conformant"],
        executed_versions=tuple(payload.get("executed_versions", [])),
        items=tuple(items),
        hedge_delivery=hedge_delivery,
        hedge_budget=hedge_budget,
    )
