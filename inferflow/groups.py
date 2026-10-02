from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Sequence

from .errors import VersionConflictError
from .executor import DynamicBatchingExecutor
from .types import BatchAttribution, PredictionFuture, Request, RequestState

_TERMINAL_STATES = frozenset(
    {
        RequestState.FINISHED,
        RequestState.FAILED,
        RequestState.CANCELLED,
        RequestState.TIMED_OUT,
    }
)


class VersionPolicy(Enum):
    """Delivery contract chosen by the caller for a request group."""

    # Every item must execute on the model version that was active when the
    # group was submitted. A version split inside the group is a contract
    # violation and is reported as such, never as success.
    PINNED = "pinned"
    # Each item may execute on whichever model version is current when its
    # batch forms; mixed versions inside one group are expected and allowed.
    FLEXIBLE = "flexible"


class GroupConclusion(Enum):
    """Stable, terminal-or-progressing verdict for a whole group."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"  # all items finished and the contract holds
    COMPLETED_PARTIAL = "completed_partial"  # terminal, some items not finished
    CONTRACT_VIOLATED = "contract_violated"  # pinned group saw a version split
    CANCELLED = "cancelled"  # caller cancelled the remaining items


@dataclass(frozen=True)
class GroupItemReport:
    """Point-in-time view of one group member.

    ``explanation`` is always set for terminal non-finished states, so a
    report never shows a failure or cancellation without a reason.
    """

    index: int
    request_id: str
    state: RequestState
    attribution: Optional[BatchAttribution]
    result: Any = None
    explanation: Optional[str] = None
    attempts: int = 1


@dataclass(frozen=True)
class GroupReport:
    """Immutable snapshot of a group; safe to read repeatedly or persist."""

    group_id: str
    policy: VersionPolicy
    pinned_version: Optional[str]
    items: tuple[GroupItemReport, ...]
    conclusion: GroupConclusion
    contract_satisfied: bool
    versions_observed: tuple[str, ...]
    contract_amended: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "policy": self.policy.value,
            "pinned_version": self.pinned_version,
            "conclusion": self.conclusion.value,
            "contract_satisfied": self.contract_satisfied,
            "versions_observed": list(self.versions_observed),
            "contract_amended": self.contract_amended,
            "items": [
                {
                    "index": item.index,
                    "request_id": item.request_id,
                    "state": item.state.value,
                    "attribution": (
                        {
                            "batch_id": item.attribution.batch_id,
                            "model_name": item.attribution.model_name,
                            "model_version": item.attribution.model_version,
                            "batch_size": item.attribution.batch_size,
                            "batch_index": item.attribution.batch_index,
                        }
                        if item.attribution
                        else None
                    ),
                    "result": item.result,
                    "explanation": item.explanation,
                    "attempts": item.attempts,
                }
                for item in self.items
            ],
        }

    @staticmethod
    def from_dict(record: dict[str, Any]) -> "GroupReport":
        """Rebuild a report from storage.

        The conclusion is recomputed from the item states, so a stored record
        that claims overall completion while containing unexplained failed
        items is never presented as COMPLETED after reopening.
        """
        items = tuple(
            GroupItemReport(
                index=raw["index"],
                request_id=raw["request_id"],
                state=RequestState(raw["state"]),
                attribution=(
                    BatchAttribution(**raw["attribution"])
                    if raw.get("attribution")
                    else None
                ),
                result=_restore_result(raw.get("result")),
                explanation=raw.get("explanation"),
                attempts=raw.get("attempts", 1),
            )
            for raw in record["items"]
        )
        policy = VersionPolicy(record["policy"])
        conclusion, satisfied = _derive_conclusion(
            [(item.state, item.attribution) for item in items],
            policy=policy,
            pinned_version=record.get("pinned_version"),
            cancelled=False,
            amended=bool(record.get("contract_amended", False)),
        )
        return GroupReport(
            group_id=record["group_id"],
            policy=policy,
            pinned_version=record.get("pinned_version"),
            items=items,
            conclusion=conclusion,
            contract_satisfied=satisfied,
            versions_observed=_observed_versions([i.attribution for i in items]),
            contract_amended=bool(record.get("contract_amended", False)),
        )


def _restore_result(result: Any) -> Any:
    # JSON stores tuples as lists; restore them so reopened reports compare
    # equal to the live snapshots they were saved from.
    if isinstance(result, list):
        return tuple(result)
    return result


def _observed_versions(attributions: Sequence[Optional[BatchAttribution]]) -> tuple[str, ...]:
    seen: list[str] = []
    for attribution in attributions:
        if attribution and attribution.model_version not in seen:
            seen.append(attribution.model_version)
    return tuple(seen)


def _derive_conclusion(
    states_and_attributions: Sequence[tuple[RequestState, Optional[BatchAttribution]]],
    policy: VersionPolicy,
    pinned_version: Optional[str],
    cancelled: bool,
    amended: bool,
) -> tuple[GroupConclusion, bool]:
    """Single source of truth for group verdicts, used live and on reload."""
    states = [state for state, _ in states_and_attributions]
    if all(state is RequestState.FINISHED for state in states):
        if policy is VersionPolicy.PINNED and not amended:
            attributions = [attribution for _, attribution in states_and_attributions]
            versions = {a.model_version for a in attributions if a is not None}
            uniform = (
                all(a is not None for a in attributions)
                and versions == {pinned_version}
            )
            if not uniform:
                return GroupConclusion.CONTRACT_VIOLATED, False
        return GroupConclusion.COMPLETED, True
    if cancelled:
        return GroupConclusion.CANCELLED, False
    if all(state in _TERMINAL_STATES for state in states):
        return GroupConclusion.COMPLETED_PARTIAL, False
    return GroupConclusion.IN_PROGRESS, False


@dataclass
class _GroupItem:
    index: int
    request: Request
    future: PredictionFuture
    attempts: int = 1


class RequestGroup:
    """A set of related requests submitted under one delivery contract."""

    def __init__(
        self,
        executor: DynamicBatchingExecutor,
        requests: Sequence[Request],
        policy: VersionPolicy = VersionPolicy.PINNED,
        timeout: Optional[float] = -1,
        group_id: Optional[str] = None,
    ):
        if not requests:
            raise ValueError("a request group needs at least one request")
        self._executor = executor
        self._policy = policy
        self._group_id = group_id or uuid.uuid4().hex
        self._lock = threading.Lock()
        self._cancelled = False
        self._amended = False
        self._pinned_version = (
            executor.model.version if policy is VersionPolicy.PINNED else None
        )
        self._items = [
            _GroupItem(
                index=index,
                request=request,
                future=executor.submit(request, timeout=timeout),
            )
            for index, request in enumerate(requests)
        ]

    @property
    def group_id(self) -> str:
        return self._group_id

    @property
    def policy(self) -> VersionPolicy:
        return self._policy

    @property
    def pinned_version(self) -> Optional[str]:
        return self._pinned_version

    def report(self) -> GroupReport:
        """Snapshot the group; repeated calls are consistent and side-effect free."""
        with self._lock:
            items = tuple(self._item_report(item) for item in self._items)
            conclusion, satisfied = _derive_conclusion(
                [(item.state, item.attribution) for item in items],
                policy=self._policy,
                pinned_version=self._pinned_version,
                cancelled=self._cancelled,
                amended=self._amended,
            )
            return GroupReport(
                group_id=self._group_id,
                policy=self._policy,
                pinned_version=self._pinned_version,
                items=items,
                conclusion=conclusion,
                contract_satisfied=satisfied,
                versions_observed=_observed_versions([i.attribution for i in items]),
                contract_amended=self._amended,
            )

    def cancel_remaining(self) -> int:
        """Cancel every item that has not reached a terminal state.

        Finished items keep their results and batch/version identity. Returns
        the number of items for which a cancellation was delivered.
        """
        delivered = 0
        with self._lock:
            self._cancelled = True
            pending = [item for item in self._items if not item.future.done()]
        for item in pending:
            if self._executor.cancel(item.request.request_id):
                delivered += 1
            else:
                item.future.cancel()
        return delivered

    def resubmit_unfinished(self, confirm_version_change: bool = False) -> list[int]:
        """Resubmit only items that ended without a result.

        Finished items are never resubmitted and keep their original batch and
        version identity. For PINNED groups, resubmission requires the pinned
        version to still be active; otherwise it is rejected with
        VersionConflictError unless the caller explicitly reconfirms the
        contract with confirm_version_change=True, which amends the contract
        and records the amendment in every later report.
        """
        with self._lock:
            if self._policy is VersionPolicy.PINNED:
                current = self._executor.model.version
                if current != self._pinned_version:
                    if not confirm_version_change:
                        raise VersionConflictError(
                            "group " + self._group_id + " is pinned to model version "
                            + repr(self._pinned_version) + " but the active version is "
                            + repr(current) + "; pass confirm_version_change=True "
                            "to amend the contract"
                        )
                    self._amended = True
            retry = [
                item
                for item in self._items
                if item.future.done() and item.future.state is not RequestState.FINISHED
            ]
            self._cancelled = False
        resubmitted: list[int] = []
        for item in retry:
            item.future = self._executor.submit(item.request)
            item.attempts += 1
            resubmitted.append(item.index)
        return resubmitted

    @staticmethod
    def _item_report(item: _GroupItem) -> GroupItemReport:
        future = item.future
        state = future.state
        explanation: Optional[str] = None
        result: Any = None
        if state is RequestState.FINISHED:
            result = future.result(0)
        elif state in _TERMINAL_STATES:
            cause = future.exception
            explanation = (
                state.value + ": " + str(cause) if cause is not None else state.value
            )
        return GroupItemReport(
            index=item.index,
            request_id=item.request.request_id,
            state=state,
            attribution=future.attribution,
            result=result,
            explanation=explanation,
            attempts=item.attempts,
        )


class JsonGroupStore:
    """JSON-file persistence for group reports.

    Saving is upsert-by-group-id; loading recomputes the conclusion from the
    stored items, so a group can never reopen as completed while still
    containing unexplained failed items.
    """

    def __init__(self, path: str):
        self._path = path
        self._lock = threading.Lock()

    def save(self, report: GroupReport) -> None:
        with self._lock:
            records = self._read_all()
            records[report.group_id] = report.to_dict()
            tmp_path = self._path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(records, handle, indent=2)
            os.replace(tmp_path, self._path)

    def load(self, group_id: str) -> GroupReport:
        with self._lock:
            records = self._read_all()
        if group_id not in records:
            raise KeyError("unknown group id: " + group_id)
        return GroupReport.from_dict(records[group_id])

    def load_all(self) -> list[GroupReport]:
        with self._lock:
            records = self._read_all()
        return [GroupReport.from_dict(record) for record in records.values()]

    def _read_all(self) -> dict[str, Any]:
        if not os.path.exists(self._path):
            return {}
        with open(self._path, "r", encoding="utf-8") as handle:
            return json.load(handle)
