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
    GroupState,
    ItemState,
    PredictionFuture,
    Request,
    RequestState,
    VersionContract,
)

if TYPE_CHECKING:
    from .executor import DynamicBatchingExecutor


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


@dataclass(frozen=True)
class GroupReport:
    group_id: str
    contract: VersionContract
    pinned_version: Optional[str]
    state: GroupState
    version_conformant: bool
    executed_versions: tuple[str, ...]
    items: tuple[ItemReport, ...]

    @property
    def complete(self) -> bool:
        return self.state is GroupState.EXECUTED


@dataclass
class _GroupItem:
    item_id: str
    request: Request
    future: PredictionFuture
    attempts: list[tuple[Request, PredictionFuture]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.attempts.append((self.request, self.future))


class RequestGroup:
    """A coordinated set of futures governed by one version agreement."""

    def __init__(
        self,
        executor: "DynamicBatchingExecutor",
        requests: list[Request],
        contract: VersionContract = VersionContract.SAME_VERSION,
        group_id: Optional[str] = None,
        pinned_version: Optional[str] = None,
        submit_timeout: Optional[float] = -1,
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

        required_version = (
            self.pinned_version if contract is VersionContract.SAME_VERSION else None
        )
        for request in requests:
            submission = request
            if required_version is not None:
                if request.required_model_version not in (None, required_version):
                    raise VersionContractError(
                        f"request requires model version {request.required_model_version!r}; "
                        f"group pins {required_version!r}"
                    )
                submission = Request(
                    data=request.data,
                    shape=request.shape,
                    dtype=request.dtype,
                    priority=request.priority,
                    required_model_version=required_version,
                    request_id=request.request_id,
                )
            future = executor.submit(submission, submit_timeout)
            self._items[submission.request_id] = _GroupItem(submission.request_id, submission, future)

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
        return group

    def refresh(self) -> GroupReport:
        with self._lock:
            if not self._items:
                return self._saved_reports["reopened"]
            reports = tuple(self._item_report(item) for item in self._items.values())
            report = self._build_report(reports)
            return report

    def report(self) -> GroupReport:
        return self.refresh()

    def wait(self, timeout: Optional[float] = None) -> GroupReport:
        report = self.refresh()
        deadline = None if timeout is None else time.monotonic() + timeout
        # Events are per-future. A short polling interval keeps this public API
        # dependency-free while timeout support remains exact at completion.
        while not self._terminal(report):
            report = self.refresh()
            if self._terminal(report):
                break
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return report
                threading.Event().wait(min(0.01, remaining))
                continue
            threading.Event().wait(0.01)
        return report

    def cancel_remaining(self) -> GroupReport:
        with self._lock:
            if self._executor is None:
                raise EngineStoppedError("reopened group records cannot be modified")
            for item in self._items.values():
                if not item.future.done():
                    item.future.cancel(
                        RequestCancelledError("remaining group request cancelled")
                    )
                    self._executor.cancel(item.request.request_id)
            return self.report()

    def retry_unfinished(self, allow_version_switch: bool = False) -> GroupReport:
        """Submit replacement futures for failed/timed-out item attempts."""
        with self._lock:
            if self._executor is None:
                raise EngineStoppedError("reopened group records cannot be modified")
            report = self.report()
            current_version = self._executor.model.version
            if self.contract is VersionContract.SAME_VERSION and current_version != self.pinned_version:
                if not allow_version_switch:
                    raise VersionContractError(
                        f"group requires model version {self.pinned_version!r}; "
                        f"current model is {current_version!r}"
                    )
                self.contract = VersionContract.CURRENT_VERSION
                self.pinned_version = None

            for item in self._items.values():
                if item.future.state not in (RequestState.FAILED, RequestState.TIMED_OUT):
                    continue
                request = item.request
                required_version = (
                    self.pinned_version
                    if self.contract is VersionContract.SAME_VERSION
                    else None
                )
                if required_version is not None:
                    request = Request(
                        data=request.data,
                        shape=request.shape,
                        dtype=request.dtype,
                        priority=request.priority,
                        required_model_version=required_version,
                    )
                else:
                    request = Request(
                        data=request.data,
                        shape=request.shape,
                        dtype=request.dtype,
                        priority=request.priority,
                    )
                item.request = request
                item.future = self._executor.submit(request, self._submit_timeout)
                item.attempts.append((request, item.future))
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

    def _item_report(self, item: _GroupItem) -> ItemReport:
        future = item.future
        state = _item_state(future.state)
        result = None
        error = None
        error_type = None
        if future.done():
            try:
                result = future.result(0)
            except BaseException as exc:
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

    def _build_report(self, reports: tuple[ItemReport, ...]) -> GroupReport:
        versions = tuple(
            dict.fromkeys(
                item.attribution.model_version
                for item in reports
                if item.attribution is not None
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
        )

    @staticmethod
    def _terminal(report: GroupReport) -> bool:
        return report.state in (GroupState.EXECUTED, GroupState.FAILED, GroupState.CANCELLED)


class GroupStore:
    """JSON-backed group snapshot storage."""

    def save(self, path: str | Path, group: RequestGroup) -> GroupReport:
        return group.save(path)

    def load(self, path: str | Path) -> RequestGroup:
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
    return {
        "group_id": report.group_id,
        "contract": report.contract.value,
        "pinned_version": report.pinned_version,
        "state": report.state.value,
        "version_conformant": report.version_conformant,
        "executed_versions": list(report.executed_versions),
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
            }
            for item in report.items
        ],
    }


def _attribution_from_jsonable(value: Optional[dict[str, Any]]) -> Optional[BatchAttribution]:
    if value is None:
        return None
    return BatchAttribution(**value)


def _report_from_jsonable(payload: dict[str, Any]) -> GroupReport:
    items = tuple(
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
        )
        for item in payload["items"]
    )
    return GroupReport(
        group_id=payload["group_id"],
        contract=VersionContract(payload["contract"]),
        pinned_version=payload.get("pinned_version"),
        state=GroupState(payload["state"]),
        version_conformant=payload["version_conformant"],
        executed_versions=tuple(payload.get("executed_versions", [])),
        items=items,
    )
