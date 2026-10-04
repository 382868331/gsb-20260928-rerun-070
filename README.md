# Inferflow

Inferflow is a small Python foundation for model inference serving. The first
iteration focuses on local, directly computable models and dynamic batching;
future adapters can follow the same request lifecycle without changing the
executor contract.

## Request lifecycle

1. Build a `Request(data, shape, dtype)`. Each request carries an ID and one
   unpadded sample.
2. Submit it to a started `DynamicBatchingExecutor` to receive a
   `PredictionFuture`.
3. The worker groups queued requests only when dtype and shape match, waits a
   short window for more compatible traffic, then enforces `max_batch_size`.
4. The selected model validates each request and receives a list of samples.
   Output list position is returned to the corresponding request; cancellation
   never shifts another request's output.
5. Futures finish as `FINISHED`, `FAILED`, `CANCELLED`, or `TIMED_OUT`.

## Execution attribution

Every `PredictionFuture` exposes `attribution`, a `BatchAttribution` bound
exactly once when the request enters a batch: `batch_id`, `model_name`,
`model_version`, `batch_size`, and the request's `batch_index`. Requests that
never reached a batch (cancelled, expired, or drained while queued) keep
`attribution is None`. `history` returns the ordered state transitions, so
queueing, batch entry, cancellation, and completion are each traceable
without instrumenting the model. During `reload_model`, batches already in
flight keep the old model's attribution while later batches bind the new
version; a failed reload changes nothing. Attribution, terminal state, and
history are stable under repeated reads, including when cancellation races
completion, a batch partially fails, or the executor stops.

## Priority and fairness

Each `Request` carries a `priority` (`Priority.LOW`, `NORMAL`, or `HIGH`).
The worker always forms the next batch around the queued request with the
highest effective priority, and only groups requests whose dtype and shape
match that anchor. To keep a continuous stream of high-priority traffic from
starving normal requests, every `aging_interval` seconds of queue wait raises
a request's effective priority by one level; ties are broken by arrival
order, so any queued request is served within a bounded time.

`stop()` fails every still-queued request with `EngineStoppedError` and, if
the worker cannot finish an in-flight batch before the join timeout, fails
those requests too; no future is left waiting forever.

## Built-in models

`LinearModel` and `AddConstantModel` are deterministic and implemented in pure
Python. They require no network access, weights download, or third-party
dependencies and are convenient for serial-vs-batched result comparison.

```python
from inferflow import DynamicBatchingExecutor, ExecutorConfig, LinearModel, Request

model = LinearModel()
config = ExecutorConfig(
    max_queue_size=128,
    max_batch_size=32,
    max_wait_time=0.02,
    default_timeout=5.0,
)

with DynamicBatchingExecutor(model, config) as executor:
    future = executor.submit(Request((1.0, 2.0, 3.0, 4.0), shape=(4,)))
    print(future.result())
```

## Adapting a model

A model adapter needs:

- `name` and `version` attributes,
- `check(request_shape, dtype)` for request validation,
- `predict(samples)` returning one output per input sample, in the same order.

`reload_model(loader)` invokes the loader outside the serving worker. A loader
failure leaves the active model untouched. When loading succeeds, the worker
finishes every already executing batch with the captured old model, then uses
the replacement for subsequent batches.

## Request groups and delivery contracts

`submit_group(requests, same_version=True)` returns a `RequestGroup`. The
default contract captures the active model version and makes every queued item
eligible only for that exact version. Pass `same_version=False` when callers
explicitly accept that some items may execute before a model switch and others
after it.

`group.report()` and `group.wait()` return an immutable `GroupReport` containing
one `ItemReport` per request. Item states are `WAITING`, `EXECUTED`, `FAILED`,
or `CANCELLED`; each executed item keeps its batch ID and model-version
attribution. A same-version group is never marked successfully complete after
observing more than one model version. The `CURRENT_VERSION` contract records a
version split separately in `executed_versions` without treating it as a
contract violation.

After a failed item, `retry_unfinished()` submits replacement futures only for
failed or timed-out items and preserves completed items. If the active model no
longer matches a pinned group, the call raises `VersionContractError`; callers
can either reload the required model or explicitly confirm the change with
`retry_unfinished(allow_version_switch=True)`. `cancel_remaining()` cancels
every still-unfinished item. Items already in a batch retain that batch's
attribution even when cancellation wins delivery.

`group.save(path)` writes a stable JSON snapshot through `GroupStore`. The
snapshot records explicit failed and cancelled items, so reloading with
`GroupStore().load(path)` cannot later report a complete group while hiding an
unexplained failure. Reopened reports are read-only.

## Redundant (hedged) execution

Pass a `HedgePolicy` to `submit_group` to give each request item one primary
execution plus a redundant candidate that starts when the item stays
undelivered past `hedge_after` seconds (or immediately when the primary fails
before any conformant result exists). Both executions belong to the same item:
each candidate keeps its own id, model/version, start condition, attribution,
and result, but an item adopts at most one result, so two candidate results are
never counted as two completed items. Redundant execution needs concurrent
batch slots; configure `ExecutorConfig(worker_count=...)` greater than one (the
default of one preserves the original single-worker behavior).

`HedgePolicy.delivery` selects the single delivery decision:

- `DeliveryMode.FIRST_CONFORMANT` (default) delivers the first candidate
  result whose model version satisfies the group contract. A fast result on a
  non-matching version is recorded but never silently downgraded into a
  delivery; the item stays waiting with `candidate_version_not_conformant` in
  its `unmet_conditions` until a conformant result exists.
- `DeliveryMode.COMPARE` delivers only once two conformant results agree. A
  mismatch fixes the item as failed with `results_disagree`; one result (or
  only failures) leaves `insufficient_conformant_results` until the redundancy
  budget is exhausted.

The model-version requirement keeps applying to every candidate separately,
for both contracts.

`max_extra_executions` is the shared group budget for extra executions. The
group report's `redundancy` (`RedundancyReport`) separates `committed_extra`
(work already promised to the executor and impossible to revoke once a batch is
attributed), `pending_extra` (candidates never started), and
`completed_extra` (extra work that actually finished). `irrevocable_executions`
counts committed batches that were already attributed when a cancel/stop raced
them; `fixed_deliveries` lists the items whose adoption decision is already
locked. After an item adopts a result, still-pending candidates are retired and
executions that finish afterwards remain recorded on their own candidate with
`late=True`, never re-triggering delivery or changing the group conclusion.

`retry_unfinished()` on a hedged group resubmits only items without an adopted
result, continuing the original delivery contract; confirmed items are never
rerun and keep the same adoption decision across retries, `save`/`load`, and
repeated queries. Groups created without a `HedgePolicy` keep the original
behavior exactly.

## Tests

```bash
python -m unittest discover -s tests -v
```
