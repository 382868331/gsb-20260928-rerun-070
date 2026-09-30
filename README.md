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

## Tests

```bash
python -m unittest discover -s tests -v
```
