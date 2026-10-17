"""OpenTelemetry adapter wrapper (optional dependency: ``atla[otel]``)."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from functools import cache
from typing import Any

from atla.adapter.base import DBAdapter
from atla.dialect import Dialect
from atla.observability.instrumented import transaction_kwargs
from atla.observability.sql_meta import extract_sql_meta

_DIALECT_SYSTEM: dict[str, str] = {
    "postgres": "postgresql",
    "cockroach": "cockroachdb",
    "mysql": "mysql",
    "tidb": "tidb",
    "sqlite": "sqlite",
}


@cache
def _require_otel() -> tuple[Any, Any, Any, Any, Any]:
    """Import OpenTelemetry symbols once; the result is cached for the process."""
    try:
        from opentelemetry import metrics, trace
        from opentelemetry.trace import SpanKind, Status, StatusCode
    except ImportError as exc:  # pragma: no cover - exercised when otel missing
        msg = "OpenTelemetry is required for OTelAdapter; install with `pip install atla[otel]`"
        raise ImportError(msg) from exc
    return metrics, trace, Status, StatusCode, SpanKind


def _record_error(span: Any, exc: BaseException) -> None:
    """Mark ``span`` failed without leaking driver error text.

    Driver messages often embed bound values (e.g. ``Duplicate entry 'a@b.c'``), so
    neither ``record_exception`` (message + stack trace) nor the status description
    may carry ``str(exc)``. Only the exception type is exported.
    """
    _, _, Status, StatusCode, _ = _require_otel()
    exc_type = type(exc)
    qualified = (
        exc_type.__qualname__
        if exc_type.__module__ == "builtins"
        else f"{exc_type.__module__}.{exc_type.__qualname__}"
    )
    span.add_event("exception", {"exception.type": qualified})
    span.set_status(Status(StatusCode.ERROR, exc_type.__name__))


def _attach_trace_context(span: Any) -> None:
    """Copy W3C trace context onto span attributes (not into SQL text).

    A per-span ``traceparent`` SQLCommenter suffix would uniquify every query
    string and defeat prepared-statement caching, so we keep SQL stable and
    expose ``trace_id`` / ``span_id`` on the active span instead.
    """
    ctx = span.get_span_context()
    if not ctx.is_valid:
        return
    span.set_attribute("trace_id", f"{ctx.trace_id:032x}")
    span.set_attribute("span_id", f"{ctx.span_id:016x}")


class OTelAdapter:
    """Decorate a ``DBAdapter`` with OpenTelemetry spans and metrics.

    When ``sqlcommenter=True``, trace context is recorded on span attributes
    only — SQL text is left unchanged so prepared-statement caches stay hot.
    (Injecting a unique ``traceparent`` comment per span would defeat caching.)

    Failed operations record only the exception *type* on the span (no message or
    stack trace), because driver errors frequently echo bound parameter values.
    """

    __slots__ = (
        "_capture_parameters",
        "_db_namespace",
        "_dialect_name",
        "_duration",
        "_inner",
        "_meter",
        "_ops",
        "_param_sanitizer",
        "_server_address",
        "_server_port",
        "_slow_query_ms",
        "_sqlcommenter",
        "_system",
        "_tracer",
        "_tx_active",
    )

    def __init__(
        self,
        inner: DBAdapter,
        *,
        db_namespace: str | None = None,
        server_address: str | None = None,
        server_port: int | None = None,
        sqlcommenter: bool = False,
        slow_query_ms: float | None = None,
        capture_parameters: bool = False,
        param_sanitizer: Callable[[Sequence[Any]], Sequence[Any]] | None = None,
        tracer: Any | None = None,
        meter: Any | None = None,
    ) -> None:
        metrics_api, trace_api, _, _, _ = _require_otel()
        self._inner = inner
        self._db_namespace = db_namespace
        self._server_address = server_address
        self._server_port = server_port
        self._sqlcommenter = sqlcommenter
        self._slow_query_ms = slow_query_ms
        self._capture_parameters = capture_parameters
        self._param_sanitizer = param_sanitizer
        self._tracer = tracer or trace_api.get_tracer("atla")
        self._meter = meter or metrics_api.get_meter("atla")
        self._duration = self._meter.create_histogram(
            "db.client.operation.duration",
            unit="s",
            description="Duration of database client operations",
        )
        self._ops = self._meter.create_counter(
            "db.client.operation.count",
            description="Number of database client operations",
        )
        self._tx_active = self._meter.create_up_down_counter(
            "db.client.transactions.active",
            description="Active database transactions",
        )
        # Tolerate adapters that do not expose ``dialect`` (omit db.system then).
        dialect_name = getattr(getattr(inner, "dialect", None), "name", None)
        self._dialect_name: str | None = dialect_name if isinstance(dialect_name, str) else None
        self._system: str | None = (
            _DIALECT_SYSTEM.get(self._dialect_name, self._dialect_name)
            if self._dialect_name is not None
            else None
        )

    def _with_inner(self, inner: DBAdapter) -> OTelAdapter:
        """Clone this wrapper around ``inner``, sharing tracer, meter, and instruments."""
        clone = object.__new__(type(self))
        for slot in OTelAdapter.__slots__:
            setattr(clone, slot, getattr(self, slot))
        clone._inner = inner
        return clone

    @property
    def dialect(self) -> Dialect:
        return self._inner.dialect

    def _base_attrs(self, operation: str, collection: str | None) -> dict[str, Any]:
        attrs: dict[str, Any] = {"db.operation.name": operation}
        if self._system is not None:
            attrs["db.system.name"] = self._system
            attrs["db.system"] = self._system
        if collection is not None:
            attrs["db.collection.name"] = collection
        if self._db_namespace is not None:
            attrs["db.namespace"] = self._db_namespace
        if self._server_address is not None:
            attrs["server.address"] = self._server_address
        if self._server_port is not None:
            attrs["server.port"] = self._server_port
        return attrs

    def _maybe_params(self, params: Sequence[Any] | None) -> Sequence[Any] | None:
        if not self._capture_parameters or params is None:
            return None
        if self._param_sanitizer is not None:
            return self._param_sanitizer(params)
        return params

    async def _run[T](
        self,
        sql: str,
        params: Sequence[Any] | None,
        coro_factory: Callable[[str], Awaitable[T]],
        *,
        row_count: Callable[[T], int | None] | None = None,
    ) -> T:
        _, _, _, _, SpanKind = _require_otel()
        operation, collection = extract_sql_meta(sql)
        span_name = f"{operation} {collection}" if collection else operation
        # Low-cardinality labels only — never put SQL text or parameter values on
        # metric attribute sets (cardinality explosion + credential leakage).
        metric_attrs = self._base_attrs(operation, collection)
        span_attrs = dict(metric_attrs)
        span_attrs["db.query.text"] = sql
        captured = self._maybe_params(params)
        if captured is not None:
            span_attrs["db.query.parameter"] = [repr(value) for value in captured]
        started = time.perf_counter()
        with self._tracer.start_as_current_span(
            span_name,
            kind=SpanKind.CLIENT,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            for key, value in span_attrs.items():
                span.set_attribute(key, value)
            # sqlcommenter: attach trace context to the span, never mutate SQL
            # (per-span traceparent comments break prepared-statement caching).
            if self._sqlcommenter:
                _attach_trace_context(span)
            try:
                result = await coro_factory(sql)
            except BaseException as exc:
                _record_error(span, exc)
                self._ops.add(1, {**metric_attrs, "error.type": type(exc).__name__})
                self._duration.record(time.perf_counter() - started, metric_attrs)
                raise
            elapsed = time.perf_counter() - started
            if row_count is not None:
                count = row_count(result)
                if count is not None:
                    span.set_attribute("db.response.returned_rows", count)
            if self._slow_query_ms is not None and elapsed * 1000.0 >= self._slow_query_ms:
                span.set_attribute("atla.slow_query", True)
            self._ops.add(1, metric_attrs)
            self._duration.record(elapsed, metric_attrs)
            return result

    async def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Sequence[Any]]:
        return await self._run(
            sql,
            params,
            lambda annotated: self._inner.fetch_all(annotated, params),
            row_count=lambda result: len(result),
        )

    async def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Any] | None:
        return await self._run(
            sql,
            params,
            lambda annotated: self._inner.fetch_one(annotated, params),
            row_count=lambda result: 0 if result is None else 1,
        )

    async def fetch_val(self, sql: str, params: Sequence[Any] = ()) -> Any:
        return await self._run(
            sql,
            params,
            lambda annotated: self._inner.fetch_val(annotated, params),
        )

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        return await self._run(
            sql,
            params,
            lambda annotated: self._inner.execute(annotated, params),
            row_count=lambda result: int(result),
        )

    async def execute_many(self, sql: str, params_seq: Sequence[Sequence[Any]]) -> None:
        # execute_many returns None; do not invent db.response.returned_rows from
        # len(params_seq) — that is submitted batch size, not rows affected.
        await self._run(
            sql,
            None,
            lambda annotated: self._inner.execute_many(annotated, params_seq),
        )

    @asynccontextmanager
    async def transaction(
        self,
        *,
        isolation: str | None = None,
        readonly: bool = False,
        deferrable: bool = False,
    ) -> AsyncGenerator[OTelAdapter, None]:
        """Begin a transaction span on the wrapped adapter.

        Options are forwarded only when set, so adapters whose ``transaction()`` does
        not yet accept them keep working; adapters that do (#53) receive them intact.
        """
        _, _, _, _, SpanKind = _require_otel()
        attrs = self._base_attrs("BEGIN", None)
        self._tx_active.add(1, attrs)
        with self._tracer.start_as_current_span(
            "BEGIN",
            kind=SpanKind.CLIENT,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            for key, value in attrs.items():
                span.set_attribute(key, value)
            try:
                async with self._inner.transaction(
                    **transaction_kwargs(isolation, readonly, deferrable)
                ) as tx:
                    yield self._with_inner(tx)
            except BaseException as exc:
                _record_error(span, exc)
                raise
            finally:
                self._tx_active.add(-1, attrs)
