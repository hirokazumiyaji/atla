"""OpenTelemetry adapter wrapper (optional dependency: ``atla[otel]``)."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from typing import Any

from atla.adapter.base import DBAdapter
from atla.dialect import Dialect
from atla.observability.sql_meta import extract_sql_meta

_DIALECT_SYSTEM: dict[str, str] = {
    "postgres": "postgresql",
    "cockroach": "cockroachdb",
    "mysql": "mysql",
    "tidb": "tidb",
    "sqlite": "sqlite",
}


def _require_otel() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from opentelemetry import metrics, trace
        from opentelemetry.trace import SpanKind, Status, StatusCode
    except ImportError as exc:  # pragma: no cover - exercised when otel missing
        msg = "OpenTelemetry is required for OTelAdapter; install with `pip install atla[otel]`"
        raise ImportError(msg) from exc
    return metrics, trace, Status, StatusCode, SpanKind


def _sqlcomment_for_span(sql: str, span: Any) -> str:
    ctx = span.get_span_context()
    if not ctx.is_valid:
        return sql
    flags = f"{ctx.trace_flags:02x}"
    traceparent = f"00-{ctx.trace_id:032x}-{ctx.span_id:016x}-{flags}"
    return f"{sql} /*traceparent='{traceparent}'*/"


class OTelAdapter:
    """Decorate a ``DBAdapter`` with OpenTelemetry spans, metrics, and SQLCommenter."""

    __slots__ = (
        "_capture_parameters",
        "_db_namespace",
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
        self._system = _DIALECT_SYSTEM.get(inner.dialect.name, inner.dialect.name)

    @property
    def dialect(self) -> Dialect:
        return self._inner.dialect

    def _base_attrs(self, operation: str, collection: str | None) -> dict[str, Any]:
        attrs: dict[str, Any] = {
            "db.system.name": self._system,
            "db.system": self._system,
            "db.operation.name": operation,
        }
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

    async def _run(
        self,
        sql: str,
        params: Sequence[Any] | None,
        coro_factory: Callable[[str], Awaitable[Any]],
        *,
        row_count: Callable[[Any], int | None] | None = None,
    ) -> Any:
        _, _, Status, StatusCode, SpanKind = _require_otel()
        operation, collection = extract_sql_meta(sql)
        span_name = f"{operation} {collection}" if collection else operation
        attrs = self._base_attrs(operation, collection)
        attrs["db.query.text"] = sql
        captured = self._maybe_params(params)
        if captured is not None:
            attrs["db.query.parameter"] = [repr(value) for value in captured]
        started = time.perf_counter()
        with self._tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT) as span:
            for key, value in attrs.items():
                span.set_attribute(key, value)
            annotated = _sqlcomment_for_span(sql, span) if self._sqlcommenter else sql
            try:
                result = await coro_factory(annotated)
            except BaseException as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                self._ops.add(1, {**attrs, "error.type": type(exc).__name__})
                self._duration.record(time.perf_counter() - started, attrs)
                raise
            elapsed = time.perf_counter() - started
            if row_count is not None:
                count = row_count(result)
                if count is not None:
                    span.set_attribute("db.response.returned_rows", count)
            if self._slow_query_ms is not None and elapsed * 1000.0 >= self._slow_query_ms:
                span.set_attribute("atla.slow_query", True)
            self._ops.add(1, attrs)
            self._duration.record(elapsed, attrs)
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
        return await self._run(
            sql,
            None,
            lambda annotated: self._inner.execute_many(annotated, params_seq),
            row_count=lambda _result: len(params_seq),
        )

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[OTelAdapter, None]:
        _, _, Status, StatusCode, SpanKind = _require_otel()
        attrs = self._base_attrs("BEGIN", None)
        self._tx_active.add(1, attrs)
        with self._tracer.start_as_current_span("BEGIN", kind=SpanKind.CLIENT) as span:
            for key, value in attrs.items():
                span.set_attribute(key, value)
            try:
                async with self._inner.transaction() as tx:
                    wrapped = OTelAdapter(
                        tx,
                        db_namespace=self._db_namespace,
                        server_address=self._server_address,
                        server_port=self._server_port,
                        sqlcommenter=self._sqlcommenter,
                        slow_query_ms=self._slow_query_ms,
                        capture_parameters=self._capture_parameters,
                        param_sanitizer=self._param_sanitizer,
                        tracer=self._tracer,
                        meter=self._meter,
                    )
                    yield wrapped
            except BaseException as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise
            finally:
                self._tx_active.add(-1, attrs)
