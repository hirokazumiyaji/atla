"""Tests for atla.observability helpers and adapters."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from typing import Any

import pytest

from atla.dialect import PostgresDialect, SQLiteDialect
from atla.observability import InstrumentedAdapter, extract_sql_meta
from atla.observability.otel import OTelAdapter


class FakeAdapter:
    def __init__(self) -> None:
        self.dialect = PostgresDialect()
        self.calls: list[tuple[str, str, list[Any]]] = []
        self.rows: list[Sequence[Any]] = [(1,)]
        self.fail = False

    async def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Sequence[Any]]:
        self.calls.append(("fetch_all", sql, list(params)))
        if self.fail:
            msg = "boom"
            raise RuntimeError(msg)
        return list(self.rows)

    async def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Any] | None:
        self.calls.append(("fetch_one", sql, list(params)))
        return self.rows[0] if self.rows else None

    async def fetch_val(self, sql: str, params: Sequence[Any] = ()) -> Any:
        self.calls.append(("fetch_val", sql, list(params)))
        return 1

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        self.calls.append(("execute", sql, list(params)))
        return 1

    async def execute_many(self, sql: str, params_seq: Sequence[Sequence[Any]]) -> None:
        self.calls.append(("execute_many", sql, [list(p) for p in params_seq]))

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[FakeAdapter, None]:
        yield self


def test_extract_sql_meta_operations() -> None:
    assert extract_sql_meta('SELECT "users"."id" FROM "users"') == ("SELECT", "users")
    assert extract_sql_meta("INSERT INTO posts (title) VALUES ($1)") == ("INSERT", "posts")
    assert extract_sql_meta("UPDATE users SET name = $1") == ("UPDATE", "users")
    assert extract_sql_meta("DELETE FROM `orders` WHERE id = %s") == ("DELETE", "orders")
    assert extract_sql_meta("BEGIN") == ("BEGIN", None)
    assert extract_sql_meta("WITH cte AS (SELECT id FROM posts) SELECT id FROM users") == (
        "SELECT",
        None,
    )


@pytest.mark.asyncio
async def test_instrumented_adapter_slow_query_hook() -> None:
    inner = FakeAdapter()
    seen: list[tuple[str, str | None, float]] = []

    adapter = InstrumentedAdapter(
        inner,
        slow_query_ms=0.0,
        on_slow_query=lambda op, coll, ms, _params: seen.append((op, coll, ms)),
    )
    rows = await adapter.fetch_all('SELECT id FROM "users"', [])
    assert rows == [(1,)]
    assert seen and seen[0][0] == "SELECT" and seen[0][1] == "users"


@pytest.mark.asyncio
async def test_otel_adapter_records_span_without_params() -> None:
    pytest.importorskip("opentelemetry")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    inner = FakeAdapter()
    adapter = OTelAdapter(
        inner,
        db_namespace="app",
        sqlcommenter=True,
        tracer=tracer,
        capture_parameters=False,
    )
    await adapter.fetch_all('SELECT id FROM "users" WHERE email = $1', ["secret@example.com"])
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "SELECT users"
    attrs = dict(span.attributes or {})
    assert attrs["db.system.name"] == "postgresql"
    assert attrs["db.operation.name"] == "SELECT"
    assert attrs["db.collection.name"] == "users"
    assert "db.query.parameter" not in attrs
    assert "secret@example.com" not in str(attrs.get("db.query.text", ""))
    # SQLCommenter appends traceparent after the original SQL.
    assert "traceparent=" in inner.calls[0][1]


@pytest.mark.asyncio
async def test_otel_metrics_omit_sql_and_params_from_labels() -> None:
    """Metric attribute sets must stay low-cardinality (no SQL text / params)."""
    pytest.importorskip("opentelemetry")
    from opentelemetry.sdk.trace import TracerProvider

    tracer = TracerProvider().get_tracer("test")
    recorded: list[dict[str, Any]] = []

    class _RecordingHistogram:
        def record(self, _amount: float, attributes: dict[str, Any] | None = None) -> None:
            recorded.append(dict(attributes or {}))

    class _RecordingCounter:
        def add(self, _amount: int, attributes: dict[str, Any] | None = None) -> None:
            recorded.append(dict(attributes or {}))

    inner = FakeAdapter()
    adapter = OTelAdapter(inner, tracer=tracer, capture_parameters=True)
    adapter._duration = _RecordingHistogram()  # pyright: ignore[reportAttributeAccessIssue]
    adapter._ops = _RecordingCounter()  # pyright: ignore[reportAttributeAccessIssue]

    await adapter.fetch_all(
        'SELECT id FROM "users" WHERE email = $1',
        ["secret@example.com"],
    )
    assert recorded
    for attrs in recorded:
        assert "db.query.text" not in attrs
        assert "db.query.parameter" not in attrs
        assert "secret@example.com" not in str(attrs)
        assert attrs.get("db.operation.name") == "SELECT"


@pytest.mark.asyncio
async def test_otel_adapter_records_errors() -> None:
    pytest.importorskip("opentelemetry")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.trace import StatusCode

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    inner = FakeAdapter()
    inner.fail = True
    adapter = OTelAdapter(inner, tracer=tracer)
    with pytest.raises(RuntimeError):
        await adapter.fetch_all("SELECT 1")
    spans = exporter.get_finished_spans()
    assert spans[0].status.status_code == StatusCode.ERROR


def test_extract_sql_meta_sqlite_quotes() -> None:
    assert extract_sql_meta('SELECT * FROM "public"."users"') == ("SELECT", "users")
    _ = SQLiteDialect()  # ensure dialect import stays available for adapters


class _NoDialectAdapter:
    """Driver-like adapter that does not expose ``dialect``."""

    async def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Sequence[Any]]:
        return [(1,)]


class _KwargsTxAdapter(FakeAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.tx_kwargs: list[dict[str, Any]] = []

    @asynccontextmanager
    async def transaction(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        *,
        isolation: str | None = None,
        readonly: bool = False,
        deferrable: bool = False,
    ) -> AsyncGenerator[FakeAdapter, None]:
        self.tx_kwargs.append(
            {"isolation": isolation, "readonly": readonly, "deferrable": deferrable}
        )
        yield self


def _tracer_with_exporter() -> tuple[Any, Any]:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test"), exporter


@pytest.mark.asyncio
async def test_otel_error_span_does_not_leak_exception_message() -> None:
    pytest.importorskip("opentelemetry.sdk")
    tracer, exporter = _tracer_with_exporter()

    class LeakyAdapter(FakeAdapter):
        async def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Sequence[Any]]:
            msg = "Duplicate entry 'dup@example.com' for key 'email'"
            raise RuntimeError(msg)

    adapter = OTelAdapter(LeakyAdapter(), tracer=tracer)
    with pytest.raises(RuntimeError):
        await adapter.fetch_all("SELECT 1", ["dup@example.com"])
    span = exporter.get_finished_spans()[0]
    exception_events = [e for e in span.events if e.name == "exception"]
    assert len(exception_events) == 1
    assert exception_events[0].attributes is not None
    assert exception_events[0].attributes["exception.type"] == "RuntimeError"
    assert "dup@example.com" not in str(exception_events[0].attributes)
    assert span.status.description == "RuntimeError"
    assert "dup@example.com" not in repr(span.status)


@pytest.mark.asyncio
async def test_otel_wraps_adapter_without_dialect() -> None:
    pytest.importorskip("opentelemetry.sdk")
    tracer, exporter = _tracer_with_exporter()
    adapter = OTelAdapter(_NoDialectAdapter(), tracer=tracer)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    assert await adapter.fetch_all("SELECT 1") == [(1,)]
    attrs = dict(exporter.get_finished_spans()[0].attributes or {})
    assert "db.system.name" not in attrs


@pytest.mark.asyncio
async def test_otel_transaction_shares_instruments_and_forwards_options() -> None:
    pytest.importorskip("opentelemetry.sdk")
    tracer, _ = _tracer_with_exporter()
    inner = _KwargsTxAdapter()
    adapter = OTelAdapter(inner, tracer=tracer)
    async with adapter.transaction(isolation="serializable", readonly=True) as tx:
        assert tx._duration is adapter._duration  # pyright: ignore[reportPrivateUsage]
        assert tx._ops is adapter._ops  # pyright: ignore[reportPrivateUsage]
        assert tx._tx_active is adapter._tx_active  # pyright: ignore[reportPrivateUsage]
    assert inner.tx_kwargs == [{"isolation": "serializable", "readonly": True, "deferrable": False}]


@pytest.mark.asyncio
async def test_instrumented_transaction_default_call_omits_unset_options() -> None:
    inner = FakeAdapter()  # transaction() accepts no options
    adapter = InstrumentedAdapter(inner)
    async with adapter.transaction() as tx:
        assert await tx.fetch_val("SELECT 1") == 1
    kwarg_inner = _KwargsTxAdapter()
    async with InstrumentedAdapter(kwarg_inner).transaction(deferrable=True):
        pass
    assert kwarg_inner.tx_kwargs == [{"isolation": None, "readonly": False, "deferrable": True}]


def test_asyncpg_adapter_exposes_dialect() -> None:
    from atla.adapter.asyncpg import AsyncpgAdapter

    class FakePool:
        def acquire(self) -> None:  # marker for pool mode
            raise NotImplementedError

    adapter = AsyncpgAdapter(FakePool())
    assert isinstance(adapter.dialect, PostgresDialect)
    OTelAdapter(adapter)  # must not raise AttributeError
    InstrumentedAdapter(adapter)
