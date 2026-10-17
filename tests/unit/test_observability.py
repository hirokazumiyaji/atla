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
