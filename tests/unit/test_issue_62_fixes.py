"""Unit tests for Issue #62 fixes.

Verifies:
1. extract_sql_meta handles line comments (--), COPY statements, and EXTRACT(... FROM ...).
2. OTelAdapter / InstrumentedAdapter record db.collection.name on bulk COPY.
3. Nested wrapper adapters (OTelAdapter, RoutingAdapter, InstrumentedAdapter) over non-copy adapters
   fail fast before starting trace spans / timer hooks.
4. RoutingAdapter validates strategy in __init__ and rejects invalid strategies with ValueError.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from typing import Any

import pytest

from atla.adapter.aiosqlite import AiosqliteAdapter
from atla.adapter.asyncmy import AsyncmyAdapter
from atla.adapter.asyncpg import AsyncpgAdapter
from atla.adapter.base import CopyAdapter, IsolationLevel, is_copy_adapter
from atla.adapter.routing import RoutingAdapter
from atla.advanced import BaseInsertManyQuery
from atla.dialect import MySQLDialect, PostgresDialect, SQLiteDialect, TiDBDialect
from atla.observability import InstrumentedAdapter, extract_sql_meta
from atla.observability.otel import OTelAdapter


class StubDBAdapter:
    """Minimal DBAdapter without COPY support."""

    def __init__(self, dialect: Any = None) -> None:
        self.dialect = dialect or SQLiteDialect()
        self.calls: list[tuple[str, str, list[Any]]] = []

    async def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Sequence[Any]]:
        self.calls.append(("fetch_all", sql, list(params)))
        return [(1,)]

    async def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Any] | None:
        self.calls.append(("fetch_one", sql, list(params)))
        return (1,)

    async def fetch_val(self, sql: str, params: Sequence[Any] = ()) -> Any:
        self.calls.append(("fetch_val", sql, list(params)))
        return 1

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        self.calls.append(("execute", sql, list(params)))
        return 1

    async def execute_many(self, sql: str, params_seq: Sequence[Sequence[Any]]) -> None:
        self.calls.append(("execute_many", sql, [list(p) for p in params_seq]))

    @asynccontextmanager
    async def transaction(
        self,
        *,
        isolation: IsolationLevel | None = None,
        readonly: bool = False,
        deferrable: bool = False,
    ) -> AsyncGenerator[StubDBAdapter, None]:
        yield self


class StubCopyAdapter(StubDBAdapter):
    """Stub adapter with COPY support."""

    def __init__(self, dialect: Any = None) -> None:
        super().__init__(dialect=dialect or PostgresDialect())
        self.copies: list[tuple[str, list[str], list[list[Any]]]] = []

    async def copy_records_to_table(
        self,
        table: str,
        *,
        columns: Sequence[str],
        records: Sequence[Sequence[Any]],
    ) -> int:
        self.copies.append((table, list(columns), [list(r) for r in records]))
        return len(records)


def test_extract_sql_meta_line_comments_and_functions() -> None:
    # 1. Line comments
    assert extract_sql_meta("-- comment\nSELECT * FROM users") == ("SELECT", "users")
    assert extract_sql_meta("-- comment 1\n-- comment 2\nSELECT * FROM users") == (
        "SELECT",
        "users",
    )
    assert extract_sql_meta("/* block */\n-- line\nSELECT * FROM users") == (
        "SELECT",
        "users",
    )
    assert extract_sql_meta("SELECT 'foo -- bar' FROM users") == ("SELECT", "users")
    assert extract_sql_meta('SELECT "a--b" FROM users') == ("SELECT", "users")
    assert extract_sql_meta("SELECT `a--b` FROM users") == ("SELECT", "users")
    assert extract_sql_meta("SELECT [a--b] FROM users") == ("SELECT", "users")

    # 2. COPY statements
    assert extract_sql_meta('COPY "users" ("id") FROM STDIN') == ("COPY", "users")
    assert extract_sql_meta("COPY users (id) FROM STDIN") == ("COPY", "users")
    assert extract_sql_meta('COPY "public"."users" ("id", "email") FROM STDIN') == (
        "COPY",
        "users",
    )
    assert extract_sql_meta("COPY public.users (id, email) FROM STDIN") == (
        "COPY",
        "users",
    )

    # 3. EXTRACT(... FROM ...) and nested parens
    assert extract_sql_meta('SELECT EXTRACT(YEAR FROM "users"."created_at") FROM "users"') == (
        "SELECT",
        "users",
    )
    assert extract_sql_meta(
        'SELECT id, EXTRACT(MONTH FROM "created_at"), EXTRACT(DAY FROM "created_at") FROM "orders"'
    ) == ("SELECT", "orders")
    assert extract_sql_meta("SELECT (SELECT 1 FROM other) FROM users") == ("SELECT", "users")
    assert extract_sql_meta("SELECT ('C:\\') FROM users") == ("SELECT", "users")
    assert extract_sql_meta("SELECT ('C:\\' || 'suffix') FROM users") == ("SELECT", "users")
    assert extract_sql_meta("SELECT ('C:\\', 'suffix') FROM users") == ("SELECT", "users")
    assert extract_sql_meta(r"SELECT ('C:\' LIKE 'C:%') FROM users") == ("SELECT", "users")
    assert extract_sql_meta(r"SELECT ('C:\' IS NOT NULL) FROM users") == ("SELECT", "users")
    assert extract_sql_meta(r"SELECT ('C:\' COLLATE BINARY) FROM users") == ("SELECT", "users")
    assert extract_sql_meta(r"SELECT CAST('C:\'AS TEXT) FROM users WHERE name = 'alice'") == (
        "SELECT",
        "users",
    )
    assert extract_sql_meta(r"SELECT ('C:\'IN ('a')) FROM users WHERE name = 'alice'") == (
        "SELECT",
        "users",
    )
    assert extract_sql_meta(r"SELECT ('C:\'IN ('--')) FROM users WHERE name = 'alice'") == (
        "SELECT",
        "users",
    )
    assert extract_sql_meta("SELECT 1--1 FROM users") == ("SELECT", "users")
    assert extract_sql_meta("SELECT 1--1 FROM users", TiDBDialect()) == ("SELECT", "users")
    assert extract_sql_meta("SELECT 1--1 FROM users", "tidb") == ("SELECT", "users")
    assert extract_sql_meta(r"SELECT 'it\' -- safe' FROM users") == ("SELECT", "users")
    assert extract_sql_meta(r"SELECT 'it\' -- safe' FROM users", TiDBDialect()) == (
        "SELECT",
        "users",
    )
    assert extract_sql_meta(r"SELECT 'it\'s -- safe' FROM users") == ("SELECT", "users")
    assert extract_sql_meta(r"SELECT 'it\'s -- text' FROM users") == ("SELECT", "users")
    assert extract_sql_meta(r"SELECT ('it\'s') FROM users") == ("SELECT", "users")
    assert extract_sql_meta(r'SELECT "it\" -- safe" FROM users') == ("SELECT", "users")
    assert extract_sql_meta(r'SELECT ("it\"(") FROM users') == ("SELECT", "users")
    assert extract_sql_meta("SELECT ARRAY['x]--', 'y'] FROM users") == ("SELECT", "users")
    assert extract_sql_meta(r"SELECT ('it\'(') FROM users", TiDBDialect()) == (
        "SELECT",
        "users",
    )
    assert extract_sql_meta(r"SELECT (E'it\'(') FROM users") == ("SELECT", "users")
    assert extract_sql_meta("SELECT $$($$ FROM users") == ("SELECT", "users")
    assert extract_sql_meta("SELECT $tag$($tag$ FROM users") == ("SELECT", "users")
    assert extract_sql_meta("SELECT $tag$price$usd($tag$ FROM users") == ("SELECT", "users")
    assert extract_sql_meta("SELECT * FROM account$$archive$$") == (
        "SELECT",
        "account$$archive$$",
    )
    assert extract_sql_meta("SELECT ('C:\\') FROM users WHERE name = 'alice'") == (
        "SELECT",
        "users",
    )
    assert extract_sql_meta("SELECT * FROM (SELECT * FROM users) AS u") == ("SELECT", None)
    assert extract_sql_meta("SELECT * FROM (SELECT * FROM users) u") == ("SELECT", None)
    assert extract_sql_meta('SELECT * FROM "as"') == ("SELECT", "as")
    assert extract_sql_meta("SELECT * FROM [as]") == ("SELECT", "as")
    assert extract_sql_meta("SELECT * FROM `as`") == ("SELECT", "as")
    assert extract_sql_meta('DELETE FROM "as"') == ("DELETE", "as")
    # ReDoS check: unterminated string with repeated backslashes
    assert extract_sql_meta("SELECT '" + "\\" * 100) == ("SELECT", None)


@pytest.mark.asyncio
async def test_copy_records_to_table_extracts_collection_name() -> None:
    # InstrumentedAdapter
    inner = StubCopyAdapter()
    seen: list[tuple[str, str | None, float, Any]] = []
    instrumented = InstrumentedAdapter(
        inner,
        on_query_end=lambda op, coll, ms, params: seen.append((op, coll, ms, params)),
    )
    copied = await instrumented.copy_records_to_table(
        "users",
        columns=["id", "name"],
        records=[(1, "alice")],
    )
    assert copied == 1
    assert len(seen) == 1
    assert seen[0][0] == "COPY"
    assert seen[0][1] == "users"

    # OTelAdapter
    pytest.importorskip("opentelemetry")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_adapter = OTelAdapter(
        inner,
        tracer=provider.get_tracer("test"),
        capture_parameters=False,
    )
    copied_otel = await otel_adapter.copy_records_to_table(
        "users",
        columns=["id", "name"],
        records=[(2, "bob")],
    )
    assert copied_otel == 1
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert attrs["db.operation.name"] == "COPY"
    assert attrs["db.collection.name"] == "users"
    assert attrs["db.response.returned_rows"] == 1


def _is_instance_of_copy(adapter: object) -> bool:
    return isinstance(adapter, CopyAdapter)


def _check_is_copy_adapter(adapter: object) -> bool:
    return bool(is_copy_adapter(adapter))


@pytest.mark.asyncio
async def test_copy_adapter_protocol_and_early_rejection() -> None:
    non_copy = StubDBAdapter()
    router = RoutingAdapter(primary=non_copy)
    assert not _is_instance_of_copy(non_copy)
    assert not _is_instance_of_copy(router)
    assert not _check_is_copy_adapter(non_copy)
    assert not _check_is_copy_adapter(router)
    assert not router.supports_copy

    copy_capable = StubCopyAdapter()
    copy_router = RoutingAdapter(primary=copy_capable)
    assert copy_router.supports_copy
    assert _is_instance_of_copy(copy_capable)
    assert _is_instance_of_copy(copy_router)
    assert _check_is_copy_adapter(copy_capable)
    assert _check_is_copy_adapter(copy_router)

    pytest.importorskip("opentelemetry")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    otel_nested = OTelAdapter(router, tracer=provider.get_tracer("test"))
    assert not _is_instance_of_copy(otel_nested)
    assert not _check_is_copy_adapter(otel_nested)
    assert not otel_nested.supports_copy

    # Calling copy_records_to_table must fail before any span is recorded
    with pytest.raises(TypeError, match="copy_records_to_table"):
        await otel_nested.copy_records_to_table("users", columns=["id"], records=[(1,)])

    # Verify no span was created
    assert len(exporter.get_finished_spans()) == 0


@pytest.mark.asyncio
async def test_instrumented_nested_adapter_early_rejection() -> None:
    non_copy = StubDBAdapter()
    router = RoutingAdapter(primary=non_copy)
    seen: list[Any] = []
    instrumented = InstrumentedAdapter(
        router,
        on_query_end=lambda op, coll, ms, params: seen.append((op, coll)),
    )
    assert not _is_instance_of_copy(instrumented)
    assert not _check_is_copy_adapter(instrumented)
    assert not instrumented.supports_copy

    with pytest.raises(TypeError, match="copy_records_to_table"):
        await instrumented.copy_records_to_table("users", columns=["id"], records=[(1,)])

    # on_query_end must not have been invoked
    assert seen == []


@pytest.mark.asyncio
async def test_base_insert_many_query_copy_checks_adapter() -> None:
    def _map_row(rows: Sequence[Any]) -> dict[str, Any]:
        return {"id": rows[0]}

    class DummyInsertMany(BaseInsertManyQuery[dict[str, Any]]):
        _RETURNING_COLS = "id"
        _ROW_MAPPER = staticmethod(_map_row)
        _COPY_TABLE = "users"

    non_copy = StubDBAdapter()
    otel_adapter = OTelAdapter(RoutingAdapter(non_copy))
    builder = DummyInsertMany(otel_adapter, table_name="users", rows=((("id", 1),),))
    with pytest.raises(TypeError, match="CopyAdapter"):
        await builder.copy()


def test_concrete_adapters_supports_copy() -> None:
    # AiosqliteAdapter
    sqlite = AiosqliteAdapter(None, dialect=SQLiteDialect(), _configure=False)
    assert not sqlite.supports_copy
    assert not _is_instance_of_copy(sqlite)
    assert not _check_is_copy_adapter(sqlite)

    # AsyncmyAdapter
    mysql = AsyncmyAdapter(None, dialect=MySQLDialect())
    assert not mysql.supports_copy
    assert not _is_instance_of_copy(mysql)
    assert not _check_is_copy_adapter(mysql)

    # AsyncpgAdapter
    class FakeTarget:
        pass

    asyncpg = AsyncpgAdapter(FakeTarget(), dialect=PostgresDialect())
    assert asyncpg.supports_copy
    assert _is_instance_of_copy(asyncpg)
    assert _check_is_copy_adapter(asyncpg)


def test_routing_adapter_strategy_validation() -> None:
    primary = StubDBAdapter()
    # Valid strategies
    r1 = RoutingAdapter(primary, strategy="round_robin")
    assert r1._strategy == "round_robin"  # pyright: ignore[reportPrivateUsage]
    r2 = RoutingAdapter(primary, strategy="random")
    assert r2._strategy == "random"  # pyright: ignore[reportPrivateUsage]

    # Invalid strategy
    with pytest.raises(ValueError, match="Unknown routing strategy 'invalid'"):
        RoutingAdapter(primary, strategy="invalid")  # type: ignore[arg-type]
