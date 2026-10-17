"""Portable DB errors, serialization retry, and replica routing."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from typing import Any

import pytest

from atla.adapter.base import DBAdapter, IsolationLevel
from atla.adapter.routing import RoutingAdapter
from atla.adapter.translate import translate_exception
from atla.dialect import PostgresDialect
from atla.errors import (
    ForeignKeyViolationError,
    SerializationError,
    UniqueViolationError,
)
from atla.transaction import begin_sql, map_asyncpg_isolation, run_in_transaction


class _FakePgError(Exception):
    def __init__(self, message: str, *, sqlstate: str) -> None:
        super().__init__(message)
        self.sqlstate = sqlstate


class _FakeMysqlError(Exception):
    def __init__(self, message: str, *, errno: int) -> None:
        super().__init__(message)
        self.args = (errno, message)
        self.errno = errno


def test_translate_unique_postgres() -> None:
    mapped = translate_exception(_FakePgError("dup", sqlstate="23505"))
    assert isinstance(mapped, UniqueViolationError)


def test_translate_fk_mysql() -> None:
    mapped = translate_exception(_FakeMysqlError("fk", errno=1452))
    assert isinstance(mapped, ForeignKeyViolationError)


def test_translate_sqlite_unique() -> None:
    mapped = translate_exception(Exception("UNIQUE constraint failed: users.email"))
    assert isinstance(mapped, UniqueViolationError)


def test_translate_serialization() -> None:
    mapped = translate_exception(_FakePgError("conflict", sqlstate="40001"))
    assert isinstance(mapped, SerializationError)


class RecordingAdapter:
    def __init__(self, name: str) -> None:
        self.name = name
        self.dialect = PostgresDialect()
        self.calls: list[tuple[str, str]] = []
        self._fail_times = 0

    async def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Sequence[Any]]:
        self.calls.append(("fetch_all", sql))
        return [(1,)]

    async def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Any] | None:
        self.calls.append(("fetch_one", sql))
        return (1,)

    async def fetch_val(self, sql: str, params: Sequence[Any] = ()) -> Any:
        self.calls.append(("fetch_val", sql))
        return 1

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        self.calls.append(("execute", sql))
        return 1

    async def execute_many(self, sql: str, params_seq: Sequence[Sequence[Any]]) -> None:
        self.calls.append(("execute_many", sql))

    @asynccontextmanager
    async def transaction(
        self,
        *,
        isolation: IsolationLevel | None = None,
        readonly: bool = False,
        deferrable: bool = False,
    ) -> AsyncGenerator[DBAdapter]:
        _ = isolation, readonly, deferrable
        self.calls.append(("transaction", ""))
        yield self


@pytest.mark.asyncio
async def test_routing_adapter_splits_read_write() -> None:
    primary = RecordingAdapter("primary")
    replica = RecordingAdapter("replica")
    router = RoutingAdapter(primary, [replica])
    await router.fetch_all("SELECT 1")
    await router.execute("INSERT INTO t VALUES (1)")
    await router.fetch_all("SELECT 1 FOR UPDATE")
    assert replica.calls[0][0] == "fetch_all"
    assert primary.calls[0][0] == "execute"
    assert primary.calls[1][0] == "fetch_all"
    assert "FOR UPDATE" in primary.calls[1][1]


@pytest.mark.asyncio
async def test_routing_adapter_routes_all_locking_reads_to_primary() -> None:
    primary = RecordingAdapter("primary")
    replica = RecordingAdapter("replica")
    router = RoutingAdapter(primary, [replica])
    locking = (
        "SELECT 1 FOR NO KEY UPDATE",
        "SELECT 1 FOR KEY SHARE",
        "SELECT 1 FOR SHARE",
        "SELECT 1 LOCK IN SHARE MODE",
        "select 1 for no key update",
        "select 1 lock in share mode",
    )
    for sql in locking:
        await router.fetch_all(sql)
    assert [c[0] for c in primary.calls] == ["fetch_all"] * len(locking)
    assert replica.calls == []


@pytest.mark.asyncio
async def test_routing_adapter_sends_returning_writes_to_primary() -> None:
    primary = RecordingAdapter("primary")
    replica = RecordingAdapter("replica")
    router = RoutingAdapter(primary, [replica])
    sql = 'INSERT INTO "users" ("name") VALUES ($1) RETURNING "id", "name"'
    await router.fetch_one(sql, ("x",))
    assert primary.calls == [("fetch_one", sql)]
    assert replica.calls == []


@pytest.mark.asyncio
async def test_routing_adapter_strips_leading_comments_for_writes() -> None:
    primary = RecordingAdapter("primary")
    replica = RecordingAdapter("replica")
    router = RoutingAdapter(primary, [replica])
    block = "/* audit */\nINSERT INTO t VALUES (1) RETURNING id"
    line = "-- write\nUPDATE t SET x = 1 RETURNING x"
    await router.fetch_one(block)
    await router.fetch_all(line)
    assert primary.calls == [("fetch_one", block), ("fetch_all", line)]
    assert replica.calls == []


def test_translate_ignores_non_driver_exceptions() -> None:
    class AppSerializationFailure(Exception):
        pass

    assert translate_exception(AppSerializationFailure("nope")) is None
    assert translate_exception(Exception("UNIQUE constraint failed: t.c")) is not None


@pytest.mark.asyncio
async def test_run_in_transaction_retries_serialization() -> None:
    class Flaky(RecordingAdapter):
        def __init__(self) -> None:
            super().__init__("flaky")
            self.attempts = 0

        @asynccontextmanager
        async def transaction(
            self,
            *,
            isolation: IsolationLevel | None = None,
            readonly: bool = False,
            deferrable: bool = False,
        ) -> AsyncGenerator[DBAdapter]:
            _ = isolation, readonly, deferrable
            self.attempts += 1
            if self.attempts < 3:
                raise SerializationError("retry me")
            yield self

    adapter = Flaky()

    async def work(tx: DBAdapter) -> str:
        await tx.execute("UPDATE t SET x = 1")
        return "ok"

    result = await run_in_transaction(adapter, work, max_retries=5, base_backoff=0.0)
    assert result == "ok"
    assert adapter.attempts == 3


def test_begin_sql_rejects_unknown_isolation() -> None:
    with pytest.raises(ValueError, match="unknown isolation level"):
        begin_sql("postgres", isolation="snapshot", readonly=False, deferrable=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown isolation level"):
        map_asyncpg_isolation("not_a_level")  # type: ignore[arg-type]
    assert begin_sql(
        "postgres",
        isolation="serializable",
        readonly=False,
        deferrable=False,
    ) == ["SET TRANSACTION ISOLATION LEVEL SERIALIZABLE"]
