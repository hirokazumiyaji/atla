"""Tests for DBAdapter implementations."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from typing import Any

import aiosqlite
import pytest

from atla.adapter import AiosqliteAdapter, AsyncmyAdapter, AsyncpgAdapter
from atla.dialect import MySQLDialect, PostgresDialect, SQLiteDialect


class _AcquireCM:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def __aenter__(self) -> Any:
        return self._conn

    async def __aexit__(self, *args: object) -> None:
        return None


class FakeAsyncpgPool:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def acquire(self) -> _AcquireCM:
        return _AcquireCM(self._conn)


class FakeAsyncpgConnection:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def fetch(self, sql: str, *params: Any) -> list[tuple[Any, ...]]:
        self.calls.append(f"fetch:{sql}")
        return [(1, "alice")]

    async def fetchrow(self, sql: str, *params: Any) -> tuple[Any, ...] | None:
        self.calls.append(f"fetchrow:{sql}")
        return (1, "alice")

    async def fetchval(self, sql: str, *params: Any) -> Any:
        self.calls.append(f"fetchval:{sql}")
        return 1

    async def execute(self, sql: str, *params: Any) -> str:
        self.calls.append(f"execute:{sql}")
        return "UPDATE 2"

    async def executemany(self, sql: str, params_seq: Sequence[Sequence[Any]]) -> None:
        self.calls.append(f"executemany:{sql}")

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[None]:
        self.calls.append("transaction")
        yield None


class FakeAsyncmyCursor:
    def __init__(self) -> None:
        self.statements: list[str] = []

    async def __aenter__(self) -> FakeAsyncmyCursor:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        self.statements.append(sql)
        return 1

    async def fetchall(self) -> list[tuple[Any, ...]]:
        return [(1, "alice")]

    async def fetchone(self) -> tuple[Any, ...] | None:
        return (1, "alice")

    async def executemany(self, sql: str, params_seq: Sequence[Sequence[Any]]) -> None:
        self.statements.append(sql)


class FakeAsyncmyConnection:
    def __init__(self) -> None:
        self.cursor_obj = FakeAsyncmyCursor()
        self.began = 0
        self.committed = 0
        self.rolled_back = 0

    def cursor(self) -> FakeAsyncmyCursor:
        return self.cursor_obj

    async def begin(self) -> None:
        self.began += 1

    async def commit(self) -> None:
        self.committed += 1

    async def rollback(self) -> None:
        self.rolled_back += 1


class FakeAsyncmyPool:
    def __init__(self, conn: FakeAsyncmyConnection) -> None:
        self._conn = conn
        self.released = 0

    async def acquire(self) -> FakeAsyncmyConnection:
        return self._conn

    async def release(self, conn: FakeAsyncmyConnection) -> None:
        self.released += 1


@pytest.mark.asyncio
async def test_aiosqlite_fetch_execute_and_transactions() -> None:
    conn = await aiosqlite.connect(":memory:", isolation_level=None)
    adapter = AiosqliteAdapter(conn)
    try:
        assert isinstance(adapter.dialect, SQLiteDialect)

        await adapter.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
        assert await adapter.execute("INSERT INTO users(name) VALUES (?)", ("alice",)) == 1
        await adapter.execute_many(
            "INSERT INTO users(name) VALUES (?)",
            [("bob",), ("carol",)],
        )

        rows = await adapter.fetch_all("SELECT name FROM users ORDER BY id")
        assert [row[0] for row in rows] == ["alice", "bob", "carol"]
        assert (await adapter.fetch_one("SELECT name FROM users WHERE id = ?", (2,))) == ("bob",)
        assert await adapter.fetch_val("SELECT COUNT(*) FROM users") == 3

        with pytest.raises(RuntimeError, match="rollback outer"):
            async with adapter.transaction() as tx:
                await tx.execute("INSERT INTO users(name) VALUES (?)", ("dave",))
                async with tx.transaction() as nested:
                    await nested.execute("INSERT INTO users(name) VALUES (?)", ("erin",))
                raise RuntimeError("rollback outer")
        assert await adapter.fetch_val("SELECT COUNT(*) FROM users") == 3

        async with adapter.transaction() as tx:
            await tx.execute("INSERT INTO users(name) VALUES (?)", ("dave",))
            with pytest.raises(RuntimeError, match="rollback nested"):
                async with tx.transaction() as nested:
                    await nested.execute("INSERT INTO users(name) VALUES (?)", ("erin",))
                    raise RuntimeError("rollback nested")
            await tx.execute("INSERT INTO users(name) VALUES (?)", ("frank",))

        names = [row[0] for row in await adapter.fetch_all("SELECT name FROM users ORDER BY id")]
        assert names == ["alice", "bob", "carol", "dave", "frank"]
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_asyncpg_adapter_with_mocks() -> None:
    conn = FakeAsyncpgConnection()
    pool = FakeAsyncpgPool(conn)
    adapter = AsyncpgAdapter(pool, dialect=PostgresDialect())

    assert await adapter.fetch_all("SELECT 1") == [(1, "alice")]
    assert await adapter.fetch_one("SELECT 1") == (1, "alice")
    assert await adapter.fetch_val("SELECT 1") == 1
    assert await adapter.execute("UPDATE t SET x=1") == 2
    await adapter.execute_many("INSERT INTO t VALUES ($1)", [(1,), (2,)])

    async with adapter.transaction() as tx:
        assert isinstance(tx, AsyncpgAdapter)
        async with tx.transaction() as nested:
            assert isinstance(nested, AsyncpgAdapter)
    assert conn.calls.count("transaction") == 2


@pytest.mark.asyncio
async def test_asyncmy_adapter_with_mocks() -> None:
    conn = FakeAsyncmyConnection()
    pool = FakeAsyncmyPool(conn)
    adapter = AsyncmyAdapter(pool, dialect=MySQLDialect())

    assert await adapter.fetch_all("SELECT 1") == [(1, "alice")]
    assert await adapter.fetch_one("SELECT 1") == (1, "alice")
    assert await adapter.fetch_val("SELECT 1") == 1
    assert await adapter.execute("UPDATE t SET x=%s", (1,)) == 1
    await adapter.execute_many("INSERT INTO t VALUES (%s)", [(1,), (2,)])

    async with adapter.transaction() as tx:
        assert isinstance(tx, AsyncmyAdapter)
        async with tx.transaction() as nested:
            assert isinstance(nested, AsyncmyAdapter)

    with pytest.raises(RuntimeError, match="boom"):
        async with adapter.transaction() as tx:
            async with tx.transaction():
                raise RuntimeError("boom")

    assert conn.rolled_back >= 1
    assert any("SAVEPOINT" in sql for sql in conn.cursor_obj.statements)
    assert any("RELEASE SAVEPOINT" in sql for sql in conn.cursor_obj.statements)


@pytest.mark.asyncio
async def test_asyncmy_pool_blocks_outer_ops_during_transaction() -> None:
    conn = FakeAsyncmyConnection()
    pool = FakeAsyncmyPool(conn)
    adapter = AsyncmyAdapter(pool, dialect=MySQLDialect())
    async with adapter.transaction() as tx:
        with pytest.raises(RuntimeError, match="active transaction"):
            await adapter.fetch_all("SELECT 1")
        assert await tx.fetch_all("SELECT 1") == [(1, "alice")]
