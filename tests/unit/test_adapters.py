"""Tests for DBAdapter implementations."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

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
    """Mirrors asyncpg.Pool: acquire() plus fetch* / execute proxies."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def acquire(self) -> _AcquireCM:
        return _AcquireCM(self._conn)

    async def fetch(self, sql: str, *params: Any) -> list[tuple[Any, ...]]:
        return cast(list[tuple[Any, ...]], await self._conn.fetch(sql, *params))

    async def fetchrow(self, sql: str, *params: Any) -> tuple[Any, ...] | None:
        return cast(tuple[Any, ...] | None, await self._conn.fetchrow(sql, *params))

    async def fetchval(self, sql: str, *params: Any) -> Any:
        return await self._conn.fetchval(sql, *params)

    async def execute(self, sql: str, *params: Any) -> str:
        return cast(str, await self._conn.execute(sql, *params))

    async def executemany(self, sql: str, params_seq: Sequence[Sequence[Any]]) -> None:
        await self._conn.executemany(sql, params_seq)


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
async def test_aiosqlite_rejects_reentrant_transaction_on_same_adapter() -> None:
    conn = await aiosqlite.connect(":memory:", isolation_level=None)
    adapter = AiosqliteAdapter(conn)
    try:
        async with adapter.transaction():
            with pytest.raises(RuntimeError, match="transaction already active"):
                async with adapter.transaction():
                    pass
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_aiosqlite_clears_tx_active_when_begin_fails() -> None:
    class BoomConn:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def execute(self, sql: str, params: Sequence[Any] = ()) -> Any:
            self.calls.append(sql)
            if sql == "BEGIN":
                raise RuntimeError("begin failed")

            class _Cursor:
                rowcount = 0

                async def fetchall(self) -> list[Any]:
                    return []

                async def fetchone(self) -> None:
                    return None

                async def close(self) -> None:
                    return None

            return _Cursor()

        async def executemany(self, sql: str, params_seq: Sequence[Sequence[Any]]) -> None:
            return None

        async def commit(self) -> None:
            return None

    conn = BoomConn()
    adapter = AiosqliteAdapter(conn, foreign_keys=False)
    with pytest.raises(RuntimeError, match="begin failed"):
        async with adapter.transaction():
            pass
    assert adapter._tx_active is False  # pyright: ignore[reportPrivateUsage]
    # Adapter remains usable after a failed BEGIN.
    assert await adapter.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)") == 0


@pytest.mark.asyncio
async def test_aiosqlite_autocommit_persists_without_transaction(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "autocommit.db"
    conn = await aiosqlite.connect(db_path)
    adapter = AiosqliteAdapter(conn)
    try:
        await adapter.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)")
        await adapter.execute("INSERT INTO t(name) VALUES (?)", ("alice",))
    finally:
        await conn.close()

    conn2 = await aiosqlite.connect(db_path)
    try:
        adapter2 = AiosqliteAdapter(conn2)
        assert await adapter2.fetch_val("SELECT COUNT(*) FROM t") == 1
    finally:
        await conn2.close()


@pytest.mark.asyncio
async def test_asyncpg_pool_blocks_outer_ops_during_transaction() -> None:
    conn = FakeAsyncpgConnection()
    pool = FakeAsyncpgPool(conn)
    adapter = AsyncpgAdapter(pool, dialect=PostgresDialect())
    async with adapter.transaction() as tx:
        with pytest.raises(RuntimeError, match="active transaction"):
            await adapter.fetch_all("SELECT 1")
        assert await tx.fetch_all("SELECT 1") == [(1, "alice")]


@pytest.mark.asyncio
async def test_asyncmy_pool_blocks_outer_ops_during_transaction() -> None:
    conn = FakeAsyncmyConnection()
    pool = FakeAsyncmyPool(conn)
    adapter = AsyncmyAdapter(pool, dialect=MySQLDialect())
    async with adapter.transaction() as tx:
        with pytest.raises(RuntimeError, match="active transaction"):
            await adapter.fetch_all("SELECT 1")
        assert await tx.fetch_all("SELECT 1") == [(1, "alice")]


@pytest.mark.asyncio
async def test_asyncpg_pool_allows_concurrent_tasks() -> None:
    """Sharing one pool adapter across tasks must not raise RuntimeError."""

    class CountingPool:
        def __init__(self) -> None:
            self.in_use = 0
            self.max_in_use = 0
            self.lock = asyncio.Lock()

        def acquire(self) -> CountingAcquireCM:
            return CountingAcquireCM(self)

    class CountingAcquireCM:
        def __init__(self, pool: CountingPool) -> None:
            self.pool = pool

        async def __aenter__(self) -> FakeAsyncpgConnection:
            async with self.pool.lock:
                self.pool.in_use += 1
                self.pool.max_in_use = max(self.pool.max_in_use, self.pool.in_use)
            return FakeAsyncpgConnection()

        async def __aexit__(self, *args: object) -> None:
            async with self.pool.lock:
                self.pool.in_use -= 1

    pool = CountingPool()
    adapter = AsyncpgAdapter(pool, dialect=PostgresDialect())

    async def run_query() -> Sequence[Sequence[Any]]:
        return await adapter.fetch_all("SELECT 1")

    async def run_tx() -> None:
        async with adapter.transaction() as tx:
            await tx.fetch_all("SELECT 1")
            await asyncio.sleep(0.01)

    results = await asyncio.gather(
        *[run_query() for _ in range(20)],
        *[run_tx() for _ in range(20)],
    )
    assert all(r == [(1, "alice")] for r in results[:20])
    assert pool.max_in_use > 1


@pytest.mark.asyncio
async def test_asyncmy_pool_allows_concurrent_tasks() -> None:
    """Sharing one asyncmy pool adapter across tasks must not raise RuntimeError."""

    class CountingAsyncmyPool:
        def __init__(self) -> None:
            self._in_use = 0
            self.max_in_use = 0
            self._lock = asyncio.Lock()

        async def acquire(self) -> FakeAsyncmyConnection:
            async with self._lock:
                self._in_use += 1
                self.max_in_use = max(self.max_in_use, self._in_use)
            return FakeAsyncmyConnection()

        async def release(self, conn: FakeAsyncmyConnection) -> None:
            async with self._lock:
                self._in_use -= 1

    pool = CountingAsyncmyPool()
    adapter = AsyncmyAdapter(pool, dialect=MySQLDialect())

    async def run_query() -> Sequence[Sequence[Any]]:
        return await adapter.fetch_all("SELECT 1")

    async def run_tx() -> None:
        async with adapter.transaction() as tx:
            await tx.fetch_all("SELECT 1")
            await asyncio.sleep(0.01)

    results = await asyncio.gather(
        *[run_query() for _ in range(20)],
        *[run_tx() for _ in range(20)],
    )
    assert all(r == [(1, "alice")] for r in results[:20])
    assert pool.max_in_use > 1


@pytest.mark.asyncio
async def test_asyncpg_pool_peer_task_not_blocked_by_transaction() -> None:
    """A peer task must keep using the pool while another task holds a transaction."""
    conn = FakeAsyncpgConnection()
    pool = FakeAsyncpgPool(conn)
    adapter = AsyncpgAdapter(pool, dialect=PostgresDialect())
    peer_done = asyncio.Event()
    gate = asyncio.Event()

    async def holder() -> None:
        async with adapter.transaction() as tx:
            gate.set()
            await peer_done.wait()
            assert await tx.fetch_all("SELECT 1") == [(1, "alice")]

    async def peer() -> None:
        await gate.wait()
        assert await adapter.fetch_all("SELECT 1") == [(1, "alice")]
        peer_done.set()

    await asyncio.gather(holder(), peer())


@pytest.mark.asyncio
async def test_asyncpg_pool_child_task_can_use_outer_adapter() -> None:
    """Child tasks must not inherit the parent's checkout restriction."""
    conn = FakeAsyncpgConnection()
    pool = FakeAsyncpgPool(conn)
    adapter = AsyncpgAdapter(pool, dialect=PostgresDialect())

    async with adapter.transaction() as tx:
        with pytest.raises(RuntimeError, match="active transaction"):
            await adapter.fetch_all("SELECT 1")

        child_result: list[Sequence[Sequence[Any]]] = []

        async def child() -> None:
            child_result.append(await adapter.fetch_all("SELECT 1"))

        await asyncio.create_task(child())
        assert child_result == [[(1, "alice")]]
        assert await tx.fetch_all("SELECT 1") == [(1, "alice")]


@pytest.mark.asyncio
async def test_aiosqlite_serializes_concurrent_writes(tmp_path: Path) -> None:
    db_path = tmp_path / "concurrent.db"
    conn = await aiosqlite.connect(db_path)
    adapter = AiosqliteAdapter(conn)
    try:
        await adapter.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)")

        async def insert_one(i: int) -> None:
            await adapter.execute("INSERT INTO t(name) VALUES (?)", (f"n{i}",))

        await asyncio.gather(*[insert_one(i) for i in range(30)])
        assert await adapter.fetch_val("SELECT COUNT(*) FROM t") == 30
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_aiosqlite_blocks_outer_ops_during_transaction() -> None:
    conn = await aiosqlite.connect(":memory:", isolation_level=None)
    adapter = AiosqliteAdapter(conn)
    try:
        await adapter.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)")
        async with adapter.transaction() as tx:
            await tx.execute("INSERT INTO t(name) VALUES (?)", ("alice",))
            with pytest.raises(RuntimeError, match="active transaction"):
                await adapter.fetch_all("SELECT 1")
            with pytest.raises(RuntimeError, match="active transaction"):
                await adapter.execute("INSERT INTO t(name) VALUES (?)", ("bob",))
        assert await adapter.fetch_val("SELECT COUNT(*) FROM t") == 1
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_aiosqlite_peer_task_rejected_during_transaction() -> None:
    conn = await aiosqlite.connect(":memory:", isolation_level=None)
    adapter = AiosqliteAdapter(conn)
    try:
        await adapter.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)")
        gate = asyncio.Event()
        peer_done = asyncio.Event()

        async def holder() -> None:
            async with adapter.transaction() as tx:
                await tx.execute("INSERT INTO t(name) VALUES (?)", ("alice",))
                gate.set()
                await peer_done.wait()

        async def peer() -> None:
            await gate.wait()
            with pytest.raises(RuntimeError, match="active transaction"):
                await adapter.execute("INSERT INTO t(name) VALUES (?)", ("bob",))
            peer_done.set()

        await asyncio.gather(holder(), peer())
        assert await adapter.fetch_val("SELECT COUNT(*) FROM t") == 1
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_asyncpg_pool_checkout_is_per_adapter_instance() -> None:
    """Checkout ownership lives on the adapter, not in a shared registry."""
    conn = FakeAsyncpgConnection()
    pool = FakeAsyncpgPool(conn)
    first = AsyncpgAdapter(pool, dialect=PostgresDialect())
    second = AsyncpgAdapter(pool, dialect=PostgresDialect())

    async with first.transaction():
        with pytest.raises(RuntimeError, match="active transaction"):
            await first.fetch_all("SELECT 1")
        assert await second.fetch_all("SELECT 1") == [(1, "alice")]


@pytest.mark.asyncio
async def test_asyncpg_pool_blocks_nested_outer_transaction() -> None:
    """Outer ``transaction()`` must not re-enter while a checkout is held."""
    conn = FakeAsyncpgConnection()
    pool = FakeAsyncpgPool(conn)
    adapter = AsyncpgAdapter(pool, dialect=PostgresDialect())

    async with adapter.transaction() as tx:
        with pytest.raises(RuntimeError, match="active transaction"):
            async with adapter.transaction():
                pass
        assert await tx.fetch_all("SELECT 1") == [(1, "alice")]
