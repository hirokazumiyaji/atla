"""Tests for DBAdapter implementations."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import aiosqlite
import pytest

from atla.adapter import AiosqliteAdapter, AsyncmyAdapter, AsyncpgAdapter
from atla.adapter.base import CopyAdapter
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

    async def copy_records_to_table(
        self,
        table_name: str,
        *,
        records: Sequence[Sequence[Any]],
        columns: Sequence[str] | None = None,
        schema_name: str | None = None,
    ) -> str:
        return cast(
            str,
            await self._conn.copy_records_to_table(
                table_name,
                records=records,
                columns=columns,
                schema_name=schema_name,
            ),
        )


class FakeAsyncpgConnection:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.copies: list[tuple[str | None, str, list[str], list[tuple[Any, ...]]]] = []

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

    async def copy_records_to_table(
        self,
        table_name: str,
        *,
        records: Sequence[Sequence[Any]],
        columns: Sequence[str] | None = None,
        schema_name: str | None = None,
    ) -> str:
        rows = [tuple(row) for row in records]
        self.copies.append((schema_name, table_name, list(columns or []), rows))
        return f"COPY {len(rows)}"

    @asynccontextmanager
    async def transaction(self, **kwargs: Any) -> AsyncGenerator[None]:
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
        self._autocommit = False

    def cursor(self) -> FakeAsyncmyCursor:
        return self.cursor_obj

    def get_autocommit(self) -> bool:
        return self._autocommit

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
async def test_asyncpg_copy_records_to_table_splits_schema() -> None:
    conn = FakeAsyncpgConnection()
    adapter = AsyncpgAdapter(FakeAsyncpgPool(conn), dialect=PostgresDialect())
    assert isinstance(adapter, CopyAdapter)

    copied = await adapter.copy_records_to_table(
        "public.users",
        columns=["id", "email"],
        records=[(1, "a@example.com"), (2, "b@example.com")],
    )

    assert copied == 2
    assert conn.copies == [
        ("public", "users", ["id", "email"], [(1, "a@example.com"), (2, "b@example.com")])
    ]

    await adapter.copy_records_to_table("users", columns=["id"], records=[(3,)])
    assert conn.copies[-1] == (None, "users", ["id"], [(3,)])
    assert not await adapter.copy_records_to_table("users", columns=["id"], records=[])

    with pytest.raises(ValueError, match="must not be empty"):
        await adapter.copy_records_to_table("", columns=["id"], records=[(1,)])
    with pytest.raises(ValueError, match="at least one column"):
        await adapter.copy_records_to_table("users", columns=[], records=[(1,)])
    with pytest.raises(ValueError, match=r"schema\.table"):
        await adapter.copy_records_to_table("db.public.users", columns=["id"], records=[(1,)])


@pytest.mark.asyncio
async def test_asyncpg_copy_encodes_params_and_rejects_untyped_adapters() -> None:
    conn = FakeAsyncpgConnection()
    adapter = AsyncpgAdapter(FakeAsyncpgPool(conn), dialect=PostgresDialect())
    value = uuid4()

    await adapter.copy_records_to_table("users", columns=["id"], records=[(value,)])

    # PostgresDialect.native_uuid keeps UUID objects as-is for asyncpg codecs.
    assert conn.copies[-1][3] == [(value,)]
    assert not isinstance(FakeAsyncmyConnection(), CopyAdapter)


@pytest.mark.asyncio
async def test_asyncpg_copy_reports_submitted_rows_without_status_tag() -> None:
    """A driver build that reports no status tag must not report zero rows."""

    class NoStatusConnection:
        def __init__(self) -> None:
            self.copies: list[tuple[Any, ...]] = []

        async def copy_records_to_table(
            self,
            table_name: str,
            *,
            records: Sequence[Sequence[Any]],
            columns: Sequence[str] | None = None,
            schema_name: str | None = None,
        ) -> None:
            self.copies.append(
                (schema_name, table_name, list(columns or []), [tuple(row) for row in records])
            )

    conn = NoStatusConnection()
    adapter = AsyncpgAdapter(FakeAsyncpgPool(conn), dialect=PostgresDialect())

    copied = await adapter.copy_records_to_table("users", columns=["id"], records=[(1,), (2,)])

    assert copied == 2
    assert conn.copies == [(None, "users", ["id"], [(1,), (2,)])]


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
async def test_aiosqlite_rejects_unsupported_isolation() -> None:
    conn = await aiosqlite.connect(":memory:", isolation_level=None)
    adapter = AiosqliteAdapter(conn)
    try:
        with pytest.raises(ValueError, match="does not support isolation"):
            async with adapter.transaction(isolation="serializable"):
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

        async def fetch(self, sql: str, *params: Any) -> list[tuple[Any, ...]]:
            # Pools expose fetch directly; count the implicit acquire here.
            async with self.acquire() as conn:
                return await conn.fetch(sql, *params)

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


@pytest.mark.asyncio
async def test_asyncmy_transaction_start_error_releases_connection() -> None:
    conn = FakeAsyncmyConnection()
    pool = FakeAsyncmyPool(conn)
    adapter = AsyncmyAdapter(pool, dialect=MySQLDialect())

    with pytest.raises(ValueError, match="deferrable transactions"):
        async with adapter.transaction(deferrable=True):
            pass

    assert pool.released == 1


@pytest.mark.asyncio
async def test_asyncmy_autocommit_false_commits_outside_transaction() -> None:
    conn = FakeAsyncmyConnection()
    conn._autocommit = False  # pyright: ignore[reportPrivateUsage]
    adapter = AsyncmyAdapter(conn, dialect=MySQLDialect())

    assert conn.committed == 0
    await adapter.execute("INSERT INTO t VALUES (1)")
    assert conn.committed == 1
    await adapter.execute_many("INSERT INTO t VALUES (%s)", [(2,), (3,)])
    assert conn.committed == 2


@pytest.mark.asyncio
async def test_asyncmy_savepoint_deadlock_does_not_mask_original_error() -> None:
    import asyncmy

    from atla.errors import SerializationError

    conn = FakeAsyncmyConnection()
    pool = FakeAsyncmyPool(conn)
    adapter = AsyncmyAdapter(pool, dialect=MySQLDialect())

    class DeadlockCursor(FakeAsyncmyCursor):
        def __init__(self) -> None:
            super().__init__()
            self.fail_rollback_sp = False

        async def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
            if "ROLLBACK TO SAVEPOINT" in sql and self.fail_rollback_sp:
                raise asyncmy.errors.OperationalError(1305, "SAVEPOINT does not exist")
            return await super().execute(sql, params)

    cursor = DeadlockCursor()
    conn.cursor_obj = cursor

    with pytest.raises(SerializationError):
        async with adapter.transaction() as tx:
            async with tx.transaction():
                cursor.fail_rollback_sp = True
                raise asyncmy.errors.OperationalError(
                    1213, "Deadlock found when trying to get lock"
                )


@pytest.mark.asyncio
async def test_adapter_guards_single_connection_and_nested_transactions() -> None:
    # 1. Asyncmy single connection and nested
    conn = FakeAsyncmyConnection()
    adapter = AsyncmyAdapter(conn, dialect=MySQLDialect())

    async with adapter.transaction() as my_tx:
        with pytest.raises(RuntimeError, match="active transaction"):
            await adapter.fetch_all("SELECT 1")
        async with my_tx.transaction() as nested:
            with pytest.raises(RuntimeError, match="active transaction"):
                await my_tx.fetch_all("SELECT 1")
            assert await nested.fetch_all("SELECT 1") == [(1, "alice")]

    # 2. Asyncpg single connection and nested
    pg_conn = FakeAsyncpgConnection()
    pg_adapter = AsyncpgAdapter(pg_conn, dialect=PostgresDialect())

    async with pg_adapter.transaction() as pg_tx:
        with pytest.raises(RuntimeError, match="active transaction"):
            await pg_adapter.fetch_all("SELECT 1")
        async with pg_tx.transaction() as nested_pg:
            with pytest.raises(RuntimeError, match="active transaction"):
                await pg_tx.fetch_all("SELECT 1")
            assert await nested_pg.fetch_all("SELECT 1") == [(1, "alice")]

    # 3. Aiosqlite single connection and nested
    sqlite_conn = await aiosqlite.connect(":memory:", isolation_level=None)
    sqlite_adapter = AiosqliteAdapter(sqlite_conn)
    try:
        await sqlite_adapter.execute("CREATE TABLE t (id INT)")
        async with sqlite_adapter.transaction() as sq_tx:
            with pytest.raises(RuntimeError, match="active transaction"):
                await sqlite_adapter.fetch_all("SELECT * FROM t")
            async with sq_tx.transaction() as nested_sq:
                with pytest.raises(RuntimeError, match="active transaction"):
                    await sq_tx.fetch_all("SELECT * FROM t")
                assert await nested_sq.fetch_all("SELECT * FROM t") == []
    finally:
        await sqlite_conn.close()


@pytest.mark.asyncio
async def test_asyncmy_pool_release_error_does_not_mask_transaction_error() -> None:
    conn = FakeAsyncmyConnection()

    class FailingReleasePool(FakeAsyncmyPool):
        async def release(self, conn: FakeAsyncmyConnection) -> None:
            raise RuntimeError("release failed")

    pool = FailingReleasePool(conn)
    adapter = AsyncmyAdapter(pool, dialect=MySQLDialect())

    with pytest.raises(RuntimeError, match="tx error"):
        async with adapter.transaction():
            raise RuntimeError("tx error")
