"""asyncmy adapter for MySQL and TiDB."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from typing import Any, cast

from atla.adapter._checkout import (
    checkout_scope,
    ensure_not_checked_out,
    new_checkout_owners,
)
from atla.dialect import Dialect, MySQLDialect, TiDBDialect


class AsyncmyAdapter:
    """Adapter over ``asyncmy`` pool or connection using tuple cursors.

    Pool-backed instances are safe to share across concurrent tasks. While a
    task holds ``async with adapter.transaction()``, that same task must use
    the yielded adapter (not the outer pool adapter). Peer tasks — including
    children created with ``asyncio.create_task`` — may still acquire from the
    pool.

    .. warning::
        With a small pool (e.g. ``maxsize=1``), a parent that holds a
        transaction and awaits a child task which uses the *outer* adapter can
        hang indefinitely waiting for a free connection. Previously this raised
        ``RuntimeError``; now the child blocks in ``pool.acquire()`` until the
        parent releases its connection. Size the pool for nested concurrency or
        pass the yielded adapter to children instead.
    """

    __slots__ = ("_checkout_owners", "_conn", "_depth", "_dialect", "_pool", "_savepoint")

    def __init__(
        self,
        pool_or_conn: Any,
        *,
        dialect: Dialect | None = None,
        _depth: int = 0,
        _savepoint: str | None = None,
    ) -> None:
        self._dialect = dialect or MySQLDialect()
        if not isinstance(self._dialect, (MySQLDialect, TiDBDialect)):
            msg = "AsyncmyAdapter requires MySQLDialect or TiDBDialect"
            raise TypeError(msg)
        self._depth = _depth
        self._checkout_owners = new_checkout_owners()
        self._savepoint = _savepoint
        if hasattr(pool_or_conn, "acquire") and not hasattr(pool_or_conn, "cursor"):
            self._pool = pool_or_conn
            self._conn: Any | None = None
        else:
            self._pool = None
            self._conn = pool_or_conn

    @property
    def dialect(self) -> Dialect:
        return self._dialect

    async def _acquire(self) -> Any:
        ensure_not_checked_out(self._checkout_owners)
        if self._conn is not None:
            return self._conn
        assert self._pool is not None
        return await self._pool.acquire()

    async def _release(self, conn: Any) -> None:
        if self._pool is not None and self._conn is None:
            await self._pool.release(conn)

    async def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Sequence[Any]]:
        conn = await self._acquire()
        try:
            async with conn.cursor() as cursor:
                await cursor.execute(sql, params)
                rows = await cursor.fetchall()
                return cast(Sequence[Sequence[Any]], rows)
        finally:
            await self._release(conn)

    async def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Any] | None:
        conn = await self._acquire()
        try:
            async with conn.cursor() as cursor:
                await cursor.execute(sql, params)
                return cast(Sequence[Any] | None, await cursor.fetchone())
        finally:
            await self._release(conn)

    async def fetch_val(self, sql: str, params: Sequence[Any] = ()) -> Any:
        row = await self.fetch_one(sql, params)
        if row is None:
            return None
        return row[0]

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        conn = await self._acquire()
        try:
            async with conn.cursor() as cursor:
                return int(await cursor.execute(sql, params))
        finally:
            await self._release(conn)

    async def execute_many(self, sql: str, params_seq: Sequence[Sequence[Any]]) -> None:
        conn = await self._acquire()
        try:
            async with conn.cursor() as cursor:
                await cursor.executemany(sql, params_seq)
        finally:
            await self._release(conn)

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[AsyncmyAdapter]:
        if self._pool is not None and self._conn is None:
            # Per-task checkout: peers may still acquire from the same pool.
            with checkout_scope(self._checkout_owners):
                active: Any = await self._pool.acquire()
                try:
                    try:
                        await active.begin()
                    except BaseException:
                        try:
                            await active.rollback()
                        except BaseException:
                            pass
                        raise
                    pinned = AsyncmyAdapter(active, dialect=self._dialect, _depth=1)
                    try:
                        yield pinned
                    except BaseException:
                        await active.rollback()
                        raise
                    else:
                        try:
                            await active.commit()
                        except BaseException:
                            try:
                                await active.rollback()
                            except BaseException:
                                pass
                            raise
                finally:
                    await self._pool.release(active)
            return

        assert self._conn is not None
        if self._depth == 0:
            await self._conn.begin()
            pinned = AsyncmyAdapter(self._conn, dialect=self._dialect, _depth=1)
            try:
                yield pinned
            except BaseException:
                await self._conn.rollback()
                raise
            else:
                try:
                    await self._conn.commit()
                except BaseException:
                    try:
                        await self._conn.rollback()
                    except BaseException:
                        pass
                    raise
            return

        savepoint = f"atla_sp_{self._depth}"
        async with self._conn.cursor() as cursor:
            await cursor.execute(f"SAVEPOINT {savepoint}")
        pinned = AsyncmyAdapter(
            self._conn,
            dialect=self._dialect,
            _depth=self._depth + 1,
            _savepoint=savepoint,
        )
        try:
            yield pinned
        except BaseException:
            async with self._conn.cursor() as cursor:
                await cursor.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                await cursor.execute(f"RELEASE SAVEPOINT {savepoint}")
            raise
        else:
            async with self._conn.cursor() as cursor:
                await cursor.execute(f"RELEASE SAVEPOINT {savepoint}")
