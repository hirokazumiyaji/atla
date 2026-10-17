"""asyncmy adapter for MySQL and TiDB."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from typing import Any, cast

from atla.dialect import Dialect, MySQLDialect, TiDBDialect


class AsyncmyAdapter:
    """Adapter over ``asyncmy`` pool or connection using tuple cursors."""

    __slots__ = ("_conn", "_depth", "_dialect", "_pool", "_savepoint", "_tx_active")

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
        self._savepoint = _savepoint
        self._tx_active = False
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
        if self._tx_active:
            msg = (
                "connection is checked out for an active transaction; "
                "use the adapter yielded by transaction()"
            )
            raise RuntimeError(msg)
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
            self._tx_active = True
            conn = await self._pool.acquire()
            try:
                await conn.begin()
                pinned = AsyncmyAdapter(conn, dialect=self._dialect, _depth=1)
                try:
                    yield pinned
                except BaseException:
                    await conn.rollback()
                    raise
                else:
                    await conn.commit()
            finally:
                await self._pool.release(conn)
                self._tx_active = False
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
                await self._conn.commit()
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
