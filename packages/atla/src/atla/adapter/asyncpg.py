"""asyncpg adapter for PostgreSQL and CockroachDB."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from typing import Any, cast

from atla.dialect import CockroachDialect, Dialect, PostgresDialect


def _parse_status(status: str) -> int:
    """Parse asyncpg command status tags like ``UPDATE 3``."""
    parts = status.split()
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return 0


class AsyncpgAdapter:
    """Adapter over ``asyncpg.Pool`` or a pinned ``asyncpg.Connection``."""

    __slots__ = ("_depth", "_dialect", "_pool_mode", "_target")

    def __init__(
        self,
        pool_or_conn: Any,
        *,
        dialect: Dialect | None = None,
        _depth: int = 0,
    ) -> None:
        self._target = pool_or_conn
        self._dialect = dialect or PostgresDialect()
        if not isinstance(self._dialect, (PostgresDialect, CockroachDialect)):
            msg = "AsyncpgAdapter requires PostgresDialect or CockroachDialect"
            raise TypeError(msg)
        self._depth = _depth
        # Prefer explicit pool markers over MagicMock getattr auto-creation.
        self._pool_mode = hasattr(pool_or_conn, "acquire") and not hasattr(pool_or_conn, "fetchrow")
        # Real asyncpg.Connection always exposes fetchrow; pools do not.

    async def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Sequence[Any]]:
        if self._pool_mode:
            async with self._target.acquire() as conn:
                return cast(Sequence[Sequence[Any]], await conn.fetch(sql, *params))
        return cast(Sequence[Sequence[Any]], await self._target.fetch(sql, *params))

    async def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Any] | None:
        if self._pool_mode:
            async with self._target.acquire() as conn:
                return cast(Sequence[Any] | None, await conn.fetchrow(sql, *params))
        return cast(Sequence[Any] | None, await self._target.fetchrow(sql, *params))

    async def fetch_val(self, sql: str, params: Sequence[Any] = ()) -> Any:
        if self._pool_mode:
            async with self._target.acquire() as conn:
                return await conn.fetchval(sql, *params)
        return await self._target.fetchval(sql, *params)

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        if self._pool_mode:
            async with self._target.acquire() as conn:
                status = await conn.execute(sql, *params)
        else:
            status = await self._target.execute(sql, *params)
        return _parse_status(str(status))

    async def execute_many(self, sql: str, params_seq: Sequence[Sequence[Any]]) -> None:
        if self._pool_mode:
            async with self._target.acquire() as conn:
                await conn.executemany(sql, params_seq)
            return
        await self._target.executemany(sql, params_seq)

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[AsyncpgAdapter]:
        if self._pool_mode:
            async with self._target.acquire() as conn:
                async with conn.transaction():
                    yield AsyncpgAdapter(conn, dialect=self._dialect, _depth=1)
            return

        async with self._target.transaction():
            yield AsyncpgAdapter(
                self._target,
                dialect=self._dialect,
                _depth=self._depth + 1,
            )
