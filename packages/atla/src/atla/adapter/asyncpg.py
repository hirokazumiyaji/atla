"""asyncpg adapter for PostgreSQL and CockroachDB."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from typing import Any, cast

from atla.adapter.base import IsolationLevel
from atla.adapter.translate import reraise_translated, with_translated_errors
from atla.dialect import CockroachDialect, Dialect, PostgresDialect
from atla.transaction import map_asyncpg_isolation


def _parse_status(status: str) -> int:
    """Parse asyncpg command status tags like ``UPDATE 3``."""
    parts = status.split()
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return 0


class AsyncpgAdapter:
    """Adapter over ``asyncpg.Pool`` or a pinned ``asyncpg.Connection``."""

    __slots__ = ("_depth", "_dialect", "_pool_mode", "_target", "_tx_active")

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
        self._tx_active = False
        # asyncpg.Pool exposes fetch/fetchrow proxies but has acquire();
        # Connection does not. Prefer acquire() as the sole pool marker.
        self._pool_mode = hasattr(pool_or_conn, "acquire")

    @property
    def dialect(self) -> Dialect:
        return self._dialect

    def _ensure_not_checked_out(self) -> None:
        if self._tx_active:
            msg = (
                "connection is checked out for an active transaction; "
                "use the adapter yielded by transaction()"
            )
            raise RuntimeError(msg)

    @with_translated_errors
    async def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Sequence[Any]]:
        self._ensure_not_checked_out()
        if self._pool_mode:
            async with self._target.acquire() as conn:
                return cast(Sequence[Sequence[Any]], await conn.fetch(sql, *params))
        return cast(Sequence[Sequence[Any]], await self._target.fetch(sql, *params))

    @with_translated_errors
    async def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Any] | None:
        self._ensure_not_checked_out()
        if self._pool_mode:
            async with self._target.acquire() as conn:
                return cast(Sequence[Any] | None, await conn.fetchrow(sql, *params))
        return cast(Sequence[Any] | None, await self._target.fetchrow(sql, *params))

    @with_translated_errors
    async def fetch_val(self, sql: str, params: Sequence[Any] = ()) -> Any:
        self._ensure_not_checked_out()
        if self._pool_mode:
            async with self._target.acquire() as conn:
                return await conn.fetchval(sql, *params)
        return await self._target.fetchval(sql, *params)

    @with_translated_errors
    async def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        self._ensure_not_checked_out()
        if self._pool_mode:
            async with self._target.acquire() as conn:
                status = await conn.execute(sql, *params)
        else:
            status = await self._target.execute(sql, *params)
        return _parse_status(str(status))

    @with_translated_errors
    async def execute_many(self, sql: str, params_seq: Sequence[Sequence[Any]]) -> None:
        self._ensure_not_checked_out()
        if self._pool_mode:
            async with self._target.acquire() as conn:
                await conn.executemany(sql, params_seq)
            return
        await self._target.executemany(sql, params_seq)

    @asynccontextmanager
    async def transaction(
        self,
        *,
        isolation: IsolationLevel | None = None,
        readonly: bool = False,
        deferrable: bool = False,
    ) -> AsyncGenerator[AsyncpgAdapter]:
        isolation_kw = map_asyncpg_isolation(isolation)
        tx_kwargs: dict[str, Any] = {}
        if isolation_kw is not None:
            tx_kwargs["isolation"] = isolation_kw
        if readonly:
            tx_kwargs["readonly"] = True
        if deferrable:
            tx_kwargs["deferrable"] = True

        if self._pool_mode:
            self._tx_active = True
            try:
                async with self._target.acquire() as conn:
                    try:
                        async with conn.transaction(**tx_kwargs):
                            yield AsyncpgAdapter(conn, dialect=self._dialect, _depth=1)
                    except Exception as exc:
                        reraise_translated(exc)
            finally:
                self._tx_active = False
            return

        try:
            async with self._target.transaction(**tx_kwargs):
                yield AsyncpgAdapter(
                    self._target,
                    dialect=self._dialect,
                    _depth=self._depth + 1,
                )
        except Exception as exc:
            reraise_translated(exc)
