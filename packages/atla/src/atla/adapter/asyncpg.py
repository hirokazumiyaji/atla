"""asyncpg adapter for PostgreSQL and CockroachDB."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from typing import Any, cast

from atla.adapter._checkout import (
    checkout_scope,
    ensure_not_checked_out,
    new_checkout_owners,
)
from atla.dialect import CockroachDialect, Dialect, PostgresDialect


def _parse_status(status: str) -> int:
    """Parse asyncpg command status tags like ``UPDATE 3``."""
    parts = status.split()
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return 0


class AsyncpgAdapter:
    """Adapter over ``asyncpg.Pool`` or a pinned ``asyncpg.Connection``.

    Pool-backed instances are safe to share across concurrent tasks. While a
    task holds ``async with adapter.transaction()``, that same task must use
    the yielded adapter (not the outer pool adapter). Peer tasks — including
    children created with ``asyncio.create_task`` — may still acquire from the
    pool.

    .. warning::
        With a small pool (e.g. ``max_size=1``), a parent that holds a
        transaction and awaits a child task which uses the *outer* adapter can
        hang indefinitely waiting for a free connection. Previously this raised
        ``RuntimeError``; now the child blocks in ``pool.acquire()`` until the
        parent releases its connection. Size the pool for nested concurrency or
        pass the yielded adapter to children instead.
    """

    __slots__ = ("_checkout_owners", "_depth", "_dialect", "_pool_mode", "_target")

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
        self._checkout_owners = new_checkout_owners()
        # asyncpg.Pool exposes fetch/fetchrow proxies but has acquire();
        # Connection does not. Prefer acquire() as the sole pool marker.
        self._pool_mode = hasattr(pool_or_conn, "acquire")

    # Public ``dialect`` accessor; unrelated to the per-task checkout work (#26)
    # and kept here only because callers read ``adapter.dialect``.
    @property
    def dialect(self) -> Dialect:
        return self._dialect

    async def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Sequence[Any]]:
        ensure_not_checked_out(self._checkout_owners)
        encoded = self._dialect.encode_params(params)
        # Pool exposes fetch/fetchrow/fetchval/execute directly (C-backed acquire).
        return cast(Sequence[Sequence[Any]], await self._target.fetch(sql, *encoded))

    async def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Any] | None:
        ensure_not_checked_out(self._checkout_owners)
        encoded = self._dialect.encode_params(params)
        return cast(Sequence[Any] | None, await self._target.fetchrow(sql, *encoded))

    async def fetch_val(self, sql: str, params: Sequence[Any] = ()) -> Any:
        ensure_not_checked_out(self._checkout_owners)
        encoded = self._dialect.encode_params(params)
        return await self._target.fetchval(sql, *encoded)

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        ensure_not_checked_out(self._checkout_owners)
        encoded = self._dialect.encode_params(params)
        status = await self._target.execute(sql, *encoded)
        return _parse_status(str(status))

    async def execute_many(self, sql: str, params_seq: Sequence[Sequence[Any]]) -> None:
        ensure_not_checked_out(self._checkout_owners)
        encoded_seq = self._dialect.encode_params_seq(params_seq)
        await self._target.executemany(sql, encoded_seq)

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[AsyncpgAdapter]:
        # Match fetch_*/execute*: the outer pool adapter must not be reused
        # while this task already holds a checkout (use the yielded adapter).
        ensure_not_checked_out(self._checkout_owners)
        if self._pool_mode:
            # Per-task checkout: peers may still acquire from the same pool.
            with checkout_scope(self._checkout_owners):
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
