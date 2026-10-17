"""Stdlib-only instrumented adapter wrapper with optional slow-query hooks."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from typing import Any

from atla.adapter.base import DBAdapter
from atla.dialect import Dialect
from atla.observability.sql_meta import extract_sql_meta

logger = logging.getLogger("atla.observability")


def transaction_kwargs(isolation: str | None, readonly: bool, deferrable: bool) -> dict[str, Any]:
    """Collect only the transaction options the caller actually set."""
    kwargs: dict[str, Any] = {}
    if isolation is not None:
        kwargs["isolation"] = isolation
    if readonly:
        kwargs["readonly"] = True
    if deferrable:
        kwargs["deferrable"] = True
    return kwargs


OnQueryEnd = Callable[[str, str | None, float, Sequence[Any] | None], None]
OnSlowQuery = Callable[[str, str | None, float, Sequence[Any] | None], None]
OnError = Callable[[str, BaseException, Sequence[Any] | None], None]


class InstrumentedAdapter:
    """Wrap a ``DBAdapter`` with timing / slow-query callbacks (no OTel required)."""

    __slots__ = (
        "_inner",
        "_on_error",
        "_on_query_end",
        "_on_slow_query",
        "_slow_query_ms",
    )

    def __init__(
        self,
        inner: DBAdapter,
        *,
        slow_query_ms: float | None = None,
        on_query_end: OnQueryEnd | None = None,
        on_slow_query: OnSlowQuery | None = None,
        on_error: OnError | None = None,
    ) -> None:
        self._inner = inner
        self._slow_query_ms = slow_query_ms
        self._on_query_end = on_query_end
        self._on_slow_query = on_slow_query
        self._on_error = on_error

    @property
    def dialect(self) -> Dialect:
        return self._inner.dialect

    async def _timed[T](
        self,
        sql: str,
        params: Sequence[Any] | None,
        coro: Awaitable[T],
    ) -> T:
        operation, collection = extract_sql_meta(sql)
        started = time.perf_counter()
        try:
            result = await coro
        except BaseException as exc:
            if self._on_error is not None:
                self._on_error(sql, exc, params)
            raise
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if self._on_query_end is not None:
            self._on_query_end(operation, collection, elapsed_ms, params)
        if (
            self._slow_query_ms is not None
            and elapsed_ms >= self._slow_query_ms
            and self._on_slow_query is not None
        ):
            self._on_slow_query(operation, collection, elapsed_ms, params)
        elif (
            self._slow_query_ms is not None
            and elapsed_ms >= self._slow_query_ms
            and self._on_slow_query is None
        ):
            logger.warning(
                "slow query %.1fms op=%s collection=%s",
                elapsed_ms,
                operation,
                collection,
            )
        return result

    async def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Sequence[Any]]:
        return await self._timed(sql, params, self._inner.fetch_all(sql, params))

    async def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Any] | None:
        return await self._timed(sql, params, self._inner.fetch_one(sql, params))

    async def fetch_val(self, sql: str, params: Sequence[Any] = ()) -> Any:
        return await self._timed(sql, params, self._inner.fetch_val(sql, params))

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        return await self._timed(sql, params, self._inner.execute(sql, params))

    async def execute_many(self, sql: str, params_seq: Sequence[Sequence[Any]]) -> None:
        await self._timed(sql, None, self._inner.execute_many(sql, params_seq))

    @asynccontextmanager
    async def transaction(
        self,
        *,
        isolation: str | None = None,
        readonly: bool = False,
        deferrable: bool = False,
    ) -> AsyncGenerator[InstrumentedAdapter, None]:
        """Begin a transaction on the wrapped adapter.

        Options are forwarded only when set, so adapters whose ``transaction()`` does
        not yet accept them keep working; adapters that do (#53) receive them intact.
        """
        async with self._inner.transaction(
            **transaction_kwargs(isolation, readonly, deferrable)
        ) as tx:
            yield InstrumentedAdapter(
                tx,
                slow_query_ms=self._slow_query_ms,
                on_query_end=self._on_query_end,
                on_slow_query=self._on_slow_query,
                on_error=self._on_error,
            )
