"""Primary / read-replica routing adapter."""

from __future__ import annotations

import itertools
import random
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from typing import Any, Literal

from atla.adapter.base import DBAdapter, IsolationLevel
from atla.dialect import Dialect

_WriteKind = Literal["execute", "execute_many", "transaction"]


class RoutingAdapter:
    """Route reads to replicas and writes / transactions to primary.

    Non-transactional ``fetch_*`` calls go to a replica (round-robin by default).
    Statements containing ``FOR UPDATE`` / ``FOR SHARE`` always hit primary.
    """

    __slots__ = ("_primary", "_replicas", "_rr", "_strategy")

    def __init__(
        self,
        primary: DBAdapter,
        replicas: Sequence[DBAdapter] = (),
        *,
        strategy: Literal["round_robin", "random"] = "round_robin",
    ) -> None:
        if not replicas:
            replicas = (primary,)
        self._primary = primary
        self._replicas = tuple(replicas)
        self._strategy = strategy
        self._rr = itertools.cycle(range(len(self._replicas)))

    @property
    def dialect(self) -> Dialect:
        return self._primary.dialect

    def _pick_replica(self) -> DBAdapter:
        if self._strategy == "random":
            return random.choice(self._replicas)
        return self._replicas[next(self._rr)]

    def _read_target(self, sql: str) -> DBAdapter:
        upper = sql.upper()
        if " FOR UPDATE" in upper or " FOR SHARE" in upper:
            return self._primary
        return self._pick_replica()

    async def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Sequence[Any]]:
        return await self._read_target(sql).fetch_all(sql, params)

    async def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Any] | None:
        return await self._read_target(sql).fetch_one(sql, params)

    async def fetch_val(self, sql: str, params: Sequence[Any] = ()) -> Any:
        return await self._read_target(sql).fetch_val(sql, params)

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        return await self._primary.execute(sql, params)

    async def execute_many(self, sql: str, params_seq: Sequence[Sequence[Any]]) -> None:
        await self._primary.execute_many(sql, params_seq)

    @asynccontextmanager
    async def transaction(
        self,
        *,
        isolation: IsolationLevel | None = None,
        readonly: bool = False,
        deferrable: bool = False,
    ) -> AsyncGenerator[DBAdapter]:
        async with self._primary.transaction(
            isolation=isolation,
            readonly=readonly,
            deferrable=deferrable,
        ) as tx:
            yield tx
