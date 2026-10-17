"""Primary / read-replica routing adapter."""

from __future__ import annotations

import itertools
import random
import re
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from typing import Any, Literal

from atla.adapter.base import DBAdapter, IsolationLevel
from atla.dialect import Dialect

# Postgres/MySQL row locks that must hit primary (incl. weaker / MySQL variants).
_LOCK_CLAUSE = re.compile(
    r"(?:"
    r"\bFOR\s+(?:NO\s+KEY\s+)?UPDATE\b|"
    r"\bFOR\s+(?:KEY\s+)?SHARE\b|"
    r"\bLOCK\s+IN\s+SHARE\s+MODE\b"
    r")",
    re.IGNORECASE,
)
_WRITE_PREFIX = re.compile(
    r"^(INSERT|UPDATE|DELETE|MERGE|REPLACE|CREATE|DROP|ALTER|TRUNCATE)\b",
    re.IGNORECASE,
)
_WRITE_IN_CTE = re.compile(r"\b(INSERT|UPDATE|DELETE|MERGE)\b", re.IGNORECASE)


def _strip_leading_sql_comments(sql: str) -> str:
    """Remove leading whitespace and ``--`` / ``/* */`` comments before verb matching."""
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]
        if ch.isspace():
            i += 1
            continue
        if sql.startswith("--", i):
            nl = sql.find("\n", i)
            i = n if nl < 0 else nl + 1
            continue
        if sql.startswith("/*", i):
            end = sql.find("*/", i + 2)
            if end < 0:
                return ""
            i = end + 2
            continue
        break
    return sql[i:]


def _is_write_sql(sql: str) -> bool:
    """Return True for DML/DDL, including ``WITH … INSERT/UPDATE/DELETE``.

    Leading SQL comments are stripped so ``/* … */ INSERT`` / ``-- …\\nUPDATE`` still
    route to primary.
    """
    stripped = _strip_leading_sql_comments(sql)
    if _WRITE_PREFIX.search(stripped):
        return True
    return stripped.upper().startswith("WITH") and _WRITE_IN_CTE.search(stripped) is not None


class RoutingAdapter:
    """Route reads to replicas and writes / transactions to primary.

    Non-transactional ``fetch_*`` calls go to a replica (round-robin by default)
    unless the SQL is a write (including ``INSERT … RETURNING`` via ``fetch_*``)
    or contains a locking clause (``FOR UPDATE`` / ``FOR NO KEY UPDATE`` /
    ``FOR SHARE`` / ``FOR KEY SHARE`` / MySQL ``LOCK IN SHARE MODE``).

    Replica selection is per-call, so multi-statement logical operations may see
    different replicas (and therefore replication lag). Prefer a single replica,
    sticky session, or primary reads when read-your-writes consistency matters.
    Transactions always use primary.
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
        if _LOCK_CLAUSE.search(sql) or _is_write_sql(sql):
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
