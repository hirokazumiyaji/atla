"""Transaction helpers: isolation options and serialization retry."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import Any, Literal, TypeVar, get_args

from atla.adapter.base import DBAdapter
from atla.errors import SerializationError

T = TypeVar("T")

IsolationLevel = Literal[
    "read_uncommitted",
    "read_committed",
    "repeatable_read",
    "serializable",
]

_ALLOWED_ISOLATION: frozenset[str] = frozenset(get_args(IsolationLevel))


async def run_in_transaction[T](
    adapter: DBAdapter,
    fn: Callable[[DBAdapter], Awaitable[T]],
    *,
    max_retries: int = 3,
    base_backoff: float = 0.01,
    isolation: IsolationLevel | None = None,
    readonly: bool = False,
    deferrable: bool = False,
) -> T:
    """Run *fn* inside a transaction, retrying on :class:`SerializationError`.

    ``max_retries`` is the number of *retries* after the first attempt (so the
    callback may run up to ``max_retries + 1`` times).

    *fn* must be idempotent: serialization failures re-run the whole callback
    (including any non-DB side effects). ``CancelledError`` and other
    ``BaseException`` subclasses are not retried.
    """
    if max_retries < 0:
        msg = "max_retries must be >= 0"
        raise ValueError(msg)
    attempt = 0
    while True:
        try:
            async with adapter.transaction(
                isolation=isolation,
                readonly=readonly,
                deferrable=deferrable,
            ) as tx:
                return await fn(tx)
        except SerializationError:
            if attempt >= max_retries:
                raise
            delay = base_backoff * (2**attempt)
            delay *= 0.5 + random.random()  # full jitter in [0.5, 1.5) * delay
            await asyncio.sleep(delay)
            attempt += 1


def begin_sql(
    dialect_name: str,
    *,
    isolation: IsolationLevel | None,
    readonly: bool,
    deferrable: bool,
) -> list[str]:
    """Return dialect-specific statements to configure a new transaction.

    Support matrix:

    - Postgres / Cockroach: ``isolation``, ``readonly``, ``deferrable``
    - MySQL / TiDB: ``isolation``, ``readonly`` (``deferrable`` raises)
    - SQLite: none of the options (any non-default raises)
    """
    statements: list[str] = []
    if dialect_name == "sqlite" and (isolation is not None or readonly or deferrable):
        msg = "SQLite does not support isolation/readonly/deferrable transaction options"
        raise ValueError(msg)
    if deferrable and dialect_name not in {"postgres", "cockroach"}:
        msg = f"{dialect_name} does not support deferrable transactions"
        raise ValueError(msg)
    if isolation is not None:
        if isolation not in _ALLOWED_ISOLATION:
            msg = f"unknown isolation level {isolation!r}"
            raise ValueError(msg)
        level = isolation.replace("_", " ").upper()
        if dialect_name in {"postgres", "cockroach", "mysql", "tidb"}:
            statements.append(f"SET TRANSACTION ISOLATION LEVEL {level}")
        else:
            msg = f"{dialect_name} does not support isolation level {isolation!r}"
            raise ValueError(msg)
    if readonly:
        if dialect_name in {"postgres", "cockroach", "mysql", "tidb"}:
            statements.append("SET TRANSACTION READ ONLY")
        else:
            msg = f"{dialect_name} does not support readonly transactions"
            raise ValueError(msg)
    if deferrable and dialect_name in {"postgres", "cockroach"}:
        statements.append("SET TRANSACTION DEFERRABLE")
    return statements


def map_asyncpg_isolation(isolation: IsolationLevel | None) -> Any | None:
    """Map atla isolation names to asyncpg transaction isolation constants."""
    if isolation is None:
        return None
    if isolation not in _ALLOWED_ISOLATION:
        msg = f"unknown isolation level {isolation!r}"
        raise ValueError(msg)
    mapping = {
        "read_uncommitted": "read_uncommitted",
        "read_committed": "read_committed",
        "repeatable_read": "repeatable_read",
        "serializable": "serializable",
    }
    return mapping[isolation]
