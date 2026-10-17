"""Shared SELECT clause helpers for query builders."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from atla.compiler import SQLCompiler
from atla.dialect import Dialect
from atla.errors import MultipleResultsError, NotFoundError
from atla.expressions import Expr, OrderByClause, and_
from atla.table import TableRef


@dataclass(slots=True, frozen=True)
class LockClause:
    mode: Literal["UPDATE", "SHARE"]
    nowait: bool = False
    skip_locked: bool = False
    of: tuple[str, ...] = ()


def lock_of_targets(tables: Sequence[TableRef[Any]], dialect: Dialect) -> tuple[str, ...]:
    # Prefer the SQL alias (self-joins / ``as_``); Postgres hides the base name.
    return tuple(dialect.quote(table.sql_alias()) for table in tables)


def write_distinct(
    compiler: SQLCompiler,
    dialect: Dialect,
    *,
    distinct: bool,
    distinct_on: tuple[Expr[Any], ...],
) -> None:
    if distinct_on:
        if not dialect.supports_distinct_on:
            msg = f"{dialect.name} does not support DISTINCT ON"
            raise ValueError(msg)
        compiler.write("DISTINCT ON (")
        for index, expr in enumerate(distinct_on):
            if index:
                compiler.write(", ")
            compiler.visit(expr)
        compiler.write(") ")
    elif distinct:
        compiler.write("DISTINCT ")


def write_group_having(
    compiler: SQLCompiler,
    *,
    group_by: tuple[Expr[Any], ...],
    having: tuple[Expr[bool], ...],
) -> None:
    if group_by:
        compiler.write(" GROUP BY ")
        for index, expr in enumerate(group_by):
            if index:
                compiler.write(", ")
            compiler.visit(expr)
    if having:
        compiler.write(" HAVING ")
        compiler.visit(and_(*having))


def write_order_limit_offset(
    compiler: SQLCompiler,
    *,
    order_by: tuple[OrderByClause, ...],
    limit: int | None,
    offset: int | None,
) -> None:
    if order_by:
        compiler.write(" ORDER BY ")
        for index, clause in enumerate(order_by):
            if index:
                compiler.write(", ")
            compiler.visit_order_by(clause)
    if limit is not None:
        compiler.write(" LIMIT ")
        compiler.write(compiler.add_param(limit))
    if offset is not None:
        compiler.write(" OFFSET ")
        compiler.write(compiler.add_param(offset))


def write_locking(compiler: SQLCompiler, dialect: Dialect, locking: LockClause | None) -> None:
    if locking is None:
        return
    if not dialect.supports_for_update:
        msg = f"{dialect.name} does not support FOR UPDATE / FOR SHARE"
        raise ValueError(msg)
    if locking.nowait and locking.skip_locked:
        msg = "nowait and skip_locked cannot both be True"
        raise ValueError(msg)
    if locking.skip_locked and not dialect.supports_skip_locked:
        msg = f"{dialect.name} does not support SKIP LOCKED"
        raise ValueError(msg)
    if locking.of and not dialect.supports_lock_of:
        msg = f"{dialect.name} does not support FOR UPDATE/SHARE OF"
        raise ValueError(msg)
    # TiDB accepts LOCK IN SHARE MODE (ignored) but not FOR SHARE.
    if locking.mode == "SHARE" and not dialect.supports_for_share:
        if dialect.name == "tidb":
            compiler.write(" LOCK IN SHARE MODE")
            return
        msg = f"{dialect.name} does not support FOR SHARE"
        raise ValueError(msg)
    compiler.write(" FOR ")
    compiler.write(locking.mode)
    if locking.of:
        compiler.write(" OF ")
        compiler.write(", ".join(locking.of))
    if locking.nowait:
        compiler.write(" NOWAIT")
    elif locking.skip_locked:
        compiler.write(" SKIP LOCKED")


async def fetch_first[T](fetch_limited: Callable[[], Awaitable[list[T]]]) -> T | None:
    """Return the first row from a limit-1 fetch, or ``None``."""
    rows = await fetch_limited()
    return rows[0] if rows else None


async def fetch_one[T](fetch_limited: Callable[[], Awaitable[list[T]]]) -> T:
    """Return exactly one row from a limit-2 fetch, or raise."""
    rows = await fetch_limited()
    if not rows:
        raise NotFoundError("expected exactly one row, found 0")
    if len(rows) > 1:
        raise MultipleResultsError("expected exactly one row, found multiple")
    return rows[0]
