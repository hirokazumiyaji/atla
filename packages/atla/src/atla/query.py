"""Immutable query / mutation builders shared by generated repositories."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import starmap
from typing import Any, ClassVar, Literal, Self, cast, overload

from atla.adapter.base import DBAdapter
from atla.advanced import ProjectedSelectQuery, UpsertQuery, conflict_column_names
from atla.compiler import SQLCompiler
from atla.errors import MultipleResultsError, NotFoundError, UnsafeMutationError
from atla.expressions import Column, Expr, OrderByClause, SubqueryExpr, and_
from atla.prepared import CompiledQuery, rows_mapper_from_ctor, rows_mapper_from_record
from atla.select_clauses import (
    LockClause,
    fetch_one,
    lock_of_targets,
    write_distinct,
    write_group_having,
    write_locking,
    write_order_limit_offset,
)
from atla.table import TableRef
from atla.types import is_unset

# Stay under common driver bind limits (SQLite ~32766, MySQL/Postgres similar).
_IN_BATCH_SIZE = 500
_LOCK_UNSET = object()


@dataclass(slots=True, frozen=True)
class _JoinSegment:
    """One SELECT projection segment produced by a base table or JOIN."""

    select_list: str
    column_count: int
    mapper: Callable[[Sequence[Any]], Any]
    nullable: bool


@dataclass(slots=True, frozen=True)
class _JoinClause:
    kind: Literal["INNER", "LEFT"]
    target_sql: str
    on: tuple[Expr[bool], ...]
    segment: _JoinSegment


def _compiled_mutation[RecordT](
    adapter: DBAdapter,
    compiler: SQLCompiler,
    *,
    mapper: Callable[[Sequence[Any]], RecordT] | None,
) -> CompiledQuery[RecordT]:
    """Build a compiled template for one mutation statement.

    ``mapper`` is ``None`` for statements without ``RETURNING``; the resulting
    ``CompiledQuery`` then only exposes ``execute()``.
    """
    sql, params = compiler.build()
    return CompiledQuery(
        adapter,
        sql=sql,
        params=params,
        slot_occurrences=compiler.slot_occurrences,
        row_mapper=mapper,
        rows_mapper=None if mapper is None else rows_mapper_from_record(mapper),
    )


class _ProjectableSelect:
    """Shared ``select()`` overloads for builders that can project columns.

    Typed for 1-8 projected expressions (``ProjectedSelectQuery[T1, ..., T8]``);
    pass more than 8 and the runtime still works but the result is untyped
    (type checkers will report no matching overload).
    Subclasses implement :meth:`_project`.
    """

    __slots__ = ()

    @overload
    def select[T](self, c0: Expr[T]) -> ProjectedSelectQuery[T]: ...

    @overload
    def select[T1, T2](self, c0: Expr[T1], c1: Expr[T2]) -> ProjectedSelectQuery[T1, T2]: ...

    @overload
    def select[T1, T2, T3](
        self, c0: Expr[T1], c1: Expr[T2], c2: Expr[T3]
    ) -> ProjectedSelectQuery[T1, T2, T3]: ...

    @overload
    def select[T1, T2, T3, T4](
        self, c0: Expr[T1], c1: Expr[T2], c2: Expr[T3], c3: Expr[T4]
    ) -> ProjectedSelectQuery[T1, T2, T3, T4]: ...

    @overload
    def select[T1, T2, T3, T4, T5](
        self, c0: Expr[T1], c1: Expr[T2], c2: Expr[T3], c3: Expr[T4], c4: Expr[T5]
    ) -> ProjectedSelectQuery[T1, T2, T3, T4, T5]: ...

    @overload
    def select[T1, T2, T3, T4, T5, T6](
        self,
        c0: Expr[T1],
        c1: Expr[T2],
        c2: Expr[T3],
        c3: Expr[T4],
        c4: Expr[T5],
        c5: Expr[T6],
    ) -> ProjectedSelectQuery[T1, T2, T3, T4, T5, T6]: ...

    @overload
    def select[T1, T2, T3, T4, T5, T6, T7](
        self,
        c0: Expr[T1],
        c1: Expr[T2],
        c2: Expr[T3],
        c3: Expr[T4],
        c4: Expr[T5],
        c5: Expr[T6],
        c6: Expr[T7],
    ) -> ProjectedSelectQuery[T1, T2, T3, T4, T5, T6, T7]: ...

    @overload
    def select[T1, T2, T3, T4, T5, T6, T7, T8](
        self,
        c0: Expr[T1],
        c1: Expr[T2],
        c2: Expr[T3],
        c3: Expr[T4],
        c4: Expr[T5],
        c5: Expr[T6],
        c6: Expr[T7],
        c7: Expr[T8],
    ) -> ProjectedSelectQuery[T1, T2, T3, T4, T5, T6, T7, T8]: ...

    def select(self, *columns: Expr[Any]) -> Any:  # type: ignore[misc]
        """Project specific columns/expressions as ``list[tuple[...]]``.

        Supports up to 8 typed columns/expressions (overloads); more than 8 are
        accepted at runtime but are not statically typed.

        Row codecs (UUID/Decimal/datetime/...) are applied only for the generated
        table column descriptors (e.g. ``users.id``), which carry a ``decoder``.
        Hand-built ``Column("users", "id")`` objects and other expressions return
        the adapter's raw values.
        """
        if not columns:
            msg = "select() requires at least one column"
            raise ValueError(msg)
        return self._project(columns)

    def _project(self, columns: tuple[Expr[Any], ...]) -> ProjectedSelectQuery[Any]:
        raise NotImplementedError


class BaseSelectQuery[RecordT](_ProjectableSelect):
    """Immutable SELECT query builder backed by AOT SQL fragments."""

    __slots__ = (
        "_adapter",
        "_distinct",
        "_distinct_on",
        "_group_by",
        "_having",
        "_limit",
        "_locking",
        "_offset",
        "_order_by",
        "_where",
    )

    _TABLE_NAME: str
    _SELECT_SQL: str
    _COUNT_SQL: str
    _ROW_MAPPER: Callable[[Sequence[Any]], RecordT]
    # When set (codec-free tables), ``all()`` uses ``starmap`` for C-level hydration.
    _ROW_CTOR: ClassVar[type[Any] | None] = None
    _SELECT_COLS: str
    _COLUMN_COUNT: int
    _FROM_SQL: str

    def __init__(
        self,
        adapter: DBAdapter,
        *,
        where: tuple[Expr[bool], ...] = (),
        order_by: tuple[OrderByClause, ...] = (),
        limit: int | None = None,
        offset: int | None = None,
        group_by: tuple[Expr[Any], ...] = (),
        having: tuple[Expr[bool], ...] = (),
        distinct: bool = False,
        distinct_on: tuple[Expr[Any], ...] = (),
        locking: LockClause | None = None,
    ) -> None:
        self._adapter = adapter
        self._where = where
        self._order_by = order_by
        self._limit = limit
        self._offset = offset
        self._group_by = group_by
        self._having = having
        self._distinct = distinct
        self._distinct_on = distinct_on
        self._locking = locking

    @property
    def adapter(self) -> DBAdapter:
        return self._adapter

    def _clone(
        self,
        *,
        where: tuple[Expr[bool], ...] | None = None,
        order_by: tuple[OrderByClause, ...] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        replace_limit: bool = False,
        replace_offset: bool = False,
        group_by: tuple[Expr[Any], ...] | None = None,
        having: tuple[Expr[bool], ...] | None = None,
        distinct: bool | None = None,
        distinct_on: tuple[Expr[Any], ...] | None = None,
        locking: LockClause | object | None = _LOCK_UNSET,
    ) -> Self:
        return type(self)(
            self._adapter,
            where=self._where if where is None else where,
            order_by=self._order_by if order_by is None else order_by,
            limit=limit if replace_limit else self._limit,
            offset=offset if replace_offset else self._offset,
            group_by=self._group_by if group_by is None else group_by,
            having=self._having if having is None else having,
            distinct=self._distinct if distinct is None else distinct,
            distinct_on=self._distinct_on if distinct_on is None else distinct_on,
            locking=self._locking if locking is _LOCK_UNSET else cast(LockClause | None, locking),
        )

    def where(self, *predicates: Expr[bool]) -> Self:
        if not predicates:
            return self
        return self._clone(where=self._where + predicates)

    def order_by(self, *clauses: OrderByClause) -> Self:
        if not clauses:
            return self
        return self._clone(order_by=self._order_by + clauses)

    def limit(self, n: int) -> Self:
        return self._clone(limit=n, replace_limit=True)

    def offset(self, n: int) -> Self:
        return self._clone(offset=n, replace_offset=True)

    def group_by(self, *exprs: Expr[Any]) -> Self:
        if not exprs:
            return self
        return self._clone(group_by=self._group_by + exprs)

    def having(self, *predicates: Expr[bool]) -> Self:
        if not predicates:
            return self
        return self._clone(having=self._having + predicates)

    def distinct(self, *on: Expr[Any]) -> Self:
        if on:
            return self._clone(distinct=True, distinct_on=self._distinct_on + on)
        # Keep any existing DISTINCT ON expressions; bare distinct() must not clear them.
        return self._clone(distinct=True)

    def for_update(
        self,
        *,
        nowait: bool = False,
        skip_locked: bool = False,
        of: Sequence[TableRef[Any]] = (),
    ) -> Self:
        return self._clone(
            locking=LockClause(
                mode="UPDATE",
                nowait=nowait,
                skip_locked=skip_locked,
                of=lock_of_targets(of, self._adapter.dialect),
            )
        )

    def for_share(
        self,
        *,
        nowait: bool = False,
        skip_locked: bool = False,
        of: Sequence[TableRef[Any]] = (),
    ) -> Self:
        return self._clone(
            locking=LockClause(
                mode="SHARE",
                nowait=nowait,
                skip_locked=skip_locked,
                of=lock_of_targets(of, self._adapter.dialect),
            )
        )

    def join[OtherT](
        self,
        table: TableRef[OtherT],
        *on: Expr[bool],
    ) -> JoinedSelectQuery[RecordT, OtherT]:
        """INNER JOIN ``table`` and widen the row type to ``tuple[RecordT, OtherT]``."""
        return self._start_join(table, *on, kind="INNER", nullable=False)

    def left_join[OtherT](
        self,
        table: TableRef[OtherT],
        *on: Expr[bool],
    ) -> JoinedSelectQuery[RecordT, OtherT | None]:
        """LEFT JOIN ``table``; unmatched right sides hydrate as ``None``."""
        return self._start_join(table, *on, kind="LEFT", nullable=True)

    def _start_join[OtherT](
        self,
        table: TableRef[OtherT],
        *on: Expr[bool],
        kind: Literal["INNER", "LEFT"],
        nullable: bool,
    ) -> JoinedSelectQuery[Any, Any]:
        if not on:
            msg = "join() requires at least one ON predicate"
            raise ValueError(msg)
        dialect = self._adapter.dialect
        left = _JoinSegment(
            select_list=type(self)._SELECT_COLS,
            column_count=type(self)._COLUMN_COUNT,
            mapper=type(self)._ROW_MAPPER,
            nullable=False,
        )
        right = _JoinSegment(
            select_list=table.select_list(dialect),
            column_count=table.column_count,
            mapper=table.map_row_optional if nullable else table.map_row,
            nullable=nullable,
        )
        join = _JoinClause(
            kind=kind,
            target_sql=table.join_target(dialect),
            on=on,
            segment=right,
        )
        return JoinedSelectQuery(
            self._adapter,
            from_sql=type(self)._FROM_SQL,
            segments=(left, right),
            joins=(join,),
            where=self._where,
            order_by=self._order_by,
            limit=self._limit,
            offset=self._offset,
            group_by=self._group_by,
            having=self._having,
            distinct=self._distinct,
            distinct_on=self._distinct_on,
            locking=self._locking,
        )

    def preload[OtherT](
        self,
        related: TableRef[OtherT],
        *predicates: Expr[bool],
        fk: Column[Any],
        key: Column[Any],
        fk_attr: str,
        key_attr: str,
        order_by: Sequence[OrderByClause] = (),
    ) -> PreloadSelectQuery[RecordT, OtherT]:
        """Batch-load related rows (has-many) with one ``WHERE fk IN (...)`` follow-up.

        ``fk_attr`` / ``key_attr`` must be the generated Record attribute names
        (from atlagen), not re-derived SQL column names. Extra ``*predicates`` and
        ``order_by`` apply to the related-table query.
        """
        return PreloadSelectQuery(
            self,
            related=related,
            fk=fk,
            key=key,
            fk_attr=fk_attr,
            key_attr=key_attr,
            related_where=predicates,
            related_order_by=tuple(order_by),
        )

    def with_[OtherT](
        self,
        related: TableRef[OtherT],
        *predicates: Expr[bool],
        fk: Column[Any],
        key: Column[Any],
        fk_attr: str,
        key_attr: str,
        order_by: Sequence[OrderByClause] = (),
    ) -> PreloadSelectQuery[RecordT, OtherT]:
        """Alias for :meth:`preload` (batch eager loading without N+1 queries)."""
        return self.preload(
            related,
            *predicates,
            fk=fk,
            key=key,
            fk_attr=fk_attr,
            key_attr=key_attr,
            order_by=order_by,
        )

    @overload
    def preload_one[OtherT](
        self,
        related: TableRef[OtherT],
        /,
        *,
        fk: Column[Any],
        key: Column[Any],
        fk_attr: str,
        key_attr: str,
        order_by: Sequence[OrderByClause] = (),
        nullable: Literal[False] = False,
    ) -> PreloadOneSelectQuery[RecordT, OtherT]: ...

    @overload
    def preload_one[OtherT](
        self,
        related: TableRef[OtherT],
        /,
        *,
        fk: Column[Any],
        key: Column[Any],
        fk_attr: str,
        key_attr: str,
        order_by: Sequence[OrderByClause] = (),
        nullable: Literal[True],
    ) -> PreloadOneSelectQuery[RecordT, OtherT | None]: ...

    @overload
    def preload_one[OtherT](
        self,
        related: TableRef[OtherT],
        predicate: Expr[bool],
        /,
        *predicates: Expr[bool],
        fk: Column[Any],
        key: Column[Any],
        fk_attr: str,
        key_attr: str,
        order_by: Sequence[OrderByClause] = (),
        nullable: bool = False,
    ) -> PreloadOneSelectQuery[RecordT, OtherT | None]: ...

    def preload_one[OtherT](
        self,
        related: TableRef[OtherT],
        *predicates: Expr[bool],
        fk: Column[Any],
        key: Column[Any],
        fk_attr: str,
        key_attr: str,
        order_by: Sequence[OrderByClause] = (),
        nullable: bool = False,
    ) -> PreloadOneSelectQuery[RecordT, OtherT] | PreloadOneSelectQuery[RecordT, OtherT | None]:
        """Batch-load a belongs-to parent via ``WHERE parent.pk IN (...)``.

        ``fk`` / ``fk_attr`` belong to *this* table; ``key`` / ``key_attr`` belong
        to ``related``. When ``nullable`` is true, missing parents hydrate as
        ``None``. Parent-side ``predicates`` can filter out a real parent, so the
        result type becomes optional whenever predicates are supplied.
        """
        # Predicates may exclude the parent row; treat as optional so callers
        # get ``None`` instead of a NotFoundError / illegal ``ParentT`` hole.
        effective_nullable = nullable or bool(predicates)
        return PreloadOneSelectQuery(
            self,
            related=related,
            fk=fk,
            key=key,
            fk_attr=fk_attr,
            key_attr=key_attr,
            related_where=predicates,
            related_order_by=tuple(order_by),
            nullable=effective_nullable,
        )

    def _project(self, columns: tuple[Expr[Any], ...]) -> ProjectedSelectQuery[Any]:
        return ProjectedSelectQuery(
            self._adapter,
            from_sql=type(self)._FROM_SQL,
            columns=columns,
            where=self._where,
            order_by=self._order_by,
            limit=self._limit,
            offset=self._offset,
            group_by=self._group_by,
            having=self._having,
            distinct=self._distinct,
            distinct_on=self._distinct_on,
            locking=self._locking,
        )

    def _has_select_modifiers(self) -> bool:
        return bool(
            self._where
            or self._order_by
            or self._limit is not None
            or self._offset is not None
            or self._group_by
            or self._having
            or self._distinct
            or self._distinct_on
            or self._locking is not None
        )

    def _write_sql(self, compiler: SQLCompiler) -> None:
        dialect = compiler.dialect
        compiler.write("SELECT ")
        write_distinct(
            compiler,
            dialect,
            distinct=self._distinct,
            distinct_on=self._distinct_on,
        )
        compiler.write(f"{type(self)._SELECT_COLS} FROM {type(self)._FROM_SQL}")
        if self._where:
            compiler.write(" WHERE ")
            compiler.visit(and_(*self._where))
        write_group_having(compiler, group_by=self._group_by, having=self._having)
        write_order_limit_offset(
            compiler,
            order_by=self._order_by,
            limit=self._limit,
            offset=self._offset,
        )
        write_locking(compiler, dialect, self._locking)

    def to_sql(self) -> tuple[str, list[Any]]:
        compiler = SQLCompiler(self._adapter.dialect)
        self._write_sql(compiler)
        return compiler.build()

    def prepare(self) -> CompiledQuery[RecordT]:
        """Compile this query once into a reusable template.

        ``Param`` slots declared in the query become named bind parameters of the
        returned :class:`~atla.prepared.CompiledQuery`; see its docstring for the
        binding rules and for what ``first()`` does and does not do.
        """
        compiler = SQLCompiler(self._adapter.dialect, slots_enabled=True)
        self._write_sql(compiler)
        sql, params = compiler.build()
        ctor = type(self)._ROW_CTOR
        return CompiledQuery(
            self._adapter,
            sql=sql,
            params=params,
            slot_occurrences=compiler.slot_occurrences,
            row_mapper=type(self)._ROW_MAPPER,
            rows_mapper=(
                rows_mapper_from_ctor(ctor)
                if ctor is not None
                else rows_mapper_from_record(type(self)._ROW_MAPPER)
            ),
        )

    def as_subquery(self) -> SubqueryExpr[Any]:
        sql, params = self.to_sql()
        return SubqueryExpr(sql, params)

    def _is_static_select(self) -> bool:
        """True when the AOT ``_SELECT_SQL`` can be used verbatim (no builder clauses).

        Single source of truth for ``all()`` / ``first()`` fast paths. Any new clause
        that alters the SELECT text (GROUP BY, DISTINCT, ...) must be checked here.
        """
        return not self._has_select_modifiers()

    async def all(self) -> list[RecordT]:
        cls = type(self)
        if self._is_static_select():
            sql = cls._SELECT_SQL
            params: list[Any] = []
        else:
            sql, params = self.to_sql()
        rows = await self._adapter.fetch_all(sql, params)
        ctor = cls._ROW_CTOR
        if ctor is not None:
            return list(starmap(ctor, rows))
        mapper = cls._ROW_MAPPER
        return [mapper(row) for row in rows]

    async def stream(self, *, batch_size: int = 1000) -> AsyncIterator[RecordT]:
        """Yield hydrated rows in batches to bound memory for large result sets.

        Callers must supply a stable ``ORDER BY``; without one, LIMIT/OFFSET
        paging can duplicate or skip rows as the result set shifts. Deep
        ``OFFSET`` values also force O(offset) scans on most engines.
        """
        if batch_size < 1:
            msg = "batch_size must be >= 1"
            raise ValueError(msg)
        start = self._offset or 0
        remaining = self._limit
        offset = start
        mapper = type(self)._ROW_MAPPER
        while True:
            size = batch_size if remaining is None else min(batch_size, remaining)
            if size <= 0:
                return
            if offset > start and not self._order_by:
                msg = "stream() requires ORDER BY when paging beyond the first batch"
                raise ValueError(msg)
            batch_query = self.limit(size).offset(offset)
            sql, params = batch_query.to_sql()
            rows = await self._adapter.fetch_all(sql, params)
            if not rows:
                return
            for row in rows:
                yield mapper(row)
            if len(rows) < size:
                return
            offset += size
            if remaining is not None:
                remaining -= size

    def _with_fetch_limit(self, maximum: int) -> Self:
        """Cap rows fetched without raising the caller's existing LIMIT."""
        if self._limit is not None and self._limit <= maximum:
            return self
        return self.limit(maximum)

    async def first(self) -> RecordT | None:
        if self._is_static_select():
            sql = f"{type(self)._SELECT_SQL} LIMIT 1"
            params: list[Any] = []
        else:
            sql, params = self._with_fetch_limit(1).to_sql()
        row = await self._adapter.fetch_one(sql, params)
        if row is None:
            return None
        return type(self)._ROW_MAPPER(row)

    async def one(self) -> RecordT:
        return await fetch_one(self._with_fetch_limit(2).all)

    async def count(self) -> int:
        """Return total rows matching WHERE (ignores LIMIT/OFFSET pagination).

        ``group_by`` / ``having`` / ``distinct`` / ``distinct_on`` are not
        applied — call ``count()`` only on ungrouped queries, or project an
        aggregate via ``select(count())`` instead.
        """
        if self._group_by or self._having or self._distinct or self._distinct_on:
            msg = (
                "count() ignores group_by/having/distinct; remove those clauses "
                "or use select(count(...)) for grouped aggregates"
            )
            raise ValueError(msg)
        compiler = SQLCompiler(self._adapter.dialect, initial_sql=self._COUNT_SQL)
        if self._where:
            compiler.write(" WHERE ")
            compiler.visit(and_(*self._where))
        sql, params = compiler.build()
        value = await self._adapter.fetch_val(sql, params)
        return int(value)

    async def exists(self) -> bool:
        """Return whether any matching row exists without hydrating records.

        Preserves the caller's LIMIT/OFFSET pagination (e.g. ``offset(100).exists()``
        is false when fewer than 101 matching rows exist).
        """
        if self._group_by or self._having or self._distinct or self._distinct_on:
            msg = (
                "exists() ignores group_by/having/distinct; remove those clauses "
                "or use select(...) with a subquery"
            )
            raise ValueError(msg)
        if self._limit == 0:
            return False
        compiler = SQLCompiler(
            self._adapter.dialect,
            initial_sql=f"SELECT 1 FROM {type(self)._FROM_SQL}",
        )
        if self._where:
            compiler.write(" WHERE ")
            compiler.visit(and_(*self._where))
        # ``_limit == 0`` returned above, so the probe is always a single row.
        compiler.write(" LIMIT ")
        compiler.write(compiler.add_param(1))
        if self._offset is not None:
            compiler.write(" OFFSET ")
            compiler.write(compiler.add_param(self._offset))
        sql, params = compiler.build()
        return (await self._adapter.fetch_val(sql, params)) is not None


class BaseInsertQuery[RecordT]:
    """Immutable INSERT builder."""

    __slots__ = ("_adapter", "_returning", "_values")

    _TABLE_NAME: str
    _RETURNING_COLS: str
    _ROW_MAPPER: Callable[[Sequence[Any]], RecordT]
    _ATTR_TO_SQL: ClassVar[Mapping[str, str]] = {}
    _UPSERT_CLS: ClassVar[type[UpsertQuery[Any]] | None] = None

    def __init__(
        self,
        adapter: DBAdapter,
        *,
        values: tuple[tuple[str, Any], ...] = (),
        returning: bool = False,
    ) -> None:
        self._adapter = adapter
        self._values = values
        self._returning = returning

    def values(self, **columns: Any) -> Self:
        """Accumulate INSERT columns.

        ``columns`` keys must be bare SQL identifiers (e.g. ``email``). They are
        dialect-quoted once in :meth:`to_sql`. Generated repositories pass Python
        keyword names, never pre-quoted fragments.
        """
        pairs = tuple((name, value) for name, value in columns.items() if not is_unset(value))
        return type(self)(
            self._adapter,
            values=self._values + pairs,
            returning=self._returning,
        )

    def returning(self) -> Self:
        return type(self)(self._adapter, values=self._values, returning=True)

    def on_conflict(self, *columns: Column[Any] | str) -> UpsertQuery[RecordT]:
        """Start a dialect-specific upsert (``ON CONFLICT`` / ``ON DUPLICATE KEY``)."""
        conflict_columns = conflict_column_names(*columns)
        if not conflict_columns and self._adapter.dialect.upsert_style == "conflict":
            msg = "on_conflict() requires at least one conflict target column"
            raise ValueError(msg)

        upsert_cls = type(self)._UPSERT_CLS
        if upsert_cls is None:

            class _Upsert(UpsertQuery[RecordT]):
                __slots__ = ()
                _RETURNING_COLS = type(self)._RETURNING_COLS
                # staticmethod keeps the callable unbound (mypy would otherwise infer a
                # bound method and clash with UpsertQuery._ROW_MAPPER).
                _ROW_MAPPER = staticmethod(type(self)._ROW_MAPPER)
                _ATTR_TO_SQL = type(self)._ATTR_TO_SQL

            upsert_cls = _Upsert

        return upsert_cls(
            self._adapter,
            table_name=type(self)._TABLE_NAME,
            values=self._values,
            conflict_columns=conflict_columns,
            returning=self._returning,
        )

    def to_sql(self) -> tuple[str, list[Any]]:
        compiler = SQLCompiler(self._adapter.dialect)
        self._write_sql(compiler)
        return compiler.build()

    def _write_sql(self, compiler: SQLCompiler) -> None:
        if not self._values:
            msg = "insert() requires at least one column value"
            raise ValueError(msg)
        dialect = compiler.dialect
        cols = ", ".join(dialect.quote(name) for name, _ in self._values)
        placeholders = [compiler.compile_value(value) for _, value in self._values]
        sql = f"INSERT INTO {self._TABLE_NAME} ({cols}) VALUES ({', '.join(placeholders)})"
        if self._returning:
            if not dialect.supports_returning:
                msg = f"{dialect.name} does not support RETURNING"
                raise ValueError(msg)
            sql = f"{sql} RETURNING {self._RETURNING_COLS}"
        compiler.write(sql)

    def prepare(self) -> CompiledQuery[RecordT]:
        """Compile this mutation once into a reusable template."""
        compiler = SQLCompiler(self._adapter.dialect, slots_enabled=True)
        self._write_sql(compiler)
        return _compiled_mutation(
            self._adapter,
            compiler,
            mapper=type(self)._ROW_MAPPER if self._returning else None,
        )

    async def execute(self) -> int:
        sql, params = self.to_sql()
        return await self._adapter.execute(sql, params)

    async def all(self) -> list[RecordT]:
        if not self._returning:
            msg = "all() requires .returning()"
            raise ValueError(msg)
        sql, params = self.to_sql()
        rows = await self._adapter.fetch_all(sql, params)
        mapper = type(self)._ROW_MAPPER
        return [mapper(row) for row in rows]

    async def first(self) -> RecordT | None:
        rows = await self.all()
        return rows[0] if rows else None

    async def one(self) -> RecordT:
        if not self._returning:
            msg = "one() requires .returning()"
            raise ValueError(msg)
        sql, params = self.to_sql()
        rows = await self._adapter.fetch_all(sql, params)
        if not rows:
            raise NotFoundError("insert ... returning produced no row")
        if len(rows) > 1:
            raise MultipleResultsError("expected exactly one row, found multiple")
        # Map only the single accepted row (UPDATE/DELETE cannot be LIMITed).
        return type(self)._ROW_MAPPER(rows[0])


class BaseUpdateQuery[RecordT]:
    """Immutable UPDATE builder with full-table mutation guard."""

    __slots__ = ("_adapter", "_allow_full", "_returning", "_set", "_where")

    _TABLE_NAME: str
    _RETURNING_COLS: str
    _ROW_MAPPER: Callable[[Sequence[Any]], RecordT]

    def __init__(
        self,
        adapter: DBAdapter,
        *,
        set_values: tuple[tuple[str, Any], ...] = (),
        where: tuple[Expr[bool], ...] = (),
        allow_full: bool = False,
        returning: bool = False,
    ) -> None:
        self._adapter = adapter
        self._set = set_values
        self._where = where
        self._allow_full = allow_full
        self._returning = returning

    def _set_values(self, **columns: Any) -> Self:
        """Accumulate SET columns (bare identifiers; quoted once in ``to_sql``)."""
        pairs = tuple((name, value) for name, value in columns.items() if not is_unset(value))
        return type(self)(
            self._adapter,
            set_values=self._set + pairs,
            where=self._where,
            allow_full=self._allow_full,
            returning=self._returning,
        )

    def where(self, *predicates: Expr[bool]) -> Self:
        if not predicates:
            return self
        return type(self)(
            self._adapter,
            set_values=self._set,
            where=self._where + predicates,
            allow_full=self._allow_full,
            returning=self._returning,
        )

    def allow_full_table(self) -> Self:
        return type(self)(
            self._adapter,
            set_values=self._set,
            where=self._where,
            allow_full=True,
            returning=self._returning,
        )

    def returning(self) -> Self:
        return type(self)(
            self._adapter,
            set_values=self._set,
            where=self._where,
            allow_full=self._allow_full,
            returning=True,
        )

    def _write_sql(self, compiler: SQLCompiler) -> None:
        if not self._set:
            msg = "update() requires at least one SET column"
            raise ValueError(msg)
        if not self._where and not self._allow_full:
            raise UnsafeMutationError(
                "UPDATE without WHERE is refused; call allow_full_table() to proceed"
            )
        dialect = compiler.dialect
        assignments = [
            f"{dialect.quote(name)} = {compiler.compile_value(value)}" for name, value in self._set
        ]
        sql = f"UPDATE {self._TABLE_NAME} SET {', '.join(assignments)}"
        if self._where:
            where_sql = compiler.compile_fragment(and_(*self._where))
            sql = f"{sql} WHERE {where_sql}"
        if self._returning:
            if not dialect.supports_returning:
                msg = f"{dialect.name} does not support RETURNING"
                raise ValueError(msg)
            sql = f"{sql} RETURNING {self._RETURNING_COLS}"
        compiler.write(sql)

    def to_sql(self) -> tuple[str, list[Any]]:
        compiler = SQLCompiler(self._adapter.dialect)
        self._write_sql(compiler)
        return compiler.build()

    def prepare(self) -> CompiledQuery[RecordT]:
        """Compile this mutation once into a reusable template."""
        compiler = SQLCompiler(self._adapter.dialect, slots_enabled=True)
        self._write_sql(compiler)
        return _compiled_mutation(
            self._adapter,
            compiler,
            mapper=type(self)._ROW_MAPPER if self._returning else None,
        )

    async def execute(self) -> int:
        sql, params = self.to_sql()
        return await self._adapter.execute(sql, params)

    async def all(self) -> list[RecordT]:
        if not self._returning:
            msg = "all() requires .returning()"
            raise ValueError(msg)
        sql, params = self.to_sql()
        rows = await self._adapter.fetch_all(sql, params)
        mapper = type(self)._ROW_MAPPER
        return [mapper(row) for row in rows]

    async def first(self) -> RecordT | None:
        rows = await self.all()
        return rows[0] if rows else None

    async def one(self) -> RecordT:
        if not self._returning:
            msg = "one() requires .returning()"
            raise ValueError(msg)
        sql, params = self.to_sql()
        rows = await self._adapter.fetch_all(sql, params)
        if not rows:
            raise NotFoundError("update ... returning produced no row")
        if len(rows) > 1:
            raise MultipleResultsError("expected exactly one row, found multiple")
        # Map only the single accepted row (UPDATE/DELETE cannot be LIMITed).
        return type(self)._ROW_MAPPER(rows[0])


class BaseDeleteQuery[RecordT]:
    """Immutable DELETE builder with full-table mutation guard."""

    __slots__ = ("_adapter", "_allow_full", "_returning", "_table_name", "_where")

    _RETURNING_COLS: str = "*"
    _ROW_MAPPER: Callable[[Sequence[Any]], RecordT]

    def __init__(
        self,
        adapter: DBAdapter,
        *,
        table_name: str,
        where: tuple[Expr[bool], ...] = (),
        allow_full: bool = False,
        returning: bool = False,
    ) -> None:
        self._adapter = adapter
        self._table_name = table_name
        self._where = where
        self._allow_full = allow_full
        self._returning = returning

    def where(self, *predicates: Expr[bool]) -> Self:
        if not predicates:
            return self
        return type(self)(
            self._adapter,
            table_name=self._table_name,
            where=self._where + predicates,
            allow_full=self._allow_full,
            returning=self._returning,
        )

    def allow_full_table(self) -> Self:
        return type(self)(
            self._adapter,
            table_name=self._table_name,
            where=self._where,
            allow_full=True,
            returning=self._returning,
        )

    def returning(self) -> Self:
        return type(self)(
            self._adapter,
            table_name=self._table_name,
            where=self._where,
            allow_full=self._allow_full,
            returning=True,
        )

    def _write_sql(self, compiler: SQLCompiler) -> None:
        if not self._where and not self._allow_full:
            raise UnsafeMutationError(
                "DELETE without WHERE is refused; call allow_full_table() to proceed"
            )
        sql = f"DELETE FROM {self._table_name}"
        if self._where:
            where_sql = compiler.compile_fragment(and_(*self._where))
            sql = f"{sql} WHERE {where_sql}"
        if self._returning:
            dialect = compiler.dialect
            if not dialect.supports_returning:
                msg = f"{dialect.name} does not support RETURNING"
                raise ValueError(msg)
            sql = f"{sql} RETURNING {type(self)._RETURNING_COLS}"
        compiler.write(sql)

    def to_sql(self) -> tuple[str, list[Any]]:
        compiler = SQLCompiler(self._adapter.dialect)
        self._write_sql(compiler)
        return compiler.build()

    def prepare(self) -> CompiledQuery[RecordT]:
        """Compile this mutation once into a reusable template."""
        compiler = SQLCompiler(self._adapter.dialect, slots_enabled=True)
        self._write_sql(compiler)
        return _compiled_mutation(
            self._adapter,
            compiler,
            mapper=type(self)._ROW_MAPPER if self._returning else None,
        )

    async def execute(self) -> int:
        sql, params = self.to_sql()
        return await self._adapter.execute(sql, params)

    async def all(self) -> list[RecordT]:
        if not self._returning:
            msg = "all() requires .returning()"
            raise ValueError(msg)
        sql, params = self.to_sql()
        rows = await self._adapter.fetch_all(sql, params)
        mapper = type(self)._ROW_MAPPER
        return [mapper(row) for row in rows]

    async def first(self) -> RecordT | None:
        rows = await self.all()
        return rows[0] if rows else None

    async def one(self) -> RecordT:
        if not self._returning:
            msg = "one() requires .returning()"
            raise ValueError(msg)
        sql, params = self.to_sql()
        rows = await self._adapter.fetch_all(sql, params)
        if not rows:
            raise NotFoundError("delete ... returning produced no row")
        if len(rows) > 1:
            raise MultipleResultsError("expected exactly one row, found multiple")
        # Map only the single accepted row (UPDATE/DELETE cannot be LIMITed).
        return type(self)._ROW_MAPPER(rows[0])


class JoinedSelectQuery[*Ts](_ProjectableSelect):
    """SELECT builder that hydrates each row into ``tuple[*Ts]`` via column slices."""

    __slots__ = (
        "_adapter",
        "_distinct",
        "_distinct_on",
        "_from_sql",
        "_group_by",
        "_having",
        "_joins",
        "_limit",
        "_locking",
        "_offset",
        "_order_by",
        "_segments",
        "_where",
    )

    def __init__(
        self,
        adapter: DBAdapter,
        *,
        from_sql: str,
        segments: tuple[_JoinSegment, ...],
        joins: tuple[_JoinClause, ...],
        where: tuple[Expr[bool], ...] = (),
        order_by: tuple[OrderByClause, ...] = (),
        limit: int | None = None,
        offset: int | None = None,
        group_by: tuple[Expr[Any], ...] = (),
        having: tuple[Expr[bool], ...] = (),
        distinct: bool = False,
        distinct_on: tuple[Expr[Any], ...] = (),
        locking: LockClause | None = None,
    ) -> None:
        self._adapter = adapter
        self._from_sql = from_sql
        self._segments = segments
        self._joins = joins
        self._where = where
        self._order_by = order_by
        self._limit = limit
        self._offset = offset
        self._group_by = group_by
        self._having = having
        self._distinct = distinct
        self._distinct_on = distinct_on
        self._locking = locking

    def _clone(
        self,
        *,
        segments: tuple[_JoinSegment, ...] | None = None,
        joins: tuple[_JoinClause, ...] | None = None,
        where: tuple[Expr[bool], ...] | None = None,
        order_by: tuple[OrderByClause, ...] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        replace_limit: bool = False,
        replace_offset: bool = False,
        group_by: tuple[Expr[Any], ...] | None = None,
        having: tuple[Expr[bool], ...] | None = None,
        distinct: bool | None = None,
        distinct_on: tuple[Expr[Any], ...] | None = None,
        locking: LockClause | object | None = _LOCK_UNSET,
    ) -> JoinedSelectQuery[*Ts]:
        return JoinedSelectQuery(
            self._adapter,
            from_sql=self._from_sql,
            segments=self._segments if segments is None else segments,
            joins=self._joins if joins is None else joins,
            where=self._where if where is None else where,
            order_by=self._order_by if order_by is None else order_by,
            limit=limit if replace_limit else self._limit,
            offset=offset if replace_offset else self._offset,
            group_by=self._group_by if group_by is None else group_by,
            having=self._having if having is None else having,
            distinct=self._distinct if distinct is None else distinct,
            distinct_on=self._distinct_on if distinct_on is None else distinct_on,
            locking=self._locking if locking is _LOCK_UNSET else cast(LockClause | None, locking),
        )

    def where(self, *predicates: Expr[bool]) -> JoinedSelectQuery[*Ts]:
        if not predicates:
            return self
        return self._clone(where=self._where + predicates)

    def order_by(self, *clauses: OrderByClause) -> JoinedSelectQuery[*Ts]:
        if not clauses:
            return self
        return self._clone(order_by=self._order_by + clauses)

    def limit(self, n: int) -> JoinedSelectQuery[*Ts]:
        return self._clone(limit=n, replace_limit=True)

    def offset(self, n: int) -> JoinedSelectQuery[*Ts]:
        return self._clone(offset=n, replace_offset=True)

    def group_by(self, *exprs: Expr[Any]) -> JoinedSelectQuery[*Ts]:
        if not exprs:
            return self
        return self._clone(group_by=self._group_by + exprs)

    def having(self, *predicates: Expr[bool]) -> JoinedSelectQuery[*Ts]:
        if not predicates:
            return self
        return self._clone(having=self._having + predicates)

    def distinct(self, *on: Expr[Any]) -> JoinedSelectQuery[*Ts]:
        if on:
            return self._clone(distinct=True, distinct_on=self._distinct_on + on)
        return self._clone(distinct=True)

    def for_update(
        self,
        *,
        nowait: bool = False,
        skip_locked: bool = False,
        of: Sequence[TableRef[Any]] = (),
    ) -> JoinedSelectQuery[*Ts]:
        return self._clone(
            locking=LockClause(
                mode="UPDATE",
                nowait=nowait,
                skip_locked=skip_locked,
                of=lock_of_targets(of, self._adapter.dialect),
            )
        )

    def for_share(
        self,
        *,
        nowait: bool = False,
        skip_locked: bool = False,
        of: Sequence[TableRef[Any]] = (),
    ) -> JoinedSelectQuery[*Ts]:
        return self._clone(
            locking=LockClause(
                mode="SHARE",
                nowait=nowait,
                skip_locked=skip_locked,
                of=lock_of_targets(of, self._adapter.dialect),
            )
        )

    def _project(self, columns: tuple[Expr[Any], ...]) -> ProjectedSelectQuery[Any]:
        join_parts = tuple(
            (f" {join.kind} JOIN {join.target_sql} ON ", join.on) for join in self._joins
        )
        return ProjectedSelectQuery(
            self._adapter,
            from_sql=self._from_sql,
            columns=columns,
            where=self._where,
            order_by=self._order_by,
            limit=self._limit,
            offset=self._offset,
            group_by=self._group_by,
            having=self._having,
            distinct=self._distinct,
            distinct_on=self._distinct_on,
            locking=self._locking,
            join_parts=join_parts,
        )

    def join[OtherT](
        self,
        table: TableRef[OtherT],
        *on: Expr[bool],
    ) -> JoinedSelectQuery[*Ts, OtherT]:
        return cast(
            JoinedSelectQuery[*Ts, OtherT],
            self._append_join(table, *on, kind="INNER", nullable=False),
        )

    def left_join[OtherT](
        self,
        table: TableRef[OtherT],
        *on: Expr[bool],
    ) -> JoinedSelectQuery[*Ts, OtherT | None]:
        return cast(
            JoinedSelectQuery[*Ts, OtherT | None],
            self._append_join(table, *on, kind="LEFT", nullable=True),
        )

    def _append_join(
        self,
        table: TableRef[Any],
        *on: Expr[bool],
        kind: Literal["INNER", "LEFT"],
        nullable: bool,
    ) -> JoinedSelectQuery[Any]:
        if not on:
            msg = "join() requires at least one ON predicate"
            raise ValueError(msg)
        dialect = self._adapter.dialect
        segment = _JoinSegment(
            select_list=table.select_list(dialect),
            column_count=table.column_count,
            mapper=table.map_row_optional if nullable else table.map_row,
            nullable=nullable,
        )
        join = _JoinClause(
            kind=kind,
            target_sql=table.join_target(dialect),
            on=on,
            segment=segment,
        )
        return JoinedSelectQuery(
            self._adapter,
            from_sql=self._from_sql,
            segments=(*self._segments, segment),
            joins=(*self._joins, join),
            where=self._where,
            order_by=self._order_by,
            limit=self._limit,
            offset=self._offset,
            group_by=self._group_by,
            having=self._having,
            distinct=self._distinct,
            distinct_on=self._distinct_on,
            locking=self._locking,
        )

    def _write_sql(self, compiler: SQLCompiler) -> None:
        dialect = compiler.dialect
        select_lists = ", ".join(segment.select_list for segment in self._segments)
        compiler.write("SELECT ")
        write_distinct(
            compiler,
            dialect,
            distinct=self._distinct,
            distinct_on=self._distinct_on,
        )
        compiler.write(f"{select_lists} FROM {self._from_sql}")
        for join in self._joins:
            compiler.write(f" {join.kind} JOIN {join.target_sql} ON ")
            compiler.visit(and_(*join.on))
        if self._where:
            compiler.write(" WHERE ")
            compiler.visit(and_(*self._where))
        write_group_having(compiler, group_by=self._group_by, having=self._having)
        write_order_limit_offset(
            compiler,
            order_by=self._order_by,
            limit=self._limit,
            offset=self._offset,
        )
        write_locking(compiler, dialect, self._locking)

    def to_sql(self) -> tuple[str, list[Any]]:
        compiler = SQLCompiler(self._adapter.dialect)
        self._write_sql(compiler)
        return compiler.build()

    def prepare(self) -> CompiledQuery[tuple[*Ts]]:
        """Compile this join query once into a reusable template."""
        compiler = SQLCompiler(self._adapter.dialect, slots_enabled=True)
        self._write_sql(compiler)
        sql, params = compiler.build()
        return CompiledQuery(
            self._adapter,
            sql=sql,
            params=params,
            slot_occurrences=compiler.slot_occurrences,
            row_mapper=self._hydrate,
            rows_mapper=rows_mapper_from_record(self._hydrate),
        )

    def as_subquery(self) -> SubqueryExpr[Any]:
        sql, params = self.to_sql()
        return SubqueryExpr(sql, params)

    def _hydrate(self, row: Sequence[Any]) -> tuple[*Ts]:
        segments = self._segments
        if len(segments) == 2:
            left, right = segments
            n0 = left.column_count
            return cast(tuple[*Ts], (left.mapper(row[:n0]), right.mapper(row[n0:])))
        if len(segments) == 3:
            a, b, c = segments
            n0 = a.column_count
            n1 = n0 + b.column_count
            return cast(
                tuple[*Ts],
                (a.mapper(row[:n0]), b.mapper(row[n0:n1]), c.mapper(row[n1:])),
            )
        values: list[Any] = []
        offset = 0
        for segment in segments:
            end = offset + segment.column_count
            values.append(segment.mapper(row[offset:end]))
            offset = end
        return cast(tuple[*Ts], tuple(values))

    async def all(self) -> list[tuple[*Ts]]:
        sql, params = self.to_sql()
        rows = await self._adapter.fetch_all(sql, params)
        hydrate = self._hydrate
        return [hydrate(row) for row in rows]

    async def stream(self, *, batch_size: int = 1000) -> AsyncIterator[tuple[*Ts]]:
        """Yield joined rows in batches; requires stable ORDER BY beyond batch 1."""
        if batch_size < 1:
            msg = "batch_size must be >= 1"
            raise ValueError(msg)
        start = self._offset or 0
        remaining = self._limit
        offset = start
        while True:
            size = batch_size if remaining is None else min(batch_size, remaining)
            if size <= 0:
                return
            if offset > start and not self._order_by:
                msg = "stream() requires ORDER BY when paging beyond the first batch"
                raise ValueError(msg)
            sql, params = self.limit(size).offset(offset).to_sql()
            rows = await self._adapter.fetch_all(sql, params)
            if not rows:
                return
            for row in rows:
                yield self._hydrate(row)
            if len(rows) < size:
                return
            offset += size
            if remaining is not None:
                remaining -= size

    def _with_fetch_limit(self, maximum: int) -> JoinedSelectQuery[*Ts]:
        if self._limit is not None and self._limit <= maximum:
            return self
        return self.limit(maximum)

    async def first(self) -> tuple[*Ts] | None:
        sql, params = self._with_fetch_limit(1).to_sql()
        row = await self._adapter.fetch_one(sql, params)
        if row is None:
            return None
        return self._hydrate(row)

    async def one(self) -> tuple[*Ts]:
        return await fetch_one(self._with_fetch_limit(2).all)


@dataclass(slots=True, frozen=True)
class _RelationSpec[RelatedT]:
    """One batch-loaded relation (has-many or belongs-to)."""

    related: TableRef[RelatedT]
    fk: Column[Any]
    key: Column[Any]
    fk_attr: str
    key_attr: str
    related_where: tuple[Expr[bool], ...] = ()
    related_order_by: tuple[OrderByClause, ...] = ()
    cardinality: Literal["many", "one"] = "many"
    nullable: bool = False


class PreloadSelectQuery[ParentT, ChildT]:
    """Two-query batch eager loader: parent SELECT + ``WHERE fk IN (...)`` (has-many).

    ``limit`` / ``offset`` / parent ``order_by`` / ``where`` apply to the *parent*
    query. Related-row ``order_by`` passed to :meth:`BaseSelectQuery.preload`
    sorts the follow-up ``IN`` query; that preserves each parent's relative
    child order without providing per-parent top-N limits.
    """

    __slots__ = ("_parent", "_spec")

    def __init__(
        self,
        parent: BaseSelectQuery[ParentT],
        *,
        related: TableRef[ChildT],
        fk: Column[Any],
        key: Column[Any],
        fk_attr: str,
        key_attr: str,
        related_where: tuple[Expr[bool], ...] = (),
        related_order_by: tuple[OrderByClause, ...] = (),
    ) -> None:
        self._parent = parent
        self._spec = _RelationSpec(
            related=related,
            fk=fk,
            key=key,
            fk_attr=fk_attr,
            key_attr=key_attr,
            related_where=related_where,
            related_order_by=related_order_by,
            cardinality="many",
        )

    def _clone(self, parent: BaseSelectQuery[ParentT]) -> PreloadSelectQuery[ParentT, ChildT]:
        spec = self._spec
        return PreloadSelectQuery(
            parent,
            related=spec.related,
            fk=spec.fk,
            key=spec.key,
            fk_attr=spec.fk_attr,
            key_attr=spec.key_attr,
            related_where=spec.related_where,
            related_order_by=spec.related_order_by,
        )

    def where(self, *predicates: Expr[bool]) -> PreloadSelectQuery[ParentT, ChildT]:
        return self._clone(self._parent.where(*predicates))

    def order_by(self, *clauses: OrderByClause) -> PreloadSelectQuery[ParentT, ChildT]:
        return self._clone(self._parent.order_by(*clauses))

    def limit(self, n: int) -> PreloadSelectQuery[ParentT, ChildT]:
        return self._clone(self._parent.limit(n))

    def offset(self, n: int) -> PreloadSelectQuery[ParentT, ChildT]:
        return self._clone(self._parent.offset(n))

    def preload[OtherT](
        self,
        related: TableRef[OtherT],
        *predicates: Expr[bool],
        fk: Column[Any],
        key: Column[Any],
        fk_attr: str,
        key_attr: str,
        order_by: Sequence[OrderByClause] = (),
    ) -> MultiPreloadSelectQuery[ParentT, list[ChildT], list[OtherT]]:
        """Chain a second has-many relation onto this preload."""
        return MultiPreloadSelectQuery(
            self._parent,
            (
                self._spec,
                _RelationSpec(
                    related=related,
                    fk=fk,
                    key=key,
                    fk_attr=fk_attr,
                    key_attr=key_attr,
                    related_where=predicates,
                    related_order_by=tuple(order_by),
                    cardinality="many",
                ),
            ),
        )

    async def all(self) -> list[tuple[ParentT, list[ChildT]]]:
        parents = await self._parent.all()
        if not parents:
            return []
        grouped = await _load_many(self._parent.adapter, parents, self._spec)
        key_attr = self._spec.key_attr
        return [(parent, list(grouped.get(getattr(parent, key_attr), []))) for parent in parents]

    async def first(self) -> tuple[ParentT, list[ChildT]] | None:
        rows = await self.limit(1).all()
        return rows[0] if rows else None

    async def one(self) -> tuple[ParentT, list[ChildT]]:
        rows = await self.limit(2).all()
        if not rows:
            raise NotFoundError("expected exactly one row, found 0")
        if len(rows) > 1:
            raise MultipleResultsError("expected exactly one row, found multiple")
        return rows[0]

    async def count(self) -> int:
        return await self._parent.count()

    async def exists(self) -> bool:
        return await self._parent.limit(1).first() is not None


class PreloadOneSelectQuery[ChildT, ParentT]:
    """Belongs-to batch loader: child SELECT + ``WHERE parent.pk IN (...)``."""

    __slots__ = ("_parent", "_spec")

    def __init__(
        self,
        parent: BaseSelectQuery[ChildT],
        *,
        related: TableRef[ParentT],
        fk: Column[Any],
        key: Column[Any],
        fk_attr: str,
        key_attr: str,
        related_where: tuple[Expr[bool], ...] = (),
        related_order_by: tuple[OrderByClause, ...] = (),
        nullable: bool = False,
    ) -> None:
        self._parent = parent
        self._spec = _RelationSpec(
            related=related,
            fk=fk,
            key=key,
            fk_attr=fk_attr,
            key_attr=key_attr,
            related_where=related_where,
            related_order_by=related_order_by,
            cardinality="one",
            nullable=nullable,
        )

    def _clone(self, parent: BaseSelectQuery[ChildT]) -> PreloadOneSelectQuery[ChildT, ParentT]:
        spec = self._spec
        return PreloadOneSelectQuery(
            parent,
            related=spec.related,
            fk=spec.fk,
            key=spec.key,
            fk_attr=spec.fk_attr,
            key_attr=spec.key_attr,
            related_where=spec.related_where,
            related_order_by=spec.related_order_by,
            nullable=spec.nullable,
        )

    def where(self, *predicates: Expr[bool]) -> PreloadOneSelectQuery[ChildT, ParentT]:
        return self._clone(self._parent.where(*predicates))

    def order_by(self, *clauses: OrderByClause) -> PreloadOneSelectQuery[ChildT, ParentT]:
        return self._clone(self._parent.order_by(*clauses))

    def limit(self, n: int) -> PreloadOneSelectQuery[ChildT, ParentT]:
        return self._clone(self._parent.limit(n))

    def offset(self, n: int) -> PreloadOneSelectQuery[ChildT, ParentT]:
        return self._clone(self._parent.offset(n))

    async def all(self) -> list[tuple[ChildT, ParentT]]:
        children = await self._parent.all()
        if not children:
            return []
        by_key = await _load_one(self._parent.adapter, children, self._spec)
        fk_attr = self._spec.fk_attr
        results: list[tuple[ChildT, ParentT]] = []
        for child in children:
            fk_value = getattr(child, fk_attr)
            if fk_value is None:
                if not self._spec.nullable:
                    msg = f"belongs-to foreign key {fk_attr!r} is NULL but relation is non-nullable"
                    raise NotFoundError(msg)
                results.append((child, cast(ParentT, None)))
                continue
            related = by_key.get(fk_value)
            if related is None:
                # Predicates force ``nullable`` at construction time so filtered
                # misses hydrate as ``None`` with a matching ``ParentT | None`` type.
                # Unfiltered non-nullable misses remain NotFoundError (dangling FK).
                if self._spec.nullable:
                    results.append((child, cast(ParentT, None)))
                else:
                    msg = f"missing related row for non-nullable belongs-to {fk_attr}={fk_value!r}"
                    raise NotFoundError(msg)
            else:
                results.append((child, related))
        return results

    async def first(self) -> tuple[ChildT, ParentT] | None:
        rows = await self.limit(1).all()
        return rows[0] if rows else None

    async def one(self) -> tuple[ChildT, ParentT]:
        rows = await self.limit(2).all()
        if not rows:
            raise NotFoundError("expected exactly one row, found 0")
        if len(rows) > 1:
            raise MultipleResultsError("expected exactly one row, found multiple")
        return rows[0]

    async def count(self) -> int:
        return await self._parent.count()

    async def exists(self) -> bool:
        return await self._parent.limit(1).first() is not None


class MultiPreloadSelectQuery[ParentT, *Ts]:
    """Parent query with two or more batch-loaded has-many relations."""

    __slots__ = ("_parent", "_specs")

    def __init__(
        self,
        parent: BaseSelectQuery[ParentT],
        specs: tuple[_RelationSpec[Any], ...],
    ) -> None:
        self._parent = parent
        self._specs = specs

    def _clone(self, parent: BaseSelectQuery[ParentT]) -> MultiPreloadSelectQuery[ParentT, *Ts]:
        return MultiPreloadSelectQuery(parent, self._specs)

    def where(self, *predicates: Expr[bool]) -> MultiPreloadSelectQuery[ParentT, *Ts]:
        return self._clone(self._parent.where(*predicates))

    def order_by(self, *clauses: OrderByClause) -> MultiPreloadSelectQuery[ParentT, *Ts]:
        return self._clone(self._parent.order_by(*clauses))

    def limit(self, n: int) -> MultiPreloadSelectQuery[ParentT, *Ts]:
        return self._clone(self._parent.limit(n))

    def offset(self, n: int) -> MultiPreloadSelectQuery[ParentT, *Ts]:
        return self._clone(self._parent.offset(n))

    def preload[OtherT](
        self,
        related: TableRef[OtherT],
        *predicates: Expr[bool],
        fk: Column[Any],
        key: Column[Any],
        fk_attr: str,
        key_attr: str,
        order_by: Sequence[OrderByClause] = (),
    ) -> MultiPreloadSelectQuery[ParentT, *Ts, list[OtherT]]:
        return MultiPreloadSelectQuery(
            self._parent,
            (
                *self._specs,
                _RelationSpec(
                    related=related,
                    fk=fk,
                    key=key,
                    fk_attr=fk_attr,
                    key_attr=key_attr,
                    related_where=predicates,
                    related_order_by=tuple(order_by),
                    cardinality="many",
                ),
            ),
        )

    async def all(self) -> list[tuple[ParentT, *Ts]]:
        parents = await self._parent.all()
        if not parents:
            return []
        # Load relations sequentially: the same adapter may be pinned to a
        # transaction connection, so concurrent gathers on it are unsafe.
        adapter = self._parent.adapter
        grouped_list = [await _load_many(adapter, parents, spec) for spec in self._specs]
        rows: list[tuple[ParentT, *Ts]] = []
        for parent in parents:
            parts: list[Any] = [parent]
            for spec, grouped in zip(self._specs, grouped_list, strict=True):
                parts.append(list(grouped.get(getattr(parent, spec.key_attr), [])))
            rows.append(cast(tuple[ParentT, *Ts], tuple(parts)))
        return rows

    async def first(self) -> tuple[ParentT, *Ts] | None:
        rows = await self.limit(1).all()
        return rows[0] if rows else None

    async def one(self) -> tuple[ParentT, *Ts]:
        rows = await self.limit(2).all()
        if not rows:
            raise NotFoundError("expected exactly one row, found 0")
        if len(rows) > 1:
            raise MultipleResultsError("expected exactly one row, found multiple")
        return rows[0]

    async def count(self) -> int:
        return await self._parent.count()

    async def exists(self) -> bool:
        return await self._parent.limit(1).first() is not None


async def _load_many[ParentT, RelatedT](
    adapter: DBAdapter,
    parents: Sequence[ParentT],
    spec: _RelationSpec[RelatedT],
) -> dict[Any, list[RelatedT]]:
    keys = [getattr(parent, spec.key_attr) for parent in parents]
    unique_keys = list(dict.fromkeys(key for key in keys if key is not None))
    grouped: dict[Any, list[RelatedT]] = {key: [] for key in unique_keys}
    if not unique_keys:
        return grouped
    related_query = _table_select(spec.related, adapter)
    if spec.related_where:
        related_query = related_query.where(*spec.related_where)
    if spec.related_order_by:
        related_query = related_query.order_by(*spec.related_order_by)
    children: list[RelatedT] = []
    for start in range(0, len(unique_keys), _IN_BATCH_SIZE):
        batch = unique_keys[start : start + _IN_BATCH_SIZE]
        children.extend(await related_query.where(spec.fk.in_(batch)).all())
    for child in children:
        grouped.setdefault(getattr(child, spec.fk_attr), []).append(child)
    return grouped


async def _load_one[ChildT, ParentT](
    adapter: DBAdapter,
    children: Sequence[ChildT],
    spec: _RelationSpec[ParentT],
) -> dict[Any, ParentT]:
    keys = [getattr(child, spec.fk_attr) for child in children]
    unique_keys = list(dict.fromkeys(key for key in keys if key is not None))
    if not unique_keys:
        return {}
    related_query = _table_select(spec.related, adapter)
    if spec.related_where:
        related_query = related_query.where(*spec.related_where)
    if spec.related_order_by:
        related_query = related_query.order_by(*spec.related_order_by)
    parents: list[ParentT] = []
    for start in range(0, len(unique_keys), _IN_BATCH_SIZE):
        batch = unique_keys[start : start + _IN_BATCH_SIZE]
        parents.extend(await related_query.where(spec.key.in_(batch)).all())
    return {getattr(parent, spec.key_attr): parent for parent in parents}


# Bounded LRU: alias comes from ``table.as_("...")`` and may be user-generated, so an
# unbounded dict could grow without limit. Eviction only costs one class rebuild.
_ADHOC_SELECT_CACHE_MAX = 256
_ADHOC_SELECT_CACHE: OrderedDict[tuple[type[Any], str, str | None], type[BaseSelectQuery[Any]]] = (
    OrderedDict()
)


def _table_select[RecordT](
    table: TableRef[RecordT],
    adapter: DBAdapter,
) -> BaseSelectQuery[RecordT]:
    """Build an ad-hoc ``BaseSelectQuery`` for a ``TableRef`` (preload follow-up)."""
    dialect = adapter.dialect
    alias = getattr(table, "_alias", None)
    cache_key = (type(table), dialect.name, alias)
    query_cls = _ADHOC_SELECT_CACHE.get(cache_key)
    if query_cls is not None:
        _ADHOC_SELECT_CACHE.move_to_end(cache_key)
    else:
        select_cols = table.select_list(dialect)
        from_sql = table.join_target(dialect)
        count_sql = f"SELECT COUNT(*) FROM {from_sql}"
        row_mapper = table.get_row_mapper()
        column_count = table.column_count

        class _AdHocQuery(BaseSelectQuery[RecordT]):
            __slots__ = ()
            _TABLE_NAME = from_sql
            _SELECT_COLS = select_cols
            _FROM_SQL = from_sql
            _SELECT_SQL = f"SELECT {select_cols} FROM {from_sql}"
            _COUNT_SQL = count_sql
            _COLUMN_COUNT = column_count
            _ROW_MAPPER = row_mapper

        query_cls = _AdHocQuery
        _ADHOC_SELECT_CACHE[cache_key] = query_cls
        if len(_ADHOC_SELECT_CACHE) > _ADHOC_SELECT_CACHE_MAX:
            _ADHOC_SELECT_CACHE.popitem(last=False)
    return cast(BaseSelectQuery[RecordT], query_cls(adapter))
