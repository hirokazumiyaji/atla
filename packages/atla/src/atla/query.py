"""Immutable query / mutation builders shared by generated repositories."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Self, cast

from atla.adapter.base import DBAdapter
from atla.compiler import SQLCompiler
from atla.errors import MultipleResultsError, NotFoundError, UnsafeMutationError
from atla.expressions import Column, Expr, OrderByClause, and_
from atla.table import TableRef, sql_attr_name
from atla.types import UNSET, UnsetType


def _is_unset(value: object) -> bool:
    return value is UNSET or isinstance(value, UnsetType)


def _compile_value(compiler: SQLCompiler, value: Any) -> str:
    if isinstance(value, Expr):
        start = len(compiler.buf)
        compiler.visit(value)  # pyright: ignore[reportUnknownArgumentType]
        fragment = "".join(compiler.buf[start:])
        del compiler.buf[start:]
        return fragment
    return compiler.add_param(value)


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


class BaseSelectQuery[RecordT]:
    """Immutable SELECT query builder backed by AOT SQL fragments."""

    __slots__ = ("_adapter", "_limit", "_offset", "_order_by", "_where")

    _TABLE_NAME: str
    _SELECT_SQL: str
    _COUNT_SQL: str
    _ROW_MAPPER: Callable[[Sequence[Any]], RecordT]
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
    ) -> None:
        self._adapter = adapter
        self._where = where
        self._order_by = order_by
        self._limit = limit
        self._offset = offset

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
    ) -> Self:
        return type(self)(
            self._adapter,
            where=self._where if where is None else where,
            order_by=self._order_by if order_by is None else order_by,
            limit=limit if replace_limit else self._limit,
            offset=offset if replace_offset else self._offset,
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
        )

    def preload[OtherT](
        self,
        related: TableRef[OtherT],
        *,
        fk: Column[Any],
        key: Column[Any],
    ) -> PreloadSelectQuery[RecordT, OtherT]:
        """Batch-load ``related`` rows with one ``WHERE fk IN (...)`` follow-up query."""
        return PreloadSelectQuery(
            self,
            related=related,
            fk=fk,
            key=key,
        )

    def with_[OtherT](
        self,
        related: TableRef[OtherT],
        *,
        fk: Column[Any],
        key: Column[Any],
    ) -> PreloadSelectQuery[RecordT, OtherT]:
        """Alias for :meth:`preload` (batch eager loading without N+1 queries)."""
        return self.preload(related, fk=fk, key=key)

    def to_sql(self) -> tuple[str, list[Any]]:
        compiler = SQLCompiler(self._adapter.dialect, initial_sql=self._SELECT_SQL)
        if self._where:
            compiler.write(" WHERE ")
            compiler.visit(and_(*self._where))
        if self._order_by:
            compiler.write(" ORDER BY ")
            for index, clause in enumerate(self._order_by):
                if index:
                    compiler.write(", ")
                compiler.visit_order_by(clause)
        if self._limit is not None:
            compiler.write(" LIMIT ")
            compiler.write(compiler.add_param(self._limit))
        if self._offset is not None:
            compiler.write(" OFFSET ")
            compiler.write(compiler.add_param(self._offset))
        return compiler.build()

    async def all(self) -> list[RecordT]:
        sql, params = self.to_sql()
        rows = await self._adapter.fetch_all(sql, params)
        mapper = type(self)._ROW_MAPPER
        return [mapper(row) for row in rows]

    def _with_fetch_limit(self, maximum: int) -> Self:
        """Cap rows fetched without raising the caller's existing LIMIT."""
        if self._limit is not None and self._limit <= maximum:
            return self
        return self.limit(maximum)

    async def first(self) -> RecordT | None:
        rows = await self._with_fetch_limit(1).all()
        return rows[0] if rows else None

    async def one(self) -> RecordT:
        rows = await self._with_fetch_limit(2).all()
        if not rows:
            raise NotFoundError("expected exactly one row, found 0")
        if len(rows) > 1:
            raise MultipleResultsError("expected exactly one row, found multiple")
        return rows[0]

    async def count(self) -> int:
        """Return total rows matching WHERE (ignores LIMIT/OFFSET pagination)."""
        compiler = SQLCompiler(self._adapter.dialect, initial_sql=self._COUNT_SQL)
        if self._where:
            compiler.write(" WHERE ")
            compiler.visit(and_(*self._where))
        sql, params = compiler.build()
        value = await self._adapter.fetch_val(sql, params)
        return int(value)

    async def exists(self) -> bool:
        return (await self.first()) is not None


class BaseInsertQuery[RecordT]:
    """Immutable INSERT builder."""

    __slots__ = ("_adapter", "_returning", "_values")

    _TABLE_NAME: str
    _RETURNING_COLS: str
    _ROW_MAPPER: Callable[[Sequence[Any]], RecordT]

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
        pairs = tuple((name, value) for name, value in columns.items() if not _is_unset(value))
        return type(self)(
            self._adapter,
            values=self._values + pairs,
            returning=self._returning,
        )

    def returning(self) -> Self:
        return type(self)(self._adapter, values=self._values, returning=True)

    def to_sql(self) -> tuple[str, list[Any]]:
        if not self._values:
            msg = "insert() requires at least one column value"
            raise ValueError(msg)
        dialect = self._adapter.dialect
        cols = ", ".join(dialect.quote(name) for name, _ in self._values)
        compiler = SQLCompiler(dialect)
        placeholders = [_compile_value(compiler, value) for _, value in self._values]
        sql = f"INSERT INTO {self._TABLE_NAME} ({cols}) VALUES ({', '.join(placeholders)})"
        if self._returning:
            if not dialect.supports_returning:
                msg = f"{dialect.name} does not support RETURNING"
                raise ValueError(msg)
            sql = f"{sql} RETURNING {self._RETURNING_COLS}"
        return sql, compiler.params

    async def execute(self) -> int:
        sql, params = self.to_sql()
        return await self._adapter.execute(sql, params)

    async def one(self) -> RecordT:
        if not self._returning:
            msg = "one() requires .returning()"
            raise ValueError(msg)
        sql, params = self.to_sql()
        row = await self._adapter.fetch_one(sql, params)
        if row is None:
            raise NotFoundError("insert ... returning produced no row")
        return type(self)._ROW_MAPPER(row)

    async def all(self) -> list[RecordT]:
        if not self._returning:
            msg = "all() requires .returning()"
            raise ValueError(msg)
        sql, params = self.to_sql()
        rows = await self._adapter.fetch_all(sql, params)
        mapper = type(self)._ROW_MAPPER
        return [mapper(row) for row in rows]


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
        pairs = tuple((name, value) for name, value in columns.items() if not _is_unset(value))
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

    def to_sql(self) -> tuple[str, list[Any]]:
        if not self._set:
            msg = "update() requires at least one SET column"
            raise ValueError(msg)
        if not self._where and not self._allow_full:
            raise UnsafeMutationError(
                "UPDATE without WHERE is refused; call allow_full_table() to proceed"
            )
        dialect = self._adapter.dialect
        compiler = SQLCompiler(dialect)
        assignments = [
            f"{dialect.quote(name)} = {_compile_value(compiler, value)}"
            for name, value in self._set
        ]
        sql = f"UPDATE {self._TABLE_NAME} SET {', '.join(assignments)}"
        if self._where:
            where_start = len(compiler.buf)
            compiler.visit(and_(*self._where))
            where_sql = "".join(compiler.buf[where_start:])
            del compiler.buf[where_start:]
            sql = f"{sql} WHERE {where_sql}"
        if self._returning:
            if not dialect.supports_returning:
                msg = f"{dialect.name} does not support RETURNING"
                raise ValueError(msg)
            sql = f"{sql} RETURNING {self._RETURNING_COLS}"
        return sql, compiler.params

    async def execute(self) -> int:
        sql, params = self.to_sql()
        return await self._adapter.execute(sql, params)

    async def one(self) -> RecordT:
        if not self._returning:
            msg = "one() requires .returning()"
            raise ValueError(msg)
        sql, params = self.to_sql()
        row = await self._adapter.fetch_one(sql, params)
        if row is None:
            raise NotFoundError("update ... returning produced no row")
        return type(self)._ROW_MAPPER(row)


class BaseDeleteQuery:
    """Immutable DELETE builder with full-table mutation guard."""

    __slots__ = ("_adapter", "_allow_full", "_table_name", "_where")

    def __init__(
        self,
        adapter: DBAdapter,
        *,
        table_name: str,
        where: tuple[Expr[bool], ...] = (),
        allow_full: bool = False,
    ) -> None:
        self._adapter = adapter
        self._table_name = table_name
        self._where = where
        self._allow_full = allow_full

    def where(self, *predicates: Expr[bool]) -> BaseDeleteQuery:
        if not predicates:
            return self
        return BaseDeleteQuery(
            self._adapter,
            table_name=self._table_name,
            where=self._where + predicates,
            allow_full=self._allow_full,
        )

    def allow_full_table(self) -> BaseDeleteQuery:
        return BaseDeleteQuery(
            self._adapter,
            table_name=self._table_name,
            where=self._where,
            allow_full=True,
        )

    def to_sql(self) -> tuple[str, list[Any]]:
        if not self._where and not self._allow_full:
            raise UnsafeMutationError(
                "DELETE without WHERE is refused; call allow_full_table() to proceed"
            )
        sql = f"DELETE FROM {self._table_name}"
        if not self._where:
            return sql, []
        compiler = SQLCompiler(self._adapter.dialect)
        compiler.visit(and_(*self._where))
        where_sql, params = compiler.build()
        return f"{sql} WHERE {where_sql}", params

    async def execute(self) -> int:
        sql, params = self.to_sql()
        return await self._adapter.execute(sql, params)


class JoinedSelectQuery[*Ts]:
    """SELECT builder that hydrates each row into ``tuple[*Ts]`` via column slices."""

    __slots__ = (
        "_adapter",
        "_from_sql",
        "_joins",
        "_limit",
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
    ) -> None:
        self._adapter = adapter
        self._from_sql = from_sql
        self._segments = segments
        self._joins = joins
        self._where = where
        self._order_by = order_by
        self._limit = limit
        self._offset = offset

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
        )

    def to_sql(self) -> tuple[str, list[Any]]:
        select_lists = ", ".join(segment.select_list for segment in self._segments)
        compiler = SQLCompiler(
            self._adapter.dialect,
            initial_sql=f"SELECT {select_lists} FROM {self._from_sql}",
        )
        for join in self._joins:
            compiler.write(f" {join.kind} JOIN {join.target_sql} ON ")
            compiler.visit(and_(*join.on))
        if self._where:
            compiler.write(" WHERE ")
            compiler.visit(and_(*self._where))
        if self._order_by:
            compiler.write(" ORDER BY ")
            for index, clause in enumerate(self._order_by):
                if index:
                    compiler.write(", ")
                compiler.visit_order_by(clause)
        if self._limit is not None:
            compiler.write(" LIMIT ")
            compiler.write(compiler.add_param(self._limit))
        if self._offset is not None:
            compiler.write(" OFFSET ")
            compiler.write(compiler.add_param(self._offset))
        return compiler.build()

    def _hydrate(self, row: Sequence[Any]) -> tuple[*Ts]:
        values: list[Any] = []
        offset = 0
        for segment in self._segments:
            end = offset + segment.column_count
            slice_row = row[offset:end]
            offset = end
            values.append(segment.mapper(slice_row))
        return cast(tuple[*Ts], tuple(values))

    async def all(self) -> list[tuple[*Ts]]:
        sql, params = self.to_sql()
        rows = await self._adapter.fetch_all(sql, params)
        return [self._hydrate(row) for row in rows]

    def _with_fetch_limit(self, maximum: int) -> JoinedSelectQuery[*Ts]:
        if self._limit is not None and self._limit <= maximum:
            return self
        return self.limit(maximum)

    async def first(self) -> tuple[*Ts] | None:
        rows = await self._with_fetch_limit(1).all()
        return rows[0] if rows else None

    async def one(self) -> tuple[*Ts]:
        rows = await self._with_fetch_limit(2).all()
        if not rows:
            raise NotFoundError("expected exactly one row, found 0")
        if len(rows) > 1:
            raise MultipleResultsError("expected exactly one row, found multiple")
        return rows[0]


class PreloadSelectQuery[ParentT, ChildT]:
    """Two-query batch eager loader: parent SELECT + ``WHERE fk IN (...)``."""

    __slots__ = ("_fk", "_key", "_parent", "_related")

    def __init__(
        self,
        parent: BaseSelectQuery[ParentT],
        *,
        related: TableRef[ChildT],
        fk: Column[Any],
        key: Column[Any],
    ) -> None:
        self._parent = parent
        self._related = related
        self._fk = fk
        self._key = key

    def where(self, *predicates: Expr[bool]) -> PreloadSelectQuery[ParentT, ChildT]:
        return PreloadSelectQuery(
            self._parent.where(*predicates),
            related=self._related,
            fk=self._fk,
            key=self._key,
        )

    def order_by(self, *clauses: OrderByClause) -> PreloadSelectQuery[ParentT, ChildT]:
        return PreloadSelectQuery(
            self._parent.order_by(*clauses),
            related=self._related,
            fk=self._fk,
            key=self._key,
        )

    def limit(self, n: int) -> PreloadSelectQuery[ParentT, ChildT]:
        return PreloadSelectQuery(
            self._parent.limit(n),
            related=self._related,
            fk=self._fk,
            key=self._key,
        )

    def offset(self, n: int) -> PreloadSelectQuery[ParentT, ChildT]:
        return PreloadSelectQuery(
            self._parent.offset(n),
            related=self._related,
            fk=self._fk,
            key=self._key,
        )

    async def all(self) -> list[tuple[ParentT, list[ChildT]]]:
        parents = await self._parent.all()
        if not parents:
            return []
        key_attr = sql_attr_name(self._key.name)
        fk_attr = sql_attr_name(self._fk.name)
        keys = [getattr(parent, key_attr) for parent in parents]
        children = (
            await _table_select(self._related, self._parent.adapter).where(self._fk.in_(keys)).all()
        )
        grouped: dict[Any, list[ChildT]] = {key: [] for key in keys}
        for child in children:
            grouped.setdefault(getattr(child, fk_attr), []).append(child)
        return [(parent, list(grouped.get(getattr(parent, key_attr), []))) for parent in parents]

    async def first(self) -> tuple[ParentT, list[ChildT]] | None:
        rows = await self.limit(1).all()
        return rows[0] if rows else None


def _table_select[RecordT](
    table: TableRef[RecordT],
    adapter: DBAdapter,
) -> BaseSelectQuery[RecordT]:
    """Build an ad-hoc ``BaseSelectQuery`` for a ``TableRef`` (preload follow-up)."""
    dialect = adapter.dialect
    select_cols = table.select_list(dialect)
    from_sql = table.join_target(dialect)
    count_sql = f"SELECT COUNT(*) FROM {from_sql}"

    class _AdHocQuery(BaseSelectQuery[RecordT]):
        __slots__ = ()
        _TABLE_NAME = from_sql
        _SELECT_COLS = select_cols
        _FROM_SQL = from_sql
        _SELECT_SQL = f"SELECT {select_cols} FROM {from_sql}"
        _COUNT_SQL = count_sql
        _COLUMN_COUNT = table.column_count
        _ROW_MAPPER = table.get_row_mapper()

    return _AdHocQuery(adapter)
