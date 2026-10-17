"""Bulk insert, upsert, and projected SELECT helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Self, cast

from atla.adapter.base import DBAdapter
from atla.compiler import SQLCompiler
from atla.errors import MultipleResultsError, NotFoundError
from atla.expressions import Column, Expr, OrderByClause, and_


def _compile_value(compiler: SQLCompiler, value: Any) -> str:
    if isinstance(value, Expr):
        start = len(compiler.buf)
        compiler.visit(value)  # pyright: ignore[reportUnknownArgumentType]
        fragment = "".join(compiler.buf[start:])
        del compiler.buf[start:]
        return fragment
    return compiler.add_param(value)


class BaseInsertManyQuery[RecordT]:
    """Multi-row INSERT builder (``VALUES (...), (...), ...``)."""

    __slots__ = ("_adapter", "_returning", "_rows", "_table_name")

    _RETURNING_COLS: str
    _ROW_MAPPER: Callable[[Sequence[Any]], RecordT]

    def __init__(
        self,
        adapter: DBAdapter,
        *,
        table_name: str,
        rows: tuple[tuple[tuple[str, Any], ...], ...] = (),
        returning: bool = False,
    ) -> None:
        self._adapter = adapter
        self._table_name = table_name
        self._rows = rows
        self._returning = returning

    def returning(self) -> Self:
        return type(self)(
            self._adapter,
            table_name=self._table_name,
            rows=self._rows,
            returning=True,
        )

    def to_sql(self) -> tuple[str, list[Any]]:
        if not self._rows:
            msg = "insert_many() requires at least one row"
            raise ValueError(msg)
        columns = [name for name, _ in self._rows[0]]
        if not columns:
            msg = "insert_many() rows must include at least one column"
            raise ValueError(msg)
        for row in self._rows[1:]:
            row_cols = [name for name, _ in row]
            if row_cols != columns:
                msg = "insert_many() requires every row to use the same column set"
                raise ValueError(msg)
        dialect = self._adapter.dialect
        cols_sql = ", ".join(dialect.quote(name) for name in columns)
        compiler = SQLCompiler(dialect)
        value_groups: list[str] = []
        for row in self._rows:
            placeholders = [_compile_value(compiler, value) for _, value in row]
            value_groups.append(f"({', '.join(placeholders)})")
        sql = f"INSERT INTO {self._table_name} ({cols_sql}) VALUES {', '.join(value_groups)}"
        if self._returning:
            if not dialect.supports_returning:
                msg = f"{dialect.name} does not support RETURNING"
                raise ValueError(msg)
            sql = f"{sql} RETURNING {type(self)._RETURNING_COLS}"
        return sql, compiler.params

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


class UpsertQuery[RecordT]:
    """Dialect-aware upsert built from an INSERT values clause."""

    __slots__ = (
        "_adapter",
        "_conflict_columns",
        "_do_nothing",
        "_returning",
        "_set_values",
        "_table_name",
        "_values",
    )

    _RETURNING_COLS: str
    _ROW_MAPPER: Any

    def __init__(
        self,
        adapter: DBAdapter,
        *,
        table_name: str,
        values: tuple[tuple[str, Any], ...],
        conflict_columns: tuple[str, ...] = (),
        set_values: tuple[tuple[str, Any], ...] = (),
        do_nothing: bool = False,
        returning: bool = False,
    ) -> None:
        self._adapter = adapter
        self._table_name = table_name
        self._values = values
        self._conflict_columns = conflict_columns
        self._set_values = set_values
        self._do_nothing = do_nothing
        self._returning = returning

    def do_nothing(self) -> Self:
        return type(self)(
            self._adapter,
            table_name=self._table_name,
            values=self._values,
            conflict_columns=self._conflict_columns,
            set_values=(),
            do_nothing=True,
            returning=self._returning,
        )

    def do_update(self, **columns: Any) -> Self:
        pairs = tuple((name, value) for name, value in columns.items())
        if not pairs:
            msg = "do_update() requires at least one column"
            raise ValueError(msg)
        return type(self)(
            self._adapter,
            table_name=self._table_name,
            values=self._values,
            conflict_columns=self._conflict_columns,
            set_values=pairs,
            do_nothing=False,
            returning=self._returning,
        )

    def returning(self) -> Self:
        return type(self)(
            self._adapter,
            table_name=self._table_name,
            values=self._values,
            conflict_columns=self._conflict_columns,
            set_values=self._set_values,
            do_nothing=self._do_nothing,
            returning=True,
        )

    def to_sql(self) -> tuple[str, list[Any]]:
        if not self._values:
            msg = "upsert requires INSERT values"
            raise ValueError(msg)
        if not self._conflict_columns and self._adapter.dialect.upsert_style == "conflict":
            msg = "on_conflict() requires at least one conflict target column"
            raise ValueError(msg)
        if not self._do_nothing and not self._set_values:
            msg = "upsert requires do_nothing() or do_update(...)"
            raise ValueError(msg)
        dialect = self._adapter.dialect
        cols = ", ".join(dialect.quote(name) for name, _ in self._values)
        compiler = SQLCompiler(dialect)
        placeholders = [_compile_value(compiler, value) for _, value in self._values]
        sql = f"INSERT INTO {self._table_name} ({cols}) VALUES ({', '.join(placeholders)})"
        if dialect.upsert_style == "duplicate":
            if self._do_nothing:
                # MySQL has no DO NOTHING; no-op update on first inserted column.
                first = self._values[0][0]
                quoted = dialect.quote(first)
                sql = f"{sql} ON DUPLICATE KEY UPDATE {quoted} = {quoted}"
            else:
                assignments = [
                    f"{dialect.quote(name)} = {_compile_value(compiler, value)}"
                    for name, value in self._set_values
                ]
                sql = f"{sql} ON DUPLICATE KEY UPDATE {', '.join(assignments)}"
        elif self._do_nothing:
            targets = ", ".join(dialect.quote(name) for name in self._conflict_columns)
            sql = f"{sql} ON CONFLICT ({targets}) DO NOTHING"
        else:
            targets = ", ".join(dialect.quote(name) for name in self._conflict_columns)
            assignments = [
                f"{dialect.quote(name)} = {_compile_value(compiler, value)}"
                for name, value in self._set_values
            ]
            sql = f"{sql} ON CONFLICT ({targets}) DO UPDATE SET {', '.join(assignments)}"
        if self._returning:
            if not dialect.supports_returning:
                msg = f"{dialect.name} does not support RETURNING"
                raise ValueError(msg)
            sql = f"{sql} RETURNING {type(self)._RETURNING_COLS}"
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
            raise NotFoundError("upsert ... returning produced no row")
        return cast(RecordT, type(self)._ROW_MAPPER(row))


class ProjectedSelectQuery[*Ts]:
    """SELECT of explicit columns hydrating ``tuple[*Ts]`` (or a scalar for one column)."""

    __slots__ = ("_adapter", "_columns", "_from_sql", "_limit", "_offset", "_order_by", "_where")

    def __init__(
        self,
        adapter: DBAdapter,
        *,
        from_sql: str,
        columns: tuple[Column[Any], ...],
        where: tuple[Expr[bool], ...] = (),
        order_by: tuple[OrderByClause, ...] = (),
        limit: int | None = None,
        offset: int | None = None,
    ) -> None:
        self._adapter = adapter
        self._from_sql = from_sql
        self._columns = columns
        self._where = where
        self._order_by = order_by
        self._limit = limit
        self._offset = offset

    def where(self, *predicates: Expr[bool]) -> ProjectedSelectQuery[*Ts]:
        if not predicates:
            return self
        return ProjectedSelectQuery(
            self._adapter,
            from_sql=self._from_sql,
            columns=self._columns,
            where=self._where + predicates,
            order_by=self._order_by,
            limit=self._limit,
            offset=self._offset,
        )

    def order_by(self, *clauses: OrderByClause) -> ProjectedSelectQuery[*Ts]:
        if not clauses:
            return self
        return ProjectedSelectQuery(
            self._adapter,
            from_sql=self._from_sql,
            columns=self._columns,
            where=self._where,
            order_by=self._order_by + clauses,
            limit=self._limit,
            offset=self._offset,
        )

    def limit(self, n: int) -> ProjectedSelectQuery[*Ts]:
        return ProjectedSelectQuery(
            self._adapter,
            from_sql=self._from_sql,
            columns=self._columns,
            where=self._where,
            order_by=self._order_by,
            limit=n,
            offset=self._offset,
        )

    def offset(self, n: int) -> ProjectedSelectQuery[*Ts]:
        return ProjectedSelectQuery(
            self._adapter,
            from_sql=self._from_sql,
            columns=self._columns,
            where=self._where,
            order_by=self._order_by,
            limit=self._limit,
            offset=n,
        )

    def to_sql(self) -> tuple[str, list[Any]]:
        dialect = self._adapter.dialect
        select_list = ", ".join(
            f"{dialect.quote(column.table)}.{dialect.quote(column.name)}"
            for column in self._columns
        )
        compiler = SQLCompiler(dialect, initial_sql=f"SELECT {select_list} FROM {self._from_sql}")
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
        return cast(tuple[*Ts], tuple(row))

    async def all(self) -> list[tuple[*Ts]]:
        sql, params = self.to_sql()
        rows = await self._adapter.fetch_all(sql, params)
        return [self._hydrate(row) for row in rows]

    async def first(self) -> tuple[*Ts] | None:
        rows = await self.limit(1).all()
        return rows[0] if rows else None

    async def one(self) -> tuple[*Ts]:
        rows = await self.limit(2).all()
        if not rows:
            raise NotFoundError("expected exactly one row, found 0")
        if len(rows) > 1:
            raise MultipleResultsError("expected exactly one row, found multiple")
        return rows[0]


def insert_many_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    attr_to_sql: Mapping[str, str],
) -> tuple[tuple[tuple[str, Any], ...], ...]:
    """Normalize attr-keyed mappings into ordered SQL-name value tuples."""
    if not rows:
        return ()
    normalized: list[tuple[tuple[str, Any], ...]] = []
    column_order: list[str] | None = None
    for row in rows:
        pairs: list[tuple[str, Any]] = []
        for attr, value in row.items():
            sql_name = attr_to_sql.get(attr, attr)
            pairs.append((sql_name, value))
        pairs_tuple = tuple(pairs)
        names = [name for name, _ in pairs_tuple]
        if column_order is None:
            column_order = names
        elif names != column_order:
            msg = "insert_many() requires every row to use the same column set/order"
            raise ValueError(msg)
        normalized.append(pairs_tuple)
    return tuple(normalized)
