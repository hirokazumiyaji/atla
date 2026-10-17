"""Bulk insert, upsert, and projected SELECT helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar, Self, cast

from atla.adapter.base import DBAdapter
from atla.compiler import SQLCompiler
from atla.errors import MultipleResultsError, NotFoundError
from atla.expressions import Column, Expr, OrderByClause, and_
from atla.types import is_unset

# Stay under common driver bind limits (SQLite ~32766).
_MAX_BIND_PARAMS = 30_000


def _compile_value(compiler: SQLCompiler, value: Any) -> str:
    if isinstance(value, Expr):
        return compiler.compile_fragment(value)  # pyright: ignore[reportUnknownArgumentType]
    return compiler.add_param(value)


def conflict_column_names(*columns: Column[Any] | str) -> tuple[str, ...]:
    """Normalize ``on_conflict`` targets to bare SQL identifier names."""
    names: list[str] = []
    for column in columns:
        if isinstance(column, Column):
            names.append(column.name)
        else:
            names.append(column)
    return tuple(names)


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

    def _validated_columns(self) -> list[str]:
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
        return columns

    def _chunk_rows(self) -> list[tuple[tuple[tuple[str, Any], ...], ...]]:
        columns = self._validated_columns()
        chunk_size = max(1, _MAX_BIND_PARAMS // len(columns))
        rows = self._rows
        return [rows[i : i + chunk_size] for i in range(0, len(rows), chunk_size)]

    def _rows_are_literals(self, rows: tuple[tuple[tuple[str, Any], ...], ...]) -> bool:
        return all(not isinstance(value, Expr) for row in rows for _, value in row)

    def _sql_for_rows(
        self,
        rows: tuple[tuple[tuple[str, Any], ...], ...],
        columns: list[str],
    ) -> tuple[str, list[Any]]:
        dialect = self._adapter.dialect
        cols_sql = ", ".join(dialect.quote(name) for name in columns)
        if self._rows_are_literals(rows):
            # Expand placeholders without per-value Expr dispatch.
            params: list[Any] = []
            value_groups: list[str] = []
            for row in rows:
                start = len(params) + 1
                ph = (
                    "("
                    + ", ".join(
                        dialect.placeholder(start + offset) for offset in range(len(columns))
                    )
                    + ")"
                )
                value_groups.append(ph)
                params.extend(value for _, value in row)
            sql = f"INSERT INTO {self._table_name} ({cols_sql}) VALUES {', '.join(value_groups)}"
            if self._returning:
                if not dialect.supports_returning:
                    msg = f"{dialect.name} does not support RETURNING"
                    raise ValueError(msg)
                sql = f"{sql} RETURNING {type(self)._RETURNING_COLS}"
            return sql, params

        compiler = SQLCompiler(dialect)
        value_groups = []
        for row in rows:
            placeholders = [_compile_value(compiler, value) for _, value in row]
            value_groups.append(f"({', '.join(placeholders)})")
        sql = f"INSERT INTO {self._table_name} ({cols_sql}) VALUES {', '.join(value_groups)}"
        if self._returning:
            if not dialect.supports_returning:
                msg = f"{dialect.name} does not support RETURNING"
                raise ValueError(msg)
            sql = f"{sql} RETURNING {type(self)._RETURNING_COLS}"
        return sql, compiler.params

    def to_sql(self) -> tuple[str, list[Any]]:
        columns = self._validated_columns()
        return self._sql_for_rows(self._rows, columns)

    async def execute(self) -> int:
        """Insert all rows and return the number of rows submitted.

        Uses atomic multi-row ``INSERT ... VALUES (...), (...)`` statements, chunked
        to the bind-parameter limit. Each chunk is one statement (all-or-nothing for
        that chunk). Chunks are not atomic together outside a transaction — wrap
        ``execute()`` in ``adapter.transaction()`` when the whole batch must commit
        or roll back as a unit.

        Return value: the sum of driver rowcounts across chunks.
        """
        columns = self._validated_columns()
        total = 0
        for chunk in self._chunk_rows():
            sql, params = self._sql_for_rows(chunk, columns)
            total += await self._adapter.execute(sql, params)
        return total

    async def all(self) -> list[RecordT]:
        if not self._returning:
            msg = "all() requires .returning()"
            raise ValueError(msg)
        columns = self._validated_columns()
        mapper = type(self)._ROW_MAPPER
        result: list[RecordT] = []
        for chunk in self._chunk_rows():
            sql, params = self._sql_for_rows(chunk, columns)
            rows = await self._adapter.fetch_all(sql, params)
            result.extend(mapper(row) for row in rows)
        return result


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
    _ROW_MAPPER: Callable[[Sequence[Any]], RecordT]
    _ATTR_TO_SQL: ClassVar[Mapping[str, str]] = {}

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
        """Accumulate DO UPDATE columns (bare SQL identifiers).

        Generated ``*UpsertBuilder.do_update`` maps Python attribute names onto
        SQL identifiers before calling :meth:`_do_update_values`. Hand-written
        callers may use this method with SQL names, or :meth:`do_update_attrs`
        for ``_ATTR_TO_SQL`` remapping.
        """
        return self._do_update_values(**columns)

    def _do_update_values(self, **columns: Any) -> Self:
        """Internal DO UPDATE accumulation shared with generated builders."""
        pairs = tuple((name, value) for name, value in columns.items() if not is_unset(value))
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

    def do_update_attrs(self, **columns: Any) -> Self:
        """Like ``do_update``, remapping Python attr names via ``_ATTR_TO_SQL``."""
        mapping = type(self)._ATTR_TO_SQL
        pairs = {
            mapping.get(name, name): value for name, value in columns.items() if not is_unset(value)
        }
        return self._do_update_values(**pairs)

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
            raise NotFoundError("upsert ... returning produced no row")
        if len(rows) > 1:
            raise MultipleResultsError("expected exactly one row, found multiple")
        # Map only the single accepted row (UPDATE/DELETE cannot be LIMITed).
        return type(self)._ROW_MAPPER(rows[0])


class ProjectedSelectQuery[*Ts]:
    """SELECT of explicit expressions hydrating ``tuple[*Ts]``.

    For a single-column projection, use :meth:`scalars` / :meth:`scalar_one` /
    :meth:`scalar_first` to unwrap ``tuple[T]`` into ``T``.
    """

    __slots__ = (
        "_adapter",
        "_columns",
        "_decoders",
        "_from_sql",
        "_join_parts",
        "_limit",
        "_needs_decode",
        "_offset",
        "_order_by",
        "_where",
    )

    def __init__(
        self,
        adapter: DBAdapter,
        *,
        from_sql: str,
        columns: tuple[Expr[Any], ...],
        where: tuple[Expr[bool], ...] = (),
        order_by: tuple[OrderByClause, ...] = (),
        limit: int | None = None,
        offset: int | None = None,
        join_parts: tuple[tuple[str, tuple[Expr[bool], ...]], ...] = (),
    ) -> None:
        self._adapter = adapter
        self._from_sql = from_sql
        self._columns = columns
        self._where = where
        self._order_by = order_by
        self._limit = limit
        self._offset = offset
        self._join_parts = join_parts
        decoders = tuple(expr.decoder if isinstance(expr, Column) else None for expr in columns)
        self._decoders = decoders
        self._needs_decode = any(decoder is not None for decoder in decoders)

    def _clone(
        self,
        *,
        where: tuple[Expr[bool], ...] | None = None,
        order_by: tuple[OrderByClause, ...] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        replace_limit: bool = False,
        replace_offset: bool = False,
    ) -> ProjectedSelectQuery[*Ts]:
        return ProjectedSelectQuery(
            self._adapter,
            from_sql=self._from_sql,
            columns=self._columns,
            where=self._where if where is None else where,
            order_by=self._order_by if order_by is None else order_by,
            limit=limit if replace_limit else self._limit,
            offset=offset if replace_offset else self._offset,
            join_parts=self._join_parts,
        )

    def where(self, *predicates: Expr[bool]) -> ProjectedSelectQuery[*Ts]:
        if not predicates:
            return self
        return self._clone(where=self._where + predicates)

    def order_by(self, *clauses: OrderByClause) -> ProjectedSelectQuery[*Ts]:
        if not clauses:
            return self
        return self._clone(order_by=self._order_by + clauses)

    def limit(self, n: int) -> ProjectedSelectQuery[*Ts]:
        return self._clone(limit=n, replace_limit=True)

    def offset(self, n: int) -> ProjectedSelectQuery[*Ts]:
        return self._clone(offset=n, replace_offset=True)

    def to_sql(self) -> tuple[str, list[Any]]:
        compiler = SQLCompiler(self._adapter.dialect)
        compiler.write("SELECT ")
        for index, expr in enumerate(self._columns):
            if index:
                compiler.write(", ")
            compiler.visit(expr)
        compiler.write(f" FROM {self._from_sql}")
        for prefix, on in self._join_parts:
            compiler.write(prefix)
            compiler.visit(and_(*on))
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
        if not self._needs_decode:
            return cast(tuple[*Ts], tuple(row))
        values: list[Any] = []
        for index, decoder in enumerate(self._decoders):
            value = row[index]
            values.append(value if decoder is None or value is None else decoder(value))
        return cast(tuple[*Ts], tuple(values))

    async def all(self) -> list[tuple[*Ts]]:
        sql, params = self.to_sql()
        rows = await self._adapter.fetch_all(sql, params)
        return [self._hydrate(row) for row in rows]

    def _with_fetch_limit(self, maximum: int) -> ProjectedSelectQuery[*Ts]:
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

    async def scalars[T](self: ProjectedSelectQuery[T]) -> list[T]:
        """Unwrap a single-column projection into a flat list."""
        if len(self._columns) != 1:
            msg = "scalars() requires a single-column projection"
            raise ValueError(msg)
        rows = await self.all()
        return [row[0] for row in rows]

    async def scalar_first[T](self: ProjectedSelectQuery[T]) -> T | None:
        """Return the first scalar from a single-column projection, or ``None``."""
        if len(self._columns) != 1:
            msg = "scalar_first() requires a single-column projection"
            raise ValueError(msg)
        row = await self.first()
        return None if row is None else row[0]

    async def scalar_one[T](self: ProjectedSelectQuery[T]) -> T:
        """Return the only scalar from a single-column projection."""
        if len(self._columns) != 1:
            msg = "scalar_one() requires a single-column projection"
            raise ValueError(msg)
        row = await self.one()
        return row[0]


def insert_many_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    attr_to_sql: Mapping[str, str],
) -> tuple[tuple[tuple[str, Any], ...], ...]:
    """Normalize attr-keyed mappings into ordered SQL-name value tuples.

    Column *set* must match across rows; key order is normalized to the first
    row's order so ordinary ``dict`` insertion-order differences are accepted.
    """
    if not rows:
        return ()
    normalized: list[tuple[tuple[str, Any], ...]] = []
    column_order: list[str] | None = None
    column_set: set[str] | None = None
    for row in rows:
        by_sql = {attr_to_sql.get(attr, attr): value for attr, value in row.items()}
        names = set(by_sql)
        if column_order is None:
            column_order = list(by_sql)
            column_set = names
        elif names != column_set:
            msg = "insert_many() requires every row to use the same column set"
            raise ValueError(msg)
        assert column_order is not None
        normalized.append(tuple((name, by_sql[name]) for name in column_order))
    return tuple(normalized)
