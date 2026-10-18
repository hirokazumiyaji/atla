"""Emit typed query / mutation builders."""

from __future__ import annotations

from atla.dialect import (
    CockroachDialect,
    Dialect,
    MySQLDialect,
    PostgresDialect,
    SQLiteDialect,
    TiDBDialect,
)
from atlagen.ir import ColumnIR, SchemaIR, TableIR
from atlagen.naming import safe_attr_name


def emit_query(schema: SchemaIR) -> str:
    imports: set[tuple[str, str]] = set()
    for table in schema.tables:
        for column in table.columns:
            if column.import_symbol is not None:
                imports.add(column.import_symbol)

    needs_many = any(_inbound_relations(table, schema) for table in schema.tables)
    needs_one = any(_outbound_relations(table, schema) for table in schema.tables)
    needs_preload = needs_many or needs_one
    query_imports = [
        "BaseDeleteQuery",
        "BaseInsertQuery",
        "BaseSelectQuery",
        "BaseUpdateQuery",
    ]
    if needs_many:
        query_imports.append("PreloadSelectQuery")
    if needs_one:
        query_imports.append("PreloadOneSelectQuery")
    expr_imports = ["Column", "Expr"]
    if needs_preload:
        expr_imports.append("OrderByClause")
    lines = [
        "from __future__ import annotations",
        "",
        "from collections.abc import Sequence",
        "from typing import Any",
        "",
        "from atla.adapter.base import DBAdapter",
        "from atla.advanced import BaseInsertManyQuery, UpsertQuery, insert_many_rows",
        "from atla.errors import NotFoundError",
        f"from atla.expressions import {', '.join(expr_imports)}",
        "from atla.query import (",
        *[f"    {name}," for name in query_imports],
        ")",
        "from atla.types import UNSET, UnsetType",
    ]
    for module, symbol in sorted(imports):
        lines.append(f"from {module} import {symbol}")
    if schema.tables:
        model_parts: list[str] = []
        for table in schema.tables:
            if needs_preload:
                model_parts.append(table.model_name)
            model_parts.append(f"{table.model_name}Record")
            model_parts.append(f"{table.model_name}InsertParam")
        lines.append(f"from .models import {', '.join(model_parts)}")
    lines.append("")

    if not schema.tables:
        lines.append("# No tables defined in schema.")
        lines.append("")
        return "\n".join(lines)

    for table in schema.tables:
        lines.extend(_emit_table_builders(table, schema=schema, dialect=schema.dialect))
        lines.append("")
    return "\n".join(lines)


def _emit_table_builders(table: TableIR, *, schema: SchemaIR, dialect: str) -> list[str]:
    quoted_table = _quote_table(table.name, dialect=dialect)
    select_cols = ", ".join(
        f"{quoted_table}.{_quote_ident(column.name, dialect=dialect)}" for column in table.columns
    )
    returning_cols = ", ".join(
        _quote_ident(column.name, dialect=dialect) for column in table.columns
    )
    row_mapper = (
        f"    _ROW_MAPPER = {table.model_name}Record._from_row"
        "  # pyright: ignore[reportPrivateUsage]"
    )
    codec_free = all(column.row_codec is None for column in table.columns)
    query_extras = [row_mapper]
    if codec_free:
        query_extras.append(f"    _ROW_CTOR = {table.model_name}Record")
    attr_map_items = ", ".join(f"{column.attr_name!r}: {column.name!r}" for column in table.columns)
    lines = [
        f"class {table.model_name}Query(BaseSelectQuery[{table.model_name}Record]):",
        "    __slots__ = ()",
        f"    _TABLE_NAME = {quoted_table!r}",
        f"    _SELECT_COLS = {select_cols!r}",
        f"    _FROM_SQL = {quoted_table!r}",
        f"    _COLUMN_COUNT = {len(table.columns)}",
        f"    _SELECT_SQL = {('SELECT ' + select_cols + ' FROM ' + quoted_table)!r}",
        f"    _COUNT_SQL = {('SELECT COUNT(*) FROM ' + quoted_table)!r}",
        *query_extras,
    ]
    lines.extend(_emit_with_methods(table, schema=schema, on_query=True))
    lines.extend(
        [
            "",
            f"class {table.model_name}UpsertBuilder(UpsertQuery[{table.model_name}Record]):",
            "    __slots__ = ()",
            f"    _RETURNING_COLS = {returning_cols!r}",
            row_mapper,
            f"    _ATTR_TO_SQL = {{{attr_map_items}}}",
            "",
        ]
    )
    lines.extend(_emit_do_update_method(table))
    if dialect in {"mysql", "tidb"}:
        lines.extend(
            [
                "",
                f"    def returning(self) -> {table.model_name}UpsertBuilder:",
                f'        raise ValueError("{dialect} does not support RETURNING")',
            ]
        )
    lines.extend(
        [
            "",
            f"class {table.model_name}InsertBuilder(BaseInsertQuery[{table.model_name}Record]):",
            "    __slots__ = ()",
            f"    _TABLE_NAME = {quoted_table!r}",
            f"    _RETURNING_COLS = {returning_cols!r}",
            row_mapper,
            f"    _UPSERT_CLS = {table.model_name}UpsertBuilder",
        ]
    )
    if dialect in {"mysql", "tidb"}:
        lines.extend(
            [
                "",
                f"    def returning(self) -> {table.model_name}InsertBuilder:",
                f'        raise ValueError("{dialect} does not support RETURNING")',
            ]
        )
    lines.extend(
        [
            "",
            "    def on_conflict(",
            "        self,",
            "        *columns: Column[Any] | str,",
            f"    ) -> {table.model_name}UpsertBuilder:",
            "        return super().on_conflict(*columns)  # type: ignore[return-value]",
            "",
            f"class {table.model_name}UpdateBuilder(BaseUpdateQuery[{table.model_name}Record]):",
            "    __slots__ = ()",
            f"    _TABLE_NAME = {quoted_table!r}",
            f"    _RETURNING_COLS = {returning_cols!r}",
            row_mapper,
            "",
        ]
    )
    # Include PK columns so natural/composite-key tables remain updatable.
    set_columns = list(table.columns)
    if set_columns:
        lines.extend(
            [
                "    def set(",
                "        self,",
                "        *,",
            ]
        )
        for column in set_columns:
            value_type = column.python_type or column.base_python_type
            lines.append(
                f"        {column.attr_name}: {value_type} | "
                f"Expr[{value_type}] | UnsetType = UNSET,"
            )
        lines.append(f"    ) -> {table.model_name}UpdateBuilder:")
        lines.extend(
            _emit_mapped_call("        return self._set_values", set_columns, indent="        ")
        )
    else:
        lines.extend(
            [
                f"    def set(self) -> {table.model_name}UpdateBuilder:",
                '        raise ValueError("table has no updatable columns")',
            ]
        )
    if dialect in {"mysql", "tidb"}:
        lines.extend(
            [
                "",
                f"    def returning(self) -> {table.model_name}UpdateBuilder:",
                f'        raise ValueError("{dialect} does not support RETURNING")',
            ]
        )
    lines.extend(
        [
            "",
            f"class {table.model_name}DeleteBuilder(BaseDeleteQuery[{table.model_name}Record]):",
            "    __slots__ = ()",
            f"    _RETURNING_COLS = {returning_cols!r}",
            row_mapper,
        ]
    )
    if dialect in {"mysql", "tidb"}:
        lines.extend(
            [
                "",
                f"    def returning(self) -> {table.model_name}DeleteBuilder:",
                f'        raise ValueError("{dialect} does not support RETURNING")',
            ]
        )
    lines.extend(
        [
            "",
            f"class {table.model_name}Repository:",
            '    __slots__ = ("_adapter",)',
            "",
            "    def __init__(self, adapter: DBAdapter) -> None:",
            "        self._adapter = adapter",
            "",
            f"    def query(self) -> {table.model_name}Query:",
            f"        return {table.model_name}Query(self._adapter)",
            "",
            f"    def where(self, *predicates: Expr[bool]) -> {table.model_name}Query:",
            f"        return {table.model_name}Query(self._adapter).where(*predicates)",
            "",
            "    def insert(",
            "        self,",
            "        *,",
        ]
    )
    required = [
        column for column in table.columns if not column.nullable and not column.has_default
    ]
    optional = [column for column in table.columns if column.nullable or column.has_default]
    for column in required:
        lines.append(f"        {column.attr_name}: {column.base_python_type},")
    for column in optional:
        value_type = column.python_type or column.base_python_type
        lines.append(f"        {column.attr_name}: {value_type} | UnsetType = UNSET,")
    lines.append(f"    ) -> {table.model_name}InsertBuilder:")
    lines.extend(
        _emit_mapped_call(
            f"        return {table.model_name}InsertBuilder(self._adapter).values",
            list(table.columns),
            indent="        ",
        )
    )
    lines.extend(
        [
            "",
            f"    _ATTR_TO_SQL = {{{attr_map_items}}}",
            "",
            "    def insert_many(",
            "        self,",
            f"        rows: Sequence[{table.model_name}InsertParam],",
            f"    ) -> {table.model_name}InsertManyBuilder:",
            f"        return {table.model_name}InsertManyBuilder(",
            "            self._adapter,",
            f"            table_name={quoted_table!r},",
            "            rows=insert_many_rows(rows, attr_to_sql=self._ATTR_TO_SQL),",
            "        )",
            "",
            f"    def update(self) -> {table.model_name}UpdateBuilder:",
            f"        return {table.model_name}UpdateBuilder(self._adapter)",
            "",
            f"    def delete(self) -> {table.model_name}DeleteBuilder:",
            f"        return {table.model_name}DeleteBuilder("
            f"self._adapter, table_name={quoted_table!r})",
        ]
    )
    lines.extend(
        _emit_pk_lookup_methods(
            table,
            dialect=dialect,
            select_cols=select_cols,
            quoted_table=quoted_table,
        )
    )
    lines.extend(_emit_with_methods(table, schema=schema, on_query=False))
    many_cls = f"{table.model_name}InsertManyBuilder"
    insert_many_cls = [
        "",
        f"class {many_cls}(BaseInsertManyQuery[{table.model_name}Record]):",
        "    __slots__ = ()",
        f"    _RETURNING_COLS = {returning_cols!r}",
        (
            f"    _ROW_MAPPER = {table.model_name}Record._from_row"
            "  # pyright: ignore[reportPrivateUsage]"
        ),
        "",
    ]
    repo_idx = next(
        i for i, line in enumerate(lines) if line.startswith(f"class {table.model_name}Repository")
    )
    lines[repo_idx:repo_idx] = insert_many_cls
    return lines


def _inbound_relations(table: TableIR, schema: SchemaIR) -> list[tuple[TableIR, str, str]]:
    """Return ``(child_table, fk_attr, parent_key_attr)`` for single-column inbound FKs."""
    pk = table.primary_key[0] if table.primary_key else None
    if pk is None:
        return []
    parent_key_attr = next(
        (column.attr_name for column in table.columns if column.name == pk),
        pk,
    )
    relations: list[tuple[TableIR, str, str]] = []
    for child in schema.tables:
        if child.name == table.name:
            continue
        for foreign_key in child.foreign_keys:
            if foreign_key.ref_table.rsplit(".", 1)[-1] != table.name.rsplit(".", 1)[-1]:
                continue
            if len(foreign_key.columns) != 1 or len(foreign_key.ref_columns) != 1:
                continue
            if foreign_key.ref_columns[0] != pk:
                continue
            fk_name = foreign_key.columns[0]
            fk_attr = next(
                (column.attr_name for column in child.columns if column.name == fk_name),
                fk_name,
            )
            relations.append((child, fk_attr, parent_key_attr))
    return relations


def _outbound_relations(table: TableIR, schema: SchemaIR) -> list[tuple[TableIR, str, str, bool]]:
    """Return ``(parent_table, fk_attr, parent_key_attr, nullable)`` for outbound FKs."""
    relations: list[tuple[TableIR, str, str, bool]] = []
    for foreign_key in table.foreign_keys:
        if len(foreign_key.columns) != 1 or len(foreign_key.ref_columns) != 1:
            continue
        fk_name = foreign_key.columns[0]
        ref_col = foreign_key.ref_columns[0]
        ref_table = foreign_key.ref_table
        ref_leaf = ref_table.rsplit(".", 1)[-1]
        parent = next(
            (candidate for candidate in schema.tables if candidate.name == ref_table),
            None,
        )
        if parent is None:
            parent = next(
                (
                    candidate
                    for candidate in schema.tables
                    if candidate.name.rsplit(".", 1)[-1] == ref_leaf
                ),
                None,
            )
        if parent is None:
            continue
        parent_pk = parent.primary_key[0] if parent.primary_key else None
        if parent_pk is None or ref_col != parent_pk:
            continue
        fk_column = next((column for column in table.columns if column.name == fk_name), None)
        fk_attr = fk_column.attr_name if fk_column is not None else fk_name
        nullable = fk_column.nullable if fk_column is not None else True
        parent_key_attr = next(
            (column.attr_name for column in parent.columns if column.name == parent_pk),
            parent_pk,
        )
        relations.append((parent, fk_attr, parent_key_attr, nullable))
    return relations


def _outbound_non_pk_fk_comments(table: TableIR, schema: SchemaIR) -> list[str]:
    """Emit generated comments for FKs that do not get ``with_*`` helpers."""
    comments: list[str] = []
    for foreign_key in table.foreign_keys:
        if len(foreign_key.columns) != 1 or len(foreign_key.ref_columns) != 1:
            continue
        fk_name = foreign_key.columns[0]
        ref_col = foreign_key.ref_columns[0]
        ref_table = foreign_key.ref_table
        ref_leaf = ref_table.rsplit(".", 1)[-1]
        parent = next(
            (candidate for candidate in schema.tables if candidate.name == ref_table),
            None,
        )
        if parent is None:
            parent = next(
                (
                    candidate
                    for candidate in schema.tables
                    if candidate.name.rsplit(".", 1)[-1] == ref_leaf
                ),
                None,
            )
        if parent is None:
            continue
        parent_pk = parent.primary_key[0] if parent.primary_key else None
        if parent_pk is None or ref_col != parent_pk:
            comments.append(
                f"    # FK to non-PK column: {fk_name} -> {ref_table}({ref_col}); "
                "no eager-load helper"
            )
    return comments


def _with_name_from_fk(fk_attr: str, parent_accessor: str) -> str:
    """Derive ``with_<name>`` from the FK attribute when it ends with ``_id``."""
    if fk_attr.endswith("_id") and len(fk_attr) > 3:
        return f"with_{fk_attr[:-3]}"
    return f"with_{parent_accessor}"


def _emit_with_methods(table: TableIR, *, schema: SchemaIR, on_query: bool) -> list[str]:
    """Emit ``with_<relation>`` batch eager-load helpers.

    When ``on_query`` is true, methods call ``self.preload`` / ``self.preload_one``
    so filters already applied on the query are preserved. Repository methods
    construct a fresh ``{Model}Query`` instead.
    """
    lines: list[str] = []
    used_names: set[str] = set()
    receiver = "self" if on_query else f"{table.model_name}Query(self._adapter)"

    for child, fk_attr, parent_key_attr in _inbound_relations(table, schema):
        child_sql = child.name.rsplit(".", 1)[-1]
        method = safe_attr_name(f"with_{child_sql}")
        if method in used_names:
            method = safe_attr_name(f"with_{child_sql}_{fk_attr}")
        used_names.add(method)
        lines.extend(
            [
                "",
                f"    def {method}(",
                "        self,",
                "        *predicates: Expr[bool],",
                "        order_by: Sequence[OrderByClause] = (),",
                f"    ) -> PreloadSelectQuery[{table.model_name}Record, {child.model_name}Record]:",
                f"        return {receiver}.preload(",
                f"            {child.model_name},",
                "            *predicates,",
                f"            fk={child.model_name}.{fk_attr},",
                f"            key={table.model_name}.{parent_key_attr},",
                f"            fk_attr={fk_attr!r},",
                f"            key_attr={parent_key_attr!r},",
                "            order_by=order_by,",
                "        )",
            ]
        )

    non_pk_comments = _outbound_non_pk_fk_comments(table, schema)
    if non_pk_comments:
        lines.append("")
        lines.extend(non_pk_comments)

    for parent, fk_attr, parent_key_attr, nullable in _outbound_relations(table, schema):
        # Prefer FK-derived names (``manager_id`` → ``with_manager``) over parent accessor.
        method = safe_attr_name(_with_name_from_fk(fk_attr, parent.accessor_name))
        if method in used_names:
            method = safe_attr_name(f"{method}_{fk_attr}")
        used_names.add(method)
        # Always ``Parent | None``: ``*predicates`` can filter out the parent even
        # when the FK column itself is non-nullable.
        parent_type = f"{parent.model_name}Record | None"
        lines.extend(
            [
                "",
                f"    def {method}(",
                "        self,",
                "        *predicates: Expr[bool],",
                "        order_by: Sequence[OrderByClause] = (),",
                f"    ) -> PreloadOneSelectQuery[{table.model_name}Record, {parent_type}]:",
                f"        return {receiver}.preload_one(",
                f"            {parent.model_name},",
                "            *predicates,",
                f"            fk={table.model_name}.{fk_attr},",
                f"            key={parent.model_name}.{parent_key_attr},",
                f"            fk_attr={fk_attr!r},",
                f"            key_attr={parent_key_attr!r},",
                "            order_by=order_by,",
                f"            nullable={nullable!r},",
                "        )",
            ]
        )
    return lines


def _emit_do_update_method(table: TableIR) -> list[str]:
    """Emit typed ``do_update`` matching ``UpdateBuilder.set``."""
    columns = list(table.columns)
    if not columns:
        return [
            # Same intentional narrowing as the typed overload below.
            "    def do_update(  # type: ignore[override]"
            "  # pyright: ignore[reportIncompatibleMethodOverride]",
            "        self,",
            f"    ) -> {table.model_name}UpsertBuilder:",
            '        raise ValueError("table has no updatable columns")',
        ]
    lines = [
        # Generated builders intentionally narrow the base ``do_update(**columns)``
        # (bare SQL names, for hand-written builders) to typed keyword arguments.
        "    def do_update(  # type: ignore[override]"
        "  # pyright: ignore[reportIncompatibleMethodOverride]",
        "        self,",
        "        *,",
    ]
    for column in columns:
        value_type = column.python_type or column.base_python_type
        lines.append(
            f"        {column.attr_name}: {value_type} | Expr[{value_type}] | UnsetType = UNSET,"
        )
    lines.append(f"    ) -> {table.model_name}UpsertBuilder:")
    lines.extend(
        _emit_mapped_call("        return self._do_update_values", columns, indent="        ")
    )
    return lines


def _emit_mapped_call(
    prefix: str,
    columns: list[ColumnIR],
    *,
    indent: str,
) -> list[str]:
    """Emit a call mapping Python attr names onto SQL column names via a dict.

    Always use ``**{sql_name: attr}`` so SQL identifiers that are keywords or
    otherwise invalid as Python kwargs remain correct.
    """
    if not columns:
        return [f"{prefix}()"]
    lines = [f"{prefix}(**{{"]
    for column in columns:
        lines.append(f"{indent}    {column.name!r}: {column.attr_name},")
    lines.append(f"{indent}}})")
    return lines


def _dialect_for(name: str) -> Dialect:
    mapping: dict[str, type[Dialect]] = {
        "postgres": PostgresDialect,
        "cockroach": CockroachDialect,
        "mysql": MySQLDialect,
        "tidb": TiDBDialect,
        "sqlite": SQLiteDialect,
    }
    try:
        return mapping[name]()
    except KeyError as exc:
        msg = f"unsupported dialect for quoting: {name!r}"
        raise ValueError(msg) from exc


def _placeholder(dialect: str, index: int) -> str:
    if dialect in {"mysql", "tidb"}:
        return "%s"
    if dialect == "sqlite":
        return "?"
    return f"${index}"


def _emit_pk_lookup_methods(
    table: TableIR,
    *,
    dialect: str,
    select_cols: str,
    quoted_table: str,
) -> list[str]:
    """Emit AOT ``get`` / ``get_one`` / ``get_many`` for tables with a primary key."""
    if not table.primary_key:
        return []
    pk_columns: list[ColumnIR] = []
    for name in table.primary_key:
        column = next((c for c in table.columns if c.name == name), None)
        if column is None:
            return []
        pk_columns.append(column)

    where_parts: list[str] = []
    for index, column in enumerate(pk_columns, start=1):
        col_sql = f"{quoted_table}.{_quote_ident(column.name, dialect=dialect)}"
        where_parts.append(f"{col_sql} = {_placeholder(dialect, index)}")
    get_sql = f"SELECT {select_cols} FROM {quoted_table} WHERE {' AND '.join(where_parts)} LIMIT 1"
    record = f"{table.model_name}Record"
    mapper = f"{record}._from_row"
    lines = [
        "",
        f"    _GET_BY_PK_SQL = {get_sql!r}",
        "",
    ]

    if len(pk_columns) == 1:
        pk = pk_columns[0]
        pk_type = pk.base_python_type
        pk_qual = f"{quoted_table}.{_quote_ident(pk.name, dialect=dialect)}"
        get_many_prefix = f"SELECT {select_cols} FROM {quoted_table} WHERE {pk_qual} IN ("
        lines.extend(
            [
                f"    _GET_MANY_PREFIX = {get_many_prefix!r}",
                "",
                f"    async def get(self, {pk.attr_name}: {pk_type}) -> {record} | None:",
                "        row = await self._adapter.fetch_one("
                f"self._GET_BY_PK_SQL, ({pk.attr_name},))",
                f"        return None if row is None else {mapper}(row)"
                "  # pyright: ignore[reportPrivateUsage]",
                "",
                f"    async def get_one(self, {pk.attr_name}: {pk_type}) -> {record}:",
                f"        row = await self.get({pk.attr_name})",
                "        if row is None:",
                f'            raise NotFoundError("{table.model_name} not found")',
                "        return row",
                "",
                f"    async def get_many(self, keys: Sequence[{pk_type}]) -> list[{record}]:",
                '        """Fetch rows for ``keys`` with batched ``WHERE pk IN (...)`` queries.',
                "",
                "        Result order follows the database row order, not ``keys`` order.",
                "        Duplicate keys collapse to one row, and missing keys are omitted.",
                "        Keys are chunked (500 per query) to stay under bind-parameter limits.",
                '        """',
                "        if not keys:",
                "            return []",
                "        dialect = self._adapter.dialect",
                "        batch_size = 500",
                f"        results: list[{record}] = []",
                "        key_list = list(keys)",
                "        for start in range(0, len(key_list), batch_size):",
                "            batch = key_list[start : start + batch_size]",
                '            placeholders = ", ".join('
                "dialect.placeholder(i) for i in range(1, len(batch) + 1))",
                '            sql = f"{self._GET_MANY_PREFIX}{placeholders})"',
                "            rows = await self._adapter.fetch_all(sql, batch)",
                f"            results.extend({mapper}(row) for row in rows)"
                "  # pyright: ignore[reportPrivateUsage]",
                "        return results",
            ]
        )
        return lines

    # Composite primary key.
    pk_params = ", ".join(f"{column.attr_name}: {column.base_python_type}" for column in pk_columns)
    pk_tuple_type = "tuple[" + ", ".join(c.base_python_type for c in pk_columns) + "]"
    pk_args = ", ".join(column.attr_name for column in pk_columns)
    get_kwargs = ", ".join(f"{column.attr_name}={column.attr_name}" for column in pk_columns)
    composite_cols = ", ".join(
        f"{quoted_table}.{_quote_ident(column.name, dialect=dialect)}" for column in pk_columns
    )
    composite_prefix = f"SELECT {select_cols} FROM {quoted_table} WHERE ({composite_cols}) IN ("
    lines.extend(
        [
            f"    async def get(self, *, {pk_params}) -> {record} | None:",
            f"        row = await self._adapter.fetch_one(self._GET_BY_PK_SQL, ({pk_args}))",
            f"        return None if row is None else {mapper}(row)"
            "  # pyright: ignore[reportPrivateUsage]",
            "",
            f"    async def get_one(self, *, {pk_params}) -> {record}:",
            f"        row = await self.get({get_kwargs})",
            "        if row is None:",
            f'            raise NotFoundError("{table.model_name} not found")',
            "        return row",
            "",
            f"    _GET_MANY_PREFIX = {composite_prefix!r}",
            "",
            f"    async def get_many(self, keys: Sequence[{pk_tuple_type}]) -> list[{record}]:",
            '        """Fetch rows for composite ``keys`` via batched row-value ``IN`` queries.',
            "",
            "        Issues ``WHERE (a, b) IN ((?, ?), ...)`` (PostgreSQL, MySQL 8+, SQLite 3.15+)",
            "        in chunks that stay under the driver bind-parameter limit.",
            "        Result order follows the database row order, not ``keys`` order.",
            "        Duplicate keys collapse to one row, and missing keys are omitted.",
            '        """',
            "        if not keys:",
            "            return []",
            "        dialect = self._adapter.dialect",
            f"        width = {len(pk_columns)}",
            "        batch_size = max(1, 500 // width)",
            f"        results: list[{record}] = []",
            "        key_list = list(keys)",
            "        for start in range(0, len(key_list), batch_size):",
            "            batch = key_list[start : start + batch_size]",
            "            groups: list[str] = []",
            "            params: list[Any] = []",
            "            for key in batch:",
            "                base = len(params)",
            "                marks = [dialect.placeholder(base + i + 1) for i in range(width)]",
            '                groups.append("(" + ", ".join(marks) + ")")',
            "                params.extend(key)",
            '            sql = self._GET_MANY_PREFIX + ", ".join(groups) + ")"',
            "            rows = await self._adapter.fetch_all(sql, params)",
            f"            results.extend({mapper}(row) for row in rows)"
            "  # pyright: ignore[reportPrivateUsage]",
            "        return results",
        ]
    )
    return lines


def _quote_table(name: str, *, dialect: str) -> str:
    # Schema-qualified names (public.users) must quote each component.
    d = _dialect_for(dialect)
    return ".".join(d.quote(part) for part in name.split("."))


def _quote_ident(name: str, *, dialect: str) -> str:
    return _dialect_for(dialect).quote(name)
