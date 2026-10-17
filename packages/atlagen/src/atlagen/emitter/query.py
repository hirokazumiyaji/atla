"""Emit typed query / mutation builders."""

from __future__ import annotations

from atlagen.ir import ColumnIR, SchemaIR, TableIR


def emit_query(schema: SchemaIR) -> str:
    lines = [
        "from __future__ import annotations",
        "",
        "from atla.adapter.base import DBAdapter",
        "from atla.expressions import Expr",
        "from atla.query import BaseDeleteQuery, BaseInsertQuery, BaseSelectQuery, BaseUpdateQuery",
        "from atla.types import UNSET, UnsetType",
    ]
    if schema.enums:
        enum_imports = ", ".join(enum.class_name for enum in schema.enums)
        lines.append(f"from .enums import {enum_imports}")
    if schema.tables:
        record_imports = ", ".join(f"{table.model_name}Record" for table in schema.tables)
        lines.append(f"from .models import {record_imports}")
    lines.append("")

    if not schema.tables:
        lines.append("# No tables defined in schema.")
        lines.append("")
        return "\n".join(lines)

    for table in schema.tables:
        lines.extend(_emit_table_builders(table, dialect=schema.dialect))
        lines.append("")
    return "\n".join(lines)


def _emit_table_builders(table: TableIR, *, dialect: str) -> list[str]:
    quoted_table = _quote_table(table.name, dialect=dialect)
    select_cols = ", ".join(
        f"{quoted_table}.{_quote_ident(column.name, dialect=dialect)}" for column in table.columns
    )
    returning_cols = ", ".join(
        _quote_ident(column.name, dialect=dialect) for column in table.columns
    )
    lines = [
        f"class {table.model_name}Query(BaseSelectQuery[{table.model_name}Record]):",
        "    __slots__ = ()",
        f"    _TABLE_NAME = {quoted_table!r}",
        f"    _SELECT_SQL = {('SELECT ' + select_cols + ' FROM ' + quoted_table)!r}",
        f"    _COUNT_SQL = {('SELECT COUNT(*) FROM ' + quoted_table)!r}",
        f"    _ROW_MAPPER = {table.model_name}Record._from_row",
        "",
        f"class {table.model_name}InsertBuilder(BaseInsertQuery[{table.model_name}Record]):",
        "    __slots__ = ()",
        f"    _TABLE_NAME = {quoted_table!r}",
        f"    _RETURNING_COLS = {returning_cols!r}",
        f"    _ROW_MAPPER = {table.model_name}Record._from_row",
        "",
        f"class {table.model_name}UpdateBuilder(BaseUpdateQuery[{table.model_name}Record]):",
        "    __slots__ = ()",
        f"    _TABLE_NAME = {quoted_table!r}",
        f"    _RETURNING_COLS = {returning_cols!r}",
        f"    _ROW_MAPPER = {table.model_name}Record._from_row",
        "",
        "    def set(",
        "        self,",
        "        *,",
    ]
    for column in table.columns:
        if column.is_primary_key:
            continue
        lines.append(
            f"        {column.attr_name}: {column.base_python_type} | "
            f"Expr[{column.base_python_type}] | UnsetType = UNSET,"
        )
    set_columns = [column for column in table.columns if not column.is_primary_key]
    lines.append(f"    ) -> {table.model_name}UpdateBuilder:")
    lines.extend(
        _emit_mapped_call("        return self._set_values", set_columns, indent="        ")
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
        lines.append(f"        {column.attr_name}: {column.base_python_type} | UnsetType = UNSET,")
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
            f"    def update(self) -> {table.model_name}UpdateBuilder:",
            f"        return {table.model_name}UpdateBuilder(self._adapter)",
            "",
            "    def delete(self) -> BaseDeleteQuery:",
            f"        return BaseDeleteQuery(self._adapter, table_name={quoted_table!r})",
        ]
    )
    return lines


def _emit_mapped_call(
    prefix: str,
    columns: list[ColumnIR],
    *,
    indent: str,
) -> list[str]:
    """Emit a call that maps Python attr names onto SQL column kwargs safely."""
    if not columns:
        return [f"{prefix}()"]
    needs_dict = any(column.attr_name != column.name for column in columns)
    if not needs_dict:
        lines = [f"{prefix}("]
        for column in columns:
            lines.append(f"{indent}    {column.name}={column.attr_name},")
        lines.append(f"{indent})")
        return lines
    lines = [f"{prefix}(**{{"]
    for column in columns:
        lines.append(f"{indent}    {column.name!r}: {column.attr_name},")
    lines.append(f"{indent}}})")
    return lines


def _quote_table(name: str, *, dialect: str) -> str:
    return _quote_ident(name, dialect=dialect)


def _quote_ident(name: str, *, dialect: str) -> str:
    if dialect in {"mysql", "tidb"}:
        return f"`{name}`"
    return f'"{name}"'
