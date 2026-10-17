"""Emit Record dataclasses and Table column descriptors."""

from __future__ import annotations

from atlagen.ir import ColumnIR, SchemaIR, TableIR
from atlagen.naming import to_pascal_case


def emit_models(schema: SchemaIR, *, frozen: bool = True) -> str:
    imports: set[tuple[str, str]] = set()
    for table in schema.tables:
        for column in table.columns:
            if column.import_symbol is not None:
                imports.add(column.import_symbol)

    needs_json = any(
        column.row_codec and "json." in column.row_codec
        for table in schema.tables
        for column in table.columns
    )
    lines = [
        "from __future__ import annotations",
        "",
    ]
    if needs_json:
        lines.append("import json")
        lines.append("")
    lines.extend(
        [
            "from dataclasses import dataclass",
            "from typing import Any, Sequence",
            "from atla.expressions import Column",
        ]
    )
    for module, symbol in sorted(imports):
        lines.append(f"from {module} import {symbol}")
    lines.append("")

    if not schema.tables:
        lines.append("# No tables defined in schema.")
        lines.append("")
        return "\n".join(lines)

    for table in schema.tables:
        table_cls = f"{to_pascal_case(table.name)}Table"
        lines.extend(_emit_record(table, frozen=frozen))
        lines.append("")
        lines.extend(_emit_table(table, table_cls=table_cls))
        lines.append("")
        lines.append(f"{table.model_name} = {table_cls}()")
        lines.append("")
    return "\n".join(lines)


def _emit_record(table: TableIR, *, frozen: bool) -> list[str]:
    decorator = f"@dataclass(slots=True, frozen={frozen!s})"
    lines = [
        decorator,
        f"class {table.model_name}Record:",
    ]
    for column in table.columns:
        lines.append(f"    {column.attr_name}: {column.python_type}")
    lines.extend(
        [
            "",
            "    @classmethod",
            f"    def _from_row(cls, row: Sequence[Any]) -> {table.model_name}Record:",
        ]
    )
    if all(column.row_codec is None for column in table.columns):
        lines.append("        return cls(*row)")
    else:
        args = ", ".join(_row_expr(column, index) for index, column in enumerate(table.columns))
        lines.append(f"        return cls({args})")
    return lines


def _row_expr(column: ColumnIR, index: int) -> str:
    value = f"row[{index}]"
    if column.row_codec is None:
        return value
    expr = column.row_codec.replace("{value}", value)
    if column.nullable:
        return f"(None if {value} is None else ({expr}))"
    return f"({expr})"


def _emit_table(table: TableIR, *, table_cls: str) -> list[str]:
    slot_names = ", ".join(["'_alias'", *[repr(column.attr_name) for column in table.columns]])
    lines = [
        f"class {table_cls}:",
        f"    __slots__ = ({slot_names})",
        f"    __table_name__ = {table.name!r}",
        "",
        "    def __init__(self, alias: str | None = None) -> None:",
        "        self._alias = alias",
        f"        t = alias or {table.name!r}",
    ]
    for column in table.columns:
        lines.append(
            f"        self.{column.attr_name}: Column[{column.python_type}] = "
            f"Column(t, {column.name!r})"
        )
    lines.extend(
        [
            "",
            f"    def as_(self, alias: str) -> {table_cls}:",
            f'        """Return a {table_cls} bound to a SQL alias."""',
            f"        return {table_cls}(alias=alias)",
        ]
    )
    return lines
