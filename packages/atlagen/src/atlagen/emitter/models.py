"""Emit Record dataclasses and Table column descriptors."""

from __future__ import annotations

from atlagen.ir import ColumnIR, SchemaIR, TableIR


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
    needs_required = False
    needs_not_required = False
    for table in schema.tables:
        for column in table.columns:
            if not column.nullable and not column.has_default:
                needs_required = True
            else:
                needs_not_required = True
    typing_imports = ["Any", "Sequence", "TypedDict"]
    if needs_not_required:
        typing_imports.insert(1, "NotRequired")
    if needs_required:
        typing_imports.insert(1, "Required")
    lines.extend(
        [
            "from dataclasses import dataclass",
            f"from typing import {', '.join(typing_imports)}",
            "from atla.expressions import Column",
            "from atla.table import TableRef",
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
        table_cls = f"{table.model_name}Table"
        lines.extend(_emit_record(table, frozen=frozen))
        lines.append("")
        lines.extend(_emit_insert_param(table))
        lines.append("")
        lines.extend(_emit_table(table, table_cls=table_cls))
        lines.append("")
        lines.append(f"{table.model_name} = {table_cls}()")
        lines.append("")
    return "\n".join(lines)


def _emit_insert_param(table: TableIR) -> list[str]:
    """Emit ``{Model}InsertParam`` TypedDict for typed ``insert_many`` rows.

    Runtime ``insert_many`` still requires every row to use the same column set
    (omitting a NotRequired key on some rows raises ValueError).
    """
    lines = [
        f"class {table.model_name}InsertParam(TypedDict):",
        '    """Row shape for ``insert_many``.',
        "",
        "    All rows must share the same column set; omitting a NotRequired key",
        "    on only some rows raises ValueError at runtime.",
        '    """',
    ]
    if not table.columns:
        lines.append("    pass")
        return lines
    for column in table.columns:
        value_type = (column.python_type or column.base_python_type or "Any").strip() or "Any"
        required = not column.nullable and not column.has_default
        wrapper = "Required" if required else "NotRequired"
        lines.append(f"    {column.attr_name}: {wrapper}[{value_type}]")
    return lines


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
        # Always tuple-unpack so a single column becomes ``(c0,) = row``, not
        # ``c0 = row`` (which would bind the whole sequence).
        locals_ = ", ".join(f"c{index}" for index in range(len(table.columns)))
        lines.append(f"        ({locals_},) = row")
        args = ", ".join(_row_expr(column, index) for index, column in enumerate(table.columns))
        lines.append(f"        return cls({args})")
    return lines


def _decode_expr(column: ColumnIR, value: str) -> str:
    """Python expression decoding *value* (an expression string) for *column*.

    Shared by ``Record._from_row`` (``c<i>`` locals) and ``Column`` decoders (``v``).
    """
    if column.row_codec is None:
        return value
    expr = column.row_codec.replace("{value}", value)
    if column.nullable:
        return f"(None if {value} is None else ({expr}))"
    return f"({expr})"


def _row_expr(column: ColumnIR, index: int) -> str:
    return _decode_expr(column, f"c{index}")


def _decoder_lambda(column: ColumnIR) -> str | None:
    if column.row_codec is None:
        return None
    return f"(lambda v: {_decode_expr(column, 'v')})"


def _emit_table(table: TableIR, *, table_cls: str) -> list[str]:
    slot_names = ", ".join(repr(column.attr_name) for column in table.columns)
    column_names = ", ".join(repr(column.name) for column in table.columns)
    # O(1) LEFT JOIN null check is only safe for a single non-nullable PK.
    # Schema-nullable PKs (e.g. SQLite TEXT PRIMARY KEY) keep an empty tuple
    # so map_row_optional falls back to a full-row NULL scan.
    pk_cols = [
        (index, column) for index, column in enumerate(table.columns) if column.is_primary_key
    ]
    if len(pk_cols) == 1 and not pk_cols[0][1].schema_nullable:
        pk_indexes: tuple[int, ...] = (pk_cols[0][0],)
    else:
        pk_indexes = ()
    lines = [
        f"class {table_cls}(TableRef[{table.model_name}Record]):",
        f"    __slots__ = ({slot_names})",
        f"    __table_name__ = {table.name!r}",
        f"    _COLUMNS = ({column_names})",
        (
            f"    _ROW_MAPPER = {table.model_name}Record._from_row"
            "  # pyright: ignore[reportPrivateUsage]"
        ),
        f"    _PK_INDEXES = {pk_indexes!r}",
        "",
        "    def __init__(self, alias: str | None = None) -> None:",
        "        super().__init__(alias)",
        # Column() quotes a single table identifier; use the unqualified table name.
        f"        t = alias or {table.name.rsplit('.', 1)[-1]!r}",
    ]
    for column in table.columns:
        decoder = _decoder_lambda(column)
        if decoder is None:
            lines.append(
                f"        self.{column.attr_name}: Column[{column.python_type}] = "
                f"Column(t, {column.name!r})"
            )
        else:
            lines.append(
                f"        self.{column.attr_name}: Column[{column.python_type}] = "
                f"Column(t, {column.name!r}, decoder={decoder})"
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
