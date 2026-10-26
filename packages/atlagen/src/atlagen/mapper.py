"""SQL type -> Python type and row-codec resolution for code generation."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from atlagen.ir import ColumnIR, EnumIR, SchemaIR, TableIR
from atlagen.naming import mysql_inline_enum_name

_MYSQL_ENUM_RE = re.compile(r"^enum\s*\(", re.IGNORECASE)

_NATIVE_DATETIME_DIALECTS = frozenset({"postgres", "cockroach", "mysql", "tidb"})
_NATIVE_UUID_DIALECTS = frozenset({"postgres", "cockroach"})
_NATIVE_BOOL_DIALECTS = frozenset({"postgres", "cockroach", "sqlite"})
# asyncpg returns json/jsonb as str unless set_type_codec is registered; asyncmy
# similarly yields text. Always apply the json.loads codec; drivers that already
# return dict/list pass through unchanged via the isinstance(str/bytes) guard.
_NATIVE_JSON_DIALECTS: frozenset[str] = frozenset()
_NATIVE_DECIMAL_DIALECTS = frozenset({"postgres", "cockroach"})

_JSON_ROW_CODEC = (
    "json.loads({value}) if isinstance({value}, (str, bytes, bytearray)) else {value}"
)


@dataclass(slots=True, frozen=True)
class TypeMapping:
    """Resolved Python typing metadata for a SQL column type."""

    base_python_type: str
    import_symbol: tuple[str, str] | None = None
    row_codec: str | None = None


def map_schema(schema: SchemaIR) -> SchemaIR:
    """Fill ``python_type`` / ``import_symbol`` / ``row_codec`` on every column."""
    enum_by_name: dict[str, EnumIR] = {}
    for enum in schema.enums:
        enum_by_name[enum.name.lower()] = enum
        enum_by_name.setdefault(enum.name.rsplit(".", 1)[-1].lower(), enum)
    tables = tuple(
        _map_table(table, dialect=schema.dialect, enums=enum_by_name) for table in schema.tables
    )
    return replace(schema, tables=tables)


def map_column(
    column: ColumnIR,
    *,
    dialect: str,
    enums: dict[str, EnumIR] | None = None,
    table_name: str | None = None,
) -> ColumnIR:
    """Resolve Python types and row codec for a single column."""
    mapping = resolve_sql_type(
        column.raw_type,
        dialect=dialect,
        enums=enums or {},
        table_name=table_name,
        column_name=column.name,
    )
    python_type = (
        f"{mapping.base_python_type} | None" if column.nullable else mapping.base_python_type
    )
    return replace(
        column,
        python_type=python_type,
        base_python_type=mapping.base_python_type,
        import_symbol=mapping.import_symbol,
        row_codec=mapping.row_codec,
    )


def resolve_sql_type(
    raw_type: str,
    *,
    dialect: str,
    enums: dict[str, EnumIR] | None = None,
    table_name: str | None = None,
    column_name: str | None = None,
) -> TypeMapping:
    """Map a raw Atlas SQL type string to Python typing metadata."""
    enums = enums or {}
    normalized, is_array, is_unsigned = normalize_sql_type(raw_type)
    dialect_key = dialect.lower()

    mapping: TypeMapping | None = None
    if _MYSQL_ENUM_RE.match(raw_type.strip()) and dialect_key in {"mysql", "tidb"}:
        if table_name and column_name:
            enum_name = mysql_inline_enum_name(table_name, column_name).lower()
            enum = enums.get(enum_name)
            if enum is not None:
                mapping = _enum_mapping(enum, dialect=dialect_key)

    if mapping is None:
        normalized_key = normalized.lower()
        enum = enums.get(normalized_key)
        if enum is None and "." in normalized_key:
            enum = enums.get(normalized_key.rsplit(".", 1)[-1])
        if enum is not None:
            mapping = _enum_mapping(enum, dialect=dialect_key)

    if mapping is None:
        mapping = _map_normalized_type(
            normalized,
            dialect=dialect_key,
            is_unsigned=is_unsigned,
            raw_type=raw_type,
        )
    if is_array:
        array_codec: str | None = None
        if mapping.row_codec is not None:
            element_expr = mapping.row_codec.replace("{value}", "item")
            # Preserve SQL NULL; do not coerce it to an empty list.
            array_codec = f"None if {{value}} is None else [({element_expr}) for item in {{value}}]"
        return TypeMapping(
            base_python_type=f"list[{mapping.base_python_type}]",
            import_symbol=mapping.import_symbol,
            row_codec=array_codec,
        )
    return mapping


def normalize_sql_type(raw_type: str) -> tuple[str, bool, bool]:
    """Return ``(normalized_base, is_array, is_unsigned)``."""
    text = " ".join(raw_type.strip().lower().split())
    is_unsigned = False
    if " unsigned" in f" {text} ":
        is_unsigned = True
        text = text.replace(" unsigned", "").strip()

    is_array = False
    if text.endswith("[]"):
        is_array = True
        text = text[:-2].rstrip()

    # Preserve tinyint(1) before stripping parameters.
    # Per generator-spec, MySQL/TiDB tinyint(1) maps to bool (boolean idiom).
    if text.startswith("tinyint(1)"):
        return "tinyint(1)", is_array, is_unsigned

    # timestamp(6) with time zone / time(3) with time zone
    if "with time zone" in text:
        if "timestamp" in text:
            return "timestamptz", is_array, is_unsigned
        if text.startswith("time") or " time " in f" {text} ":
            return "timetz", is_array, is_unsigned
    if "without time zone" in text:
        if "timestamp" in text:
            return "timestamp", is_array, is_unsigned
        return "time", is_array, is_unsigned

    # Match multi-word SQL types before stripping "(...)" modifiers.
    for prefix, canonical in (
        ("double precision", "double precision"),
        ("character varying", "character varying"),
        ("bit varying", "bit varying"),
    ):
        if text == prefix or text.startswith(f"{prefix}(") or text.startswith(f"{prefix} ("):
            return canonical, is_array, is_unsigned

    if "(" in text:
        text = text.split("(", 1)[0].rstrip()

    aliases = {
        "character": "char",
    }
    text = aliases.get(text, text)
    return text, is_array, is_unsigned


def _map_table(table: TableIR, *, dialect: str, enums: dict[str, EnumIR]) -> TableIR:
    columns = tuple(
        map_column(
            column,
            dialect=dialect,
            enums=enums,
            table_name=table.name,
        )
        for column in table.columns
    )
    return replace(table, columns=columns)


def _enum_mapping(enum: EnumIR, *, dialect: str) -> TypeMapping:
    _ = dialect
    # Drivers may return plain strings (incl. asyncpg enums); wrap into StrEnum.
    codec = (
        f"{enum.class_name}({{value}}) "
        f"if not isinstance({{value}}, {enum.class_name}) else {{value}}"
    )
    return TypeMapping(
        base_python_type=enum.class_name,
        import_symbol=(".enums", enum.class_name),
        row_codec=codec,
    )


def _map_normalized_type(
    normalized: str,
    *,
    dialect: str,
    is_unsigned: bool,
    raw_type: str,
) -> TypeMapping:
    _ = is_unsigned  # unsigned integers still map to Python int
    _ = raw_type

    if dialect in {"postgres", "cockroach"}:
        return _map_postgres_family(normalized, dialect=dialect)
    if dialect in {"mysql", "tidb"}:
        return _map_mysql_family(normalized, dialect=dialect)
    if dialect == "sqlite":
        return _map_sqlite(normalized)
    # Unknown dialect: fall back to postgres-like mapping.
    return _map_postgres_family(normalized, dialect=dialect)


def _map_postgres_family(normalized: str, *, dialect: str) -> TypeMapping:
    int_types = {
        "smallint",
        "integer",
        "bigint",
        "serial",
        "bigserial",
        "smallserial",
        "serial2",
        "serial4",
        "serial8",
        "int2",
        "int4",
        "int8",
        "rowid",
        "int",
    }
    if normalized in int_types:
        return TypeMapping("int")
    if normalized in {"real", "double precision", "float4", "float8", "float"}:
        return TypeMapping("float")
    if normalized in {"numeric", "decimal"}:
        return TypeMapping(
            "Decimal",
            import_symbol=("decimal", "Decimal"),
            row_codec=None if dialect in _NATIVE_DECIMAL_DIALECTS else "Decimal({value})",
        )
    if normalized == "money":
        # asyncpg MONEYOID returns formatted text (e.g. "$1,234.56"), not Decimal.
        return TypeMapping("str")
    if normalized in {"boolean", "bool"}:
        return TypeMapping("bool")
    if normalized in {
        "varchar",
        "character varying",
        "character",
        "char",
        "text",
        "citext",
        "inet",
        "cidr",
        "name",
    }:
        return TypeMapping("str")
    if normalized in {"bytea", "bytes"}:
        return TypeMapping("bytes")
    if normalized == "uuid":
        return TypeMapping(
            "UUID",
            import_symbol=("uuid", "UUID"),
            row_codec=None if dialect in _NATIVE_UUID_DIALECTS else "UUID({value})",
        )
    if normalized == "date":
        return TypeMapping(
            "date",
            import_symbol=("datetime", "date"),
            row_codec=None
            if dialect in _NATIVE_DATETIME_DIALECTS
            else "date.fromisoformat({value})",
        )
    if normalized in {"time", "timetz"}:
        return TypeMapping(
            "time",
            import_symbol=("datetime", "time"),
            row_codec=None
            if dialect in _NATIVE_DATETIME_DIALECTS
            else "time.fromisoformat({value})",
        )
    if normalized in {"timestamp", "timestamptz"}:
        return TypeMapping(
            "datetime",
            import_symbol=("datetime", "datetime"),
            row_codec=None
            if dialect in _NATIVE_DATETIME_DIALECTS
            else "datetime.fromisoformat({value})",
        )
    if normalized == "interval":
        return TypeMapping(
            "timedelta",
            import_symbol=("datetime", "timedelta"),
        )
    if normalized in {"json", "jsonb"}:
        return TypeMapping(
            "Any",
            import_symbol=("typing", "Any"),
            row_codec=None
            if dialect in _NATIVE_JSON_DIALECTS
            else "json.loads({value}) if isinstance({value}, (str, bytes, bytearray)) else {value}",
        )
    return TypeMapping("Any", import_symbol=("typing", "Any"))


def _map_mysql_family(normalized: str, *, dialect: str) -> TypeMapping:
    if normalized in {"tinyint(1)", "boolean", "bool"}:
        # asyncmy may return 0/1 ints; string "0"/"1" must not use Python bool("0")==True.
        return TypeMapping(
            "bool",
            row_codec=(
                None
                if dialect in _NATIVE_BOOL_DIALECTS
                else ("{value} if isinstance({value}, bool) else bool(int({value}))")
            ),
        )
    if normalized in {
        "tinyint",
        "smallint",
        "mediumint",
        "int",
        "integer",
        "bigint",
        "year",
    }:
        return TypeMapping("int")
    if normalized in {"float", "double", "real"}:
        return TypeMapping("float")
    if normalized in {"decimal", "numeric"}:
        return TypeMapping(
            "Decimal",
            import_symbol=("decimal", "Decimal"),
            row_codec="Decimal({value}) if not isinstance({value}, Decimal) else {value}",
        )
    if normalized in {
        "char",
        "varchar",
        "tinytext",
        "text",
        "mediumtext",
        "longtext",
    }:
        return TypeMapping("str")
    if normalized in {
        "binary",
        "varbinary",
        "tinyblob",
        "blob",
        "mediumblob",
        "longblob",
    }:
        return TypeMapping("bytes")
    if normalized == "date":
        return TypeMapping("date", import_symbol=("datetime", "date"))
    if normalized == "time":
        return TypeMapping(
            "time",
            import_symbol=("datetime", "time"),
            row_codec=(
                "{value} if isinstance({value}, time) else ("
                "(datetime.min + {value}).time() "
                "if isinstance({value}, timedelta) "
                "and timedelta(0) <= {value} < timedelta(days=1) else ("
                "{value} if isinstance({value}, timedelta) or ("
                "isinstance({value}, str) and ("
                "{value}.startswith('-') or "
                "(':' in {value} and int({value}.split(':', 1)[0]) >= 24)"
                ")"
                ") else time.fromisoformat(str({value}))))"
            ),
        )
    if normalized in {"datetime", "timestamp"}:
        return TypeMapping("datetime", import_symbol=("datetime", "datetime"))
    if normalized == "json":
        return TypeMapping(
            "Any",
            import_symbol=("typing", "Any"),
            row_codec=_JSON_ROW_CODEC,
        )
    return TypeMapping("Any", import_symbol=("typing", "Any"))


def _map_sqlite(normalized: str) -> TypeMapping:
    if normalized in {"integer", "int", "tinyint", "smallint", "mediumint", "bigint"}:
        return TypeMapping("int")
    if normalized in {"real", "float", "double", "double precision"}:
        return TypeMapping("float")
    if normalized in {"numeric", "decimal"}:
        return TypeMapping(
            "Decimal",
            import_symbol=("decimal", "Decimal"),
            row_codec="Decimal({value}) if not isinstance({value}, Decimal) else {value}",
        )
    if normalized in {"boolean", "bool"}:
        return TypeMapping(
            "bool",
            row_codec="bool({value}) if not isinstance({value}, bool) else {value}",
        )
    if normalized in {"text", "varchar", "char", "clob", "character varying", "character"}:
        return TypeMapping("str")
    if normalized == "blob":
        return TypeMapping("bytes")
    if normalized in {"datetime", "timestamp"}:
        return TypeMapping(
            "datetime",
            import_symbol=("datetime", "datetime"),
            row_codec=("datetime.fromisoformat({value}) if isinstance({value}, str) else {value}"),
        )
    if normalized == "date":
        return TypeMapping(
            "date",
            import_symbol=("datetime", "date"),
            row_codec=("date.fromisoformat({value}) if isinstance({value}, str) else {value}"),
        )
    if normalized in {"time", "timetz"}:
        return TypeMapping(
            "time",
            import_symbol=("datetime", "time"),
            row_codec=("time.fromisoformat({value}) if isinstance({value}, str) else {value}"),
        )
    if normalized == "uuid":
        return TypeMapping(
            "UUID",
            import_symbol=("uuid", "UUID"),
            row_codec="UUID({value}) if isinstance({value}, str) else {value}",
        )
    if normalized == "json":
        return TypeMapping(
            "Any",
            import_symbol=("typing", "Any"),
            row_codec=_JSON_ROW_CODEC,
        )
    return TypeMapping("Any", import_symbol=("typing", "Any"))
