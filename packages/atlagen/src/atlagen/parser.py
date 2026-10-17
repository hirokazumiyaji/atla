"""Atlas JSON schema parser producing dialect-agnostic SchemaIR."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, cast

from atlagen.ir import (
    ColumnIR,
    EnumIR,
    ForeignKeyIR,
    IndexIR,
    SchemaIR,
    TableIR,
)
from atlagen.naming import (
    enum_class_name,
    mysql_inline_enum_class_name,
    mysql_inline_enum_name,
    safe_attr_name,
    table_accessor_name,
    table_model_name,
)

JsonDict = dict[str, Any]

_MYSQL_ENUM_RE = re.compile(r"^enum\s*\((.*)\)\s*$", re.IGNORECASE | re.DOTALL)
_SERIAL_TYPES = frozenset(
    {
        "serial",
        "serial2",
        "serial4",
        "serial8",
        "smallserial",
        "bigserial",
    }
)


class AtlasParseError(ValueError):
    """Raised when Atlas JSON cannot be normalized into SchemaIR."""


def parse_atlas_json(
    source: str | bytes | Mapping[str, Any],
    *,
    dialect: str,
    schema_name: str | None = None,
) -> SchemaIR:
    """Parse Atlas ``schema inspect --format '{{ json . }}'`` output into SchemaIR."""
    data = _load_json(source)
    schemas_value = data.get("schemas")
    if not isinstance(schemas_value, list) or not schemas_value:
        msg = "Atlas JSON must contain a non-empty 'schemas' array"
        raise AtlasParseError(msg)

    selected = _select_schema(
        _as_list(schemas_value),  # pyright: ignore[reportUnknownArgumentType]
        schema_name,
    )
    name = str(selected.get("name") or "public")

    enums: list[EnumIR] = []
    enums.extend(_parse_schema_enums(selected.get("enums")))

    tables: list[TableIR] = []
    for table_raw in _as_list(selected.get("tables")):
        table_obj = _as_dict(table_raw)
        if table_obj is None:
            continue
        table, inline_enums = _parse_table(table_obj, dialect=dialect)
        tables.append(table)
        enums.extend(inline_enums)

    return SchemaIR(
        name=name,
        dialect=dialect,
        enums=tuple(enums),
        tables=tuple(tables),
    )


def _load_json(source: str | bytes | Mapping[str, Any]) -> JsonDict:
    if isinstance(source, Mapping):
        return {str(key): value for key, value in source.items()}
    text = source.decode("utf-8") if isinstance(source, bytes) else source
    try:
        loaded: object = json.loads(text)
    except json.JSONDecodeError as exc:
        msg = f"invalid Atlas JSON: {exc}"
        raise AtlasParseError(msg) from exc
    root = _as_dict(loaded)
    if root is None:
        msg = "Atlas JSON root must be an object"
        raise AtlasParseError(msg)
    return root


def _select_schema(schemas: object, schema_name: str | None) -> JsonDict:
    typed: list[JsonDict] = []
    for schema in _as_list(schemas):
        obj = _as_dict(schema)
        if obj is not None:
            typed.append(obj)
    if not typed:
        msg = "Atlas JSON schemas entries must be objects"
        raise AtlasParseError(msg)
    if schema_name is None:
        return typed[0]
    for schema in typed:
        if schema.get("name") == schema_name:
            return schema
    msg = f"schema {schema_name!r} not found in Atlas JSON"
    raise AtlasParseError(msg)


def _parse_schema_enums(raw_enums: object) -> list[EnumIR]:
    result: list[EnumIR] = []
    for item in _as_list(raw_enums):
        obj = _as_dict(item)
        if obj is None:
            continue
        name = str(obj.get("name") or "")
        values = _string_tuple(obj.get("values"))
        if not name or not values:
            continue
        result.append(EnumIR(name=name, class_name=enum_class_name(name), values=values))
    return result


def _parse_table(table_raw: JsonDict, *, dialect: str) -> tuple[TableIR, list[EnumIR]]:
    table_name = str(table_raw.get("name") or "")
    if not table_name:
        msg = "table is missing a name"
        raise AtlasParseError(msg)

    model_name = table_model_name(table_name)
    accessor_name = table_accessor_name(table_name)
    primary_key = _parse_primary_key(table_raw.get("primary_key"))
    pk_set = set(primary_key)

    inline_enums: list[EnumIR] = []
    columns: list[ColumnIR] = []
    for column_raw in _as_list(table_raw.get("columns")):
        column_obj = _as_dict(column_raw)
        if column_obj is None:
            continue
        column, inline_enum = _parse_column(
            column_obj,
            dialect=dialect,
            table_name=table_name,
            model_name=model_name,
            primary_key_names=pk_set,
        )
        columns.append(column)
        if inline_enum is not None:
            inline_enums.append(inline_enum)

    foreign_keys: list[ForeignKeyIR] = []
    for fk_raw in _as_list(table_raw.get("foreign_keys")):
        fk_obj = _as_dict(fk_raw)
        if fk_obj is None:
            continue
        fk = _parse_foreign_key(fk_obj)
        if fk is not None:
            foreign_keys.append(fk)

    indexes: list[IndexIR] = []
    for idx_raw in _as_list(table_raw.get("indexes")):
        idx_obj = _as_dict(idx_raw)
        if idx_obj is None:
            continue
        idx = _parse_index(idx_obj)
        if idx is not None:
            indexes.append(idx)

    return (
        TableIR(
            name=table_name,
            model_name=model_name,
            accessor_name=accessor_name,
            columns=tuple(columns),
            primary_key=primary_key,
            foreign_keys=tuple(foreign_keys),
            indexes=tuple(indexes),
        ),
        inline_enums,
    )


def _parse_column(
    column_raw: JsonDict,
    *,
    dialect: str,
    table_name: str,
    model_name: str,
    primary_key_names: set[str],
) -> tuple[ColumnIR, EnumIR | None]:
    name = str(column_raw.get("name") or "")
    if not name:
        msg = "column is missing a name"
        raise AtlasParseError(msg)

    raw_type = _extract_raw_type(column_raw.get("type"))
    nullable = bool(column_raw.get("null", False))
    has_default = _column_has_default(column_raw, raw_type=raw_type)

    inline_enum: EnumIR | None = None
    enum_match = _MYSQL_ENUM_RE.match(raw_type.strip())
    if enum_match and dialect in {"mysql", "tidb"}:
        values = _parse_mysql_enum_values(enum_match.group(1))
        if values:
            enum_name = mysql_inline_enum_name(table_name, name)
            inline_enum = EnumIR(
                name=enum_name,
                class_name=mysql_inline_enum_class_name(model_name, name),
                values=values,
            )

    return (
        ColumnIR(
            name=name,
            attr_name=safe_attr_name(name),
            raw_type=raw_type,
            # Type mapping / codecs are resolved by atlagen.mapper (Issue #7).
            python_type="",
            base_python_type="",
            import_symbol=None,
            nullable=nullable,
            has_default=has_default,
            is_primary_key=name in primary_key_names,
            row_codec=None,
        ),
        inline_enum,
    )


def _extract_raw_type(type_value: object) -> str:
    if type_value is None:
        return ""
    if isinstance(type_value, str):
        return type_value
    type_obj = _as_dict(type_value)
    if type_obj is not None:
        for key in ("raw", "type", "name"):
            value = type_obj.get(key)
            if isinstance(value, str) and value:
                return value
        nested = type_obj.get("Type") or type_obj.get("T")
        nested_obj = _as_dict(nested)
        if nested_obj is not None:
            return _extract_raw_type(nested_obj)
    return str(type_value)


def _column_has_default(column_raw: JsonDict, *, raw_type: str) -> bool:
    if column_raw.get("default") is not None:
        return True

    if _normalize_type_name(raw_type) in _SERIAL_TYPES:
        return True

    if _truthy_attr(column_raw.get("auto_increment")):
        return True
    if column_raw.get("identity") is not None:
        return True

    for attr in _as_list(column_raw.get("attrs")):
        attr_obj = _as_dict(attr)
        if attr_obj is None:
            continue
        attr_type = str(attr_obj.get("type") or attr_obj.get("@type") or "").lower()
        if "auto_increment" in attr_type or attr_obj.get("auto_increment") is True:
            return True
        if "identity" in attr_type or attr_obj.get("identity") is not None:
            return True
        name = str(attr_obj.get("name") or "").lower()
        if name in {"auto_increment", "identity"}:
            return True

    type_obj = _as_dict(column_raw.get("type"))
    if type_obj is not None and _truthy_attr(type_obj.get("auto_increment")):
        return True
    return False


def _normalize_type_name(raw_type: str) -> str:
    base = raw_type.strip().lower()
    if "(" in base:
        base = base.split("(", 1)[0].strip()
    return base


def _truthy_attr(value: object) -> bool:
    if value is True:
        return True
    if isinstance(value, str) and value.lower() in {"true", "1", "yes"}:
        return True
    return False


def _parse_mysql_enum_values(body: str) -> tuple[str, ...]:
    values: list[str] = []
    current: list[str] = []
    in_quote = False
    quote_char = ""
    escape = False
    for char in body:
        if in_quote:
            if escape:
                current.append(char)
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == quote_char:
                values.append("".join(current))
                current = []
                in_quote = False
                quote_char = ""
                continue
            current.append(char)
            continue
        if char in {"'", '"'}:
            in_quote = True
            quote_char = char
    return tuple(values)


def _parse_primary_key(raw_pk: object) -> tuple[str, ...]:
    pk_obj = _as_dict(raw_pk)
    if pk_obj is None:
        return ()
    return _column_name_tuple(pk_obj.get("columns"))


def _parse_foreign_key(raw_fk: JsonDict) -> ForeignKeyIR | None:
    columns = _column_name_tuple(raw_fk.get("columns"))
    ref_columns = _column_name_tuple(raw_fk.get("ref_columns") or raw_fk.get("referenced_columns"))
    ref_table = _extract_ref_table(raw_fk)
    symbol = str(raw_fk.get("name") or raw_fk.get("symbol") or "")
    if not columns or not ref_columns or not ref_table:
        return None
    if not symbol:
        symbol = f"fk_{'_'.join(columns)}_{ref_table}"
    return ForeignKeyIR(
        symbol=symbol,
        columns=columns,
        ref_table=ref_table,
        ref_columns=ref_columns,
    )


def _extract_ref_table(raw_fk: JsonDict) -> str:
    for key in ("ref_table", "referenced_table", "table"):
        value = raw_fk.get(key)
        if isinstance(value, str) and value:
            return value
        value_obj = _as_dict(value)
        if value_obj is not None:
            name = value_obj.get("name")
            if isinstance(name, str) and name:
                return name
    ref = raw_fk.get("ref")
    if isinstance(ref, str) and ref:
        return ref.rsplit(".", 1)[-1]
    ref_obj = _as_dict(ref)
    if ref_obj is not None:
        table = ref_obj.get("table") or ref_obj.get("name")
        if isinstance(table, str) and table:
            return table.rsplit(".", 1)[-1]
        table_obj = _as_dict(table)
        if table_obj is not None:
            name = table_obj.get("name")
            if isinstance(name, str) and name:
                return name
    return ""


def _parse_index(raw_index: JsonDict) -> IndexIR | None:
    name = str(raw_index.get("name") or "")
    columns = _column_name_tuple(raw_index.get("columns"))
    if not columns:
        columns = _column_name_tuple(raw_index.get("parts"))
    if not columns:
        extracted: list[str] = []
        for part in _as_list(raw_index.get("parts")):
            part_obj = _as_dict(part)
            if part_obj is None:
                continue
            col = part_obj.get("column") or part_obj.get("c")
            if isinstance(col, str):
                extracted.append(col)
                continue
            col_obj = _as_dict(col)
            if col_obj is not None:
                col_name = col_obj.get("name")
                if isinstance(col_name, str):
                    extracted.append(col_name)
        columns = tuple(extracted)
    if not name or not columns:
        return None
    unique = bool(raw_index.get("unique", False))
    return IndexIR(name=name, columns=columns, unique=unique)


def _column_name_tuple(raw_columns: object) -> tuple[str, ...]:
    names: list[str] = []
    for item in _as_list(raw_columns):
        if isinstance(item, str):
            names.append(item)
            continue
        item_obj = _as_dict(item)
        if item_obj is None:
            continue
        name = item_obj.get("name")
        if isinstance(name, str) and name:
            names.append(name)
            continue
        column = item_obj.get("column") or item_obj.get("c")
        if isinstance(column, str) and column:
            names.append(column)
            continue
        column_obj = _as_dict(column)
        if column_obj is not None:
            col_name = column_obj.get("name")
            if isinstance(col_name, str) and col_name:
                names.append(col_name)
    return tuple(names)


def _string_tuple(raw_values: object) -> tuple[str, ...]:
    return tuple(str(value) for value in _as_list(raw_values))


def _as_dict(value: object) -> JsonDict | None:
    if isinstance(value, dict):
        return cast(JsonDict, value)
    return None


def _as_list(value: object) -> list[Any]:
    if not isinstance(value, list):
        return []
    out: list[Any] = []
    for item in value:  # pyright: ignore[reportUnknownVariableType]
        out.append(item)
    return out
