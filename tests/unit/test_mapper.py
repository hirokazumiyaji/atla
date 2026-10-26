"""Tests for SQL type -> Python type mapper and row codecs."""

from __future__ import annotations

from atlagen.ir import ColumnIR, EnumIR, SchemaIR, TableIR
from atlagen.mapper import map_column, map_schema, normalize_sql_type, resolve_sql_type


def _col(
    raw_type: str,
    *,
    nullable: bool = False,
    name: str = "c",
) -> ColumnIR:
    return ColumnIR(
        name=name,
        attr_name=name,
        raw_type=raw_type,
        python_type="",
        base_python_type="",
        import_symbol=None,
        nullable=nullable,
        schema_nullable=nullable,
        has_default=False,
        is_primary_key=False,
        row_codec=None,
    )


def test_normalize_strips_params_and_unsigned() -> None:
    assert normalize_sql_type("character varying(255)") == ("character varying", False, False)
    assert normalize_sql_type("double precision(53)") == ("double precision", False, False)
    assert normalize_sql_type("numeric(10, 2)") == ("numeric", False, False)
    assert normalize_sql_type("bigint(20) unsigned") == ("bigint", False, True)
    assert normalize_sql_type("timestamp(6) with time zone") == ("timestamptz", False, False)
    assert normalize_sql_type("text[]") == ("text", True, False)
    assert normalize_sql_type("tinyint(1)") == ("tinyint(1)", False, False)
    assert resolve_sql_type("double precision(53)", dialect="postgres").base_python_type == "float"
    assert resolve_sql_type("character varying(255)", dialect="postgres").base_python_type == "str"


def test_postgres_core_types() -> None:
    assert resolve_sql_type("bigint", dialect="postgres").base_python_type == "int"
    assert resolve_sql_type("double precision", dialect="postgres").base_python_type == "float"
    decimal = resolve_sql_type("numeric(10,2)", dialect="postgres")
    assert decimal.base_python_type == "Decimal"
    assert decimal.import_symbol == ("decimal", "Decimal")
    assert decimal.row_codec is None
    assert resolve_sql_type("boolean", dialect="postgres").base_python_type == "bool"
    assert resolve_sql_type("uuid", dialect="postgres").row_codec is None
    jsonb = resolve_sql_type("jsonb", dialect="postgres")
    assert jsonb.import_symbol == ("typing", "Any")
    assert jsonb.row_codec is not None
    assert "json.loads" in jsonb.row_codec
    assert resolve_sql_type("money", dialect="postgres").base_python_type == "str"
    array = resolve_sql_type("integer[]", dialect="postgres")
    assert array.base_python_type == "list[int]"


def test_postgres_enum_and_nullable() -> None:
    enums = {
        "user_role": EnumIR(
            name="user_role",
            class_name="UserRole",
            values=("admin", "member"),
        )
    }
    mapped = map_column(
        _col("user_role", nullable=True, name="role"),
        dialect="postgres",
        enums=enums,
    )
    assert mapped.base_python_type == "UserRole"
    assert mapped.python_type == "UserRole | None"
    assert mapped.import_symbol == (".enums", "UserRole")
    assert mapped.row_codec is not None
    assert "UserRole({value})" in mapped.row_codec


def test_mysql_tinyint_bool_and_inline_enum() -> None:
    bool_mapping = resolve_sql_type("tinyint(1)", dialect="mysql")
    assert bool_mapping.base_python_type == "bool"
    assert bool_mapping.row_codec is not None
    assert "bool(int(" in bool_mapping.row_codec
    assert "isinstance({value}, bool)" in bool_mapping.row_codec

    enums = {
        "users_status": EnumIR(
            name="users_status",
            class_name="UserStatusEnum",
            values=("active", "inactive"),
        )
    }
    mapped = map_column(
        _col("enum('active','inactive')", name="status"),
        dialect="mysql",
        enums=enums,
        table_name="users",
    )
    assert mapped.base_python_type == "UserStatusEnum"
    assert mapped.row_codec is not None
    assert "UserStatusEnum({value})" in mapped.row_codec

    js = resolve_sql_type("json", dialect="mysql")
    assert js.base_python_type == "Any"
    assert js.row_codec is not None
    assert "json.loads" in js.row_codec

    tm = resolve_sql_type("time", dialect="mysql")
    assert tm.base_python_type == "time"
    assert tm.import_symbol == ("datetime", "time")
    assert tm.row_codec is not None
    assert "timedelta" in tm.row_codec
    assert "datetime.min" in tm.row_codec


def test_sqlite_codecs() -> None:
    dt = resolve_sql_type("DATETIME", dialect="sqlite")
    assert dt.base_python_type == "datetime"
    assert dt.row_codec is not None
    assert "fromisoformat" in dt.row_codec

    tm = resolve_sql_type("time", dialect="sqlite")
    assert tm.base_python_type == "time"
    assert tm.import_symbol == ("datetime", "time")
    assert tm.row_codec is not None
    assert "fromisoformat" in tm.row_codec

    tmtz = resolve_sql_type("timetz", dialect="sqlite")
    assert tmtz.base_python_type == "time"
    assert tmtz.row_codec is not None
    assert "fromisoformat" in tmtz.row_codec

    uuid = resolve_sql_type("uuid", dialect="sqlite")
    assert uuid.row_codec is not None
    assert "UUID" in uuid.row_codec

    js = resolve_sql_type("json", dialect="sqlite")
    assert js.row_codec is not None
    assert "json.loads" in js.row_codec


def test_map_schema_fills_columns() -> None:
    schema = SchemaIR(
        name="public",
        dialect="postgres",
        enums=(EnumIR(name="user_role", class_name="UserRole", values=("admin", "member")),),
        tables=(
            TableIR(
                name="users",
                model_name="User",
                accessor_name="user",
                columns=(
                    _col("bigserial", name="id"),
                    _col("timestamptz", name="created_at"),
                    _col("user_role", name="role"),
                    _col("text", name="bio", nullable=True),
                ),
                primary_key=("id",),
                foreign_keys=(),
            ),
        ),
    )
    mapped = map_schema(schema)
    by_name = {column.name: column for column in mapped.tables[0].columns}
    assert by_name["id"].python_type == "int"
    assert by_name["created_at"].python_type == "datetime"
    assert by_name["created_at"].import_symbol == ("datetime", "datetime")
    assert by_name["role"].python_type == "UserRole"
    assert by_name["bio"].python_type == "str | None"


def test_postgres_enum_array() -> None:
    enums = {
        "status": EnumIR(name="status", class_name="Status", values=("on", "off")),
    }
    mapping = resolve_sql_type("status[]", dialect="postgres", enums=enums)
    assert mapping.base_python_type == "list[Status]"
    assert mapping.import_symbol == (".enums", "Status")
    assert mapping.row_codec is not None
    assert "Status(item)" in mapping.row_codec


def test_sqlite_array_preserves_element_codec() -> None:
    mapping = resolve_sql_type("uuid[]", dialect="sqlite")
    assert mapping.base_python_type == "list[UUID]"
    assert mapping.row_codec is not None
    assert "UUID(item)" in mapping.row_codec
    assert "for item in" in mapping.row_codec
    assert "or ()" not in mapping.row_codec
    assert "None if {value} is None" in mapping.row_codec


def test_cockroach_and_tidb_aliases() -> None:
    assert resolve_sql_type("int8", dialect="cockroach").base_python_type == "int"
    assert resolve_sql_type("varchar(64)", dialect="tidb").base_python_type == "str"
    assert resolve_sql_type("tinyint(1)", dialect="tidb").base_python_type == "bool"


def test_mysql_and_sqlite_codec_evaluation() -> None:
    import json
    from datetime import datetime, time, timedelta

    eval_scope = {
        "json": json,
        "datetime": datetime,
        "time": time,
        "timedelta": timedelta,
    }

    # MySQL JSON
    js_map = resolve_sql_type("json", dialect="mysql")
    assert js_map.row_codec is not None
    decode_json = eval(
        f"lambda value: {js_map.row_codec.replace('{value}', 'value')}",
        eval_scope,
    )
    assert decode_json('{"foo": 123}') == {"foo": 123}
    assert decode_json(b'{"foo": 123}') == {"foo": 123}
    assert decode_json({"already": "parsed"}) == {"already": "parsed"}

    # MySQL TIME (asyncmy yields timedelta)
    tm_map = resolve_sql_type("time", dialect="mysql")
    assert tm_map.row_codec is not None
    decode_time = eval(
        f"lambda value: {tm_map.row_codec.replace('{value}', 'value')}",
        eval_scope,
    )
    td = timedelta(hours=14, minutes=30, seconds=15, microseconds=500)
    assert decode_time(td) == time(14, 30, 15, 500)
    assert decode_time("14:30:15") == time(14, 30, 15)
    assert decode_time(time(14, 30, 15)) == time(14, 30, 15)
    # Negative timedelta / string should not raise OverflowError and remain as duration
    neg_td = timedelta(hours=-8, minutes=-30)
    assert decode_time(neg_td) == neg_td
    assert decode_time("-08:30:00") == "-08:30:00"
    # Durations beyond 24 hours cannot be represented by datetime.time and remain as duration
    over_24h_td = timedelta(hours=25, minutes=10)
    assert decode_time(over_24h_td) == over_24h_td
    assert decode_time("25:10:00") == "25:10:00"

    # SQLite TIME
    sq_tm_map = resolve_sql_type("time", dialect="sqlite")
    assert sq_tm_map.row_codec is not None
    decode_sq_time = eval(
        f"lambda value: {sq_tm_map.row_codec.replace('{value}', 'value')}",
        eval_scope,
    )
    assert decode_sq_time("14:30:15") == time(14, 30, 15)

