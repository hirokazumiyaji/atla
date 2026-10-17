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
        has_default=False,
        is_primary_key=False,
        row_codec=None,
    )


def test_normalize_strips_params_and_unsigned() -> None:
    assert normalize_sql_type("character varying(255)") == ("character varying", False, False)
    assert normalize_sql_type("numeric(10, 2)") == ("numeric", False, False)
    assert normalize_sql_type("bigint(20) unsigned") == ("bigint", False, True)
    assert normalize_sql_type("timestamp(6) with time zone") == ("timestamptz", False, False)
    assert normalize_sql_type("text[]") == ("text", True, False)
    assert normalize_sql_type("tinyint(1)") == ("tinyint(1)", False, False)


def test_postgres_core_types() -> None:
    assert resolve_sql_type("bigint", dialect="postgres").base_python_type == "int"
    assert resolve_sql_type("double precision", dialect="postgres").base_python_type == "float"
    decimal = resolve_sql_type("numeric(10,2)", dialect="postgres")
    assert decimal.base_python_type == "Decimal"
    assert decimal.import_symbol == ("decimal", "Decimal")
    assert decimal.row_codec is None
    assert resolve_sql_type("boolean", dialect="postgres").base_python_type == "bool"
    assert resolve_sql_type("uuid", dialect="postgres").row_codec is None
    assert resolve_sql_type("jsonb", dialect="postgres").import_symbol == ("typing", "Any")
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
    assert mapped.row_codec is None  # asyncpg returns enum/str already usable


def test_mysql_tinyint_bool_and_inline_enum() -> None:
    bool_mapping = resolve_sql_type("tinyint(1)", dialect="mysql")
    assert bool_mapping.base_python_type == "bool"
    assert bool_mapping.row_codec == "bool({value})"

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
    assert mapped.row_codec == "UserStatusEnum({value})"


def test_sqlite_codecs() -> None:
    dt = resolve_sql_type("DATETIME", dialect="sqlite")
    assert dt.base_python_type == "datetime"
    assert dt.row_codec is not None
    assert "fromisoformat" in dt.row_codec

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


def test_cockroach_and_tidb_aliases() -> None:
    assert resolve_sql_type("int8", dialect="cockroach").base_python_type == "int"
    assert resolve_sql_type("varchar(64)", dialect="tidb").base_python_type == "str"
    assert resolve_sql_type("tinyint(1)", dialect="tidb").base_python_type == "bool"
