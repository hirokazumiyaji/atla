"""Tests for Atlas JSON -> SchemaIR parser."""

from __future__ import annotations

from pathlib import Path

from atlagen.naming import safe_attr_name, table_model_name, unique_attr_names
from atlagen.parser import parse_atlas_json

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "atlas"


def test_table_model_name_singular_pascal() -> None:
    assert table_model_name("users") == "User"
    assert table_model_name("categories") == "Category"
    assert table_model_name("post_tags") == "PostTag"


def test_safe_attr_name_keywords() -> None:
    assert safe_attr_name("email") == "email"
    assert safe_attr_name("from") == "from_"
    assert safe_attr_name("class") == "class_"
    assert safe_attr_name("type") == "type_"
    assert safe_attr_name("user-name") == "user_name"
    assert safe_attr_name("1st") == "col_1st"
    assert unique_attr_names(["user-name", "user_name"]) == ["user_name", "user_name_2"]


def test_parse_postgres_fixture() -> None:
    schema = parse_atlas_json(
        (FIXTURES / "postgres_users.json").read_text(),
        dialect="postgres",
    )
    assert schema.name == "public"
    assert schema.dialect == "postgres"
    assert len(schema.enums) == 1
    assert schema.enums[0].name == "user_role"
    assert schema.enums[0].class_name == "UserRole"
    assert schema.enums[0].values == ("admin", "member", "guest")

    users = schema.tables[0]
    assert users.name == "users"
    assert users.model_name == "User"
    assert users.accessor_name == "user"
    assert users.primary_key == ("id",)

    by_name = {column.name: column for column in users.columns}
    assert by_name["id"].has_default is True  # bigserial
    assert by_name["id"].is_primary_key is True
    assert by_name["email"].has_default is False
    assert by_name["email"].nullable is False
    assert by_name["role"].has_default is True
    assert by_name["from"].attr_name == "from_"
    assert by_name["from"].nullable is True
    assert by_name["created_at"].has_default is True
    assert users.indexes[0].name == "users_email_key"
    assert users.indexes[0].unique is True
    assert users.indexes[0].columns == ("email",)

    posts = schema.tables[1]
    assert posts.model_name == "Post"
    assert posts.columns[0].has_default is True  # identity
    assert posts.foreign_keys[0].symbol == "posts_user_id_fkey"
    assert posts.foreign_keys[0].columns == ("user_id",)
    assert posts.foreign_keys[0].ref_table == "users"
    assert posts.foreign_keys[0].ref_columns == ("id",)


def test_parse_mysql_fixture() -> None:
    schema = parse_atlas_json(
        (FIXTURES / "mysql_users.json").read_text(),
        dialect="mysql",
        schema_name="app",
    )
    assert schema.name == "app"
    assert len(schema.enums) == 1
    assert schema.enums[0].name == "users_status"
    assert schema.enums[0].class_name == "UserStatusEnum"
    assert schema.enums[0].values == ("active", "inactive", "banned")

    users = next(table for table in schema.tables if table.name == "users")
    by_name = {column.name: column for column in users.columns}
    assert by_name["id"].has_default is True
    assert by_name["status"].has_default is True
    assert by_name["class"].attr_name == "class_"
    assert by_name["email"].raw_type == "varchar(255)"

    categories = next(table for table in schema.tables if table.name == "categories")
    assert categories.model_name == "Category"
    assert categories.columns[0].has_default is True


def test_primary_key_forces_non_nullable() -> None:
    schema = parse_atlas_json(
        {
            "schemas": [
                {
                    "name": "public",
                    "tables": [
                        {
                            "name": "users",
                            "columns": [
                                {"name": "id", "type": "int", "null": True},
                            ],
                            "primary_key": {"columns": [{"name": "id"}]},
                        }
                    ],
                }
            ]
        },
        dialect="postgres",
    )
    column = schema.tables[0].columns[0]
    assert column.is_primary_key is True
    assert column.nullable is False


def test_parse_foreign_key_string_ref_keeps_qualification() -> None:
    schema = parse_atlas_json(
        {
            "schemas": [
                {
                    "name": "public",
                    "tables": [
                        {
                            "name": "posts",
                            "columns": [
                                {"name": "id", "type": "int", "null": False},
                                {"name": "user_id", "type": "int", "null": False},
                            ],
                            "primary_key": {"columns": [{"name": "id"}]},
                            "foreign_keys": [
                                {
                                    "name": "posts_user_fk",
                                    "columns": [{"name": "user_id"}],
                                    "ref_columns": [{"name": "id"}],
                                    "ref": "other.users",
                                }
                            ],
                        }
                    ],
                }
            ]
        },
        dialect="postgres",
    )
    assert schema.tables[0].foreign_keys[0].ref_table == "other.users"


def test_parse_foreign_key_nested_ref_object() -> None:
    schema = parse_atlas_json(
        {
            "schemas": [
                {
                    "name": "public",
                    "tables": [
                        {
                            "name": "posts",
                            "columns": [
                                {"name": "id", "type": "int", "null": False},
                                {"name": "user_id", "type": "int", "null": False},
                            ],
                            "primary_key": {"columns": [{"name": "id"}]},
                            "foreign_keys": [
                                {
                                    "name": "posts_user_fk",
                                    "columns": [{"name": "user_id"}],
                                    "ref": {
                                        "table": {
                                            "name": "users",
                                            "schema": {"name": "public"},
                                        },
                                        "columns": [{"name": "id"}],
                                    },
                                }
                            ],
                        }
                    ],
                }
            ]
        },
        dialect="postgres",
    )
    fk = schema.tables[0].foreign_keys[0]
    assert fk.symbol == "posts_user_fk"
    assert fk.columns == ("user_id",)
    assert fk.ref_table == "public.users"
    assert fk.ref_columns == ("id",)


def test_parse_index_parts_variants() -> None:
    schema = parse_atlas_json(
        {
            "schemas": [
                {
                    "name": "public",
                    "tables": [
                        {
                            "name": "items",
                            "columns": [{"name": "id", "type": "int", "null": False}],
                            "primary_key": {"columns": [{"name": "id"}]},
                            "indexes": [
                                {
                                    "name": "idx_string_parts",
                                    "parts": ["id"],
                                },
                                {
                                    "name": "idx_c_object",
                                    "unique": True,
                                    "parts": [{"c": {"name": "id"}}],
                                },
                                {
                                    "name": "idx_expr_only",
                                    "parts": [{"x": "lower(id)"}],
                                },
                            ],
                        }
                    ],
                }
            ]
        },
        dialect="postgres",
    )
    indexes = {index.name: index for index in schema.tables[0].indexes}
    assert indexes["idx_string_parts"].columns == ("id",)
    assert indexes["idx_c_object"].columns == ("id",)
    assert indexes["idx_c_object"].unique is True
    assert "idx_expr_only" not in indexes


def test_parse_sqlite_fixture() -> None:
    schema = parse_atlas_json(
        (FIXTURES / "sqlite_users.json").read_text(),
        dialect="sqlite",
    )
    assert schema.name == "main"
    users = schema.tables[0]
    assert users.model_name == "User"
    by_name = {column.name: column for column in users.columns}
    assert by_name["id"].has_default is True
    assert by_name["type"].attr_name == "type_"
    assert by_name["created_at"].has_default is True
    assert users.indexes[0].columns == ("email",)
    assert users.indexes[0].unique is True
