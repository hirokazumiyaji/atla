"""atlagen naming / singularize / reserved-attr tests."""

from __future__ import annotations

import pytest

from atlagen.config import ConfigError, _parse_table_overrides  # pyright: ignore[reportPrivateUsage]
from atlagen.naming import (
    singularize,
    singularize_identifier,
    table_accessor_name,
    table_model_name,
    to_snake_case,
)
from atlagen.parser import AtlasParseError, parse_atlas_json


@pytest.mark.parametrize(
    ("plural", "singular"),
    [
        ("movies", "movie"),
        ("ties", "tie"),
        ("pies", "pie"),
        ("lies", "lie"),
        ("dies", "die"),
        ("cookies", "cookie"),
        ("zombies", "zombie"),
        ("buses", "bus"),
        ("gases", "gas"),
        ("aliases", "alias"),
        ("campuses", "campus"),
        ("viruses", "virus"),
        ("bonuses", "bonus"),
        ("people", "person"),
        ("children", "child"),
        ("leaves", "leaf"),
        ("knives", "knife"),
        ("statuses", "status"),
        ("series", "series"),
        ("users", "user"),
        ("parties", "party"),
        ("classes", "class"),
        ("houses", "house"),
        ("addresses", "address"),
        ("analyses", "analysis"),
        ("crises", "crisis"),
        ("bases", "base"),
    ],
)
def test_singularize_cases(plural: str, singular: str) -> None:
    assert singularize(plural) == singular


def test_snake_case_and_accessor_escaping() -> None:
    from atlagen.naming import safe_attr_name

    assert to_snake_case("UserAccounts") == "user_accounts"
    assert singularize_identifier("UserAccounts") == "user_account"
    assert table_model_name("UserAccounts") == "UserAccount"
    assert table_accessor_name("UserAccounts") == "user_account"
    assert table_model_name("archives") == "Archive"  # trailing-s only, not -ves→f
    assert singularize("archives") == "archive"
    assert table_accessor_name("transactions") == "transaction_"
    assert table_accessor_name("classes") == "class_"
    assert table_accessor_name("returns") == "return_"
    # Client-reserved names apply to accessors only, not column attrs.
    assert safe_attr_name("transaction") == "transaction"
    assert safe_attr_name("adapter") == "adapter"
    assert table_model_name("movies") == "Movie"
    assert table_model_name("cookies") == "Cookie"
    assert table_model_name("buses") == "Bus"
    assert table_model_name("people") == "Person"
    assert singularize_identifier("user_accounts") == "user_account"
    # Singular forms of irregular plurals must not be mangled by trailing-s.
    assert singularize("thesis") == "thesis"
    assert singularize("bus") == "bus"
    assert singularize("diagnosis") == "diagnosis"
    assert singularize("oasis") == "oasis"
    assert singularize("alias") == "alias"
    assert singularize("campus") == "campus"
    assert table_model_name("thesis") == "Thesis"
    assert table_model_name("buses") == "Bus"
    assert table_model_name("bus") == "Bus"


def test_collision_raises() -> None:
    schema = {
        "schemas": [
            {
                "name": "main",
                "tables": [
                    {"name": "user", "columns": [{"name": "id", "type": "INTEGER"}]},
                    {"name": "users", "columns": [{"name": "id", "type": "INTEGER"}]},
                ],
            }
        ]
    }
    with pytest.raises(AtlasParseError, match="collides"):
        parse_atlas_json(schema, dialect="sqlite")


def test_duplicate_enum_class_raises() -> None:
    schema = {
        "schemas": [
            {
                "name": "main",
                "enums": [
                    {"name": "user_role", "values": ["a", "b"]},
                    {"name": "UserRole", "values": ["x", "y"]},
                ],
                "tables": [],
            }
        ]
    }
    with pytest.raises(AtlasParseError, match="enum class"):
        parse_atlas_json(schema, dialect="postgres")


def test_invalid_override_rejected() -> None:
    schema = {
        "schemas": [
            {
                "name": "main",
                "tables": [{"name": "users", "columns": [{"name": "id", "type": "INTEGER"}]}],
            }
        ]
    }
    with pytest.raises(AtlasParseError, match="model_name"):
        parse_atlas_json(
            schema,
            dialect="sqlite",
            table_overrides={"users": {"model_name": "User Model"}},
        )
    with pytest.raises(AtlasParseError, match="accessor_name"):
        parse_atlas_json(
            schema,
            dialect="sqlite",
            table_overrides={"users": {"accessor_name": "bad-name"}},
        )


def test_table_override_resolves_collision() -> None:
    schema = {
        "schemas": [
            {
                "name": "main",
                "tables": [
                    {"name": "user", "columns": [{"name": "id", "type": "INTEGER"}]},
                    {"name": "users", "columns": [{"name": "id", "type": "INTEGER"}]},
                ],
            }
        ]
    }
    ir = parse_atlas_json(
        schema,
        dialect="sqlite",
        table_overrides={
            "users": {"model_name": "Users", "accessor_name": "users"},
        },
    )
    names = {(t.name, t.model_name, t.accessor_name) for t in ir.tables}
    assert ("user", "User", "user") in names
    assert ("users", "Users", "users") in names


def test_default_name_colliding_with_reserved_symbol_is_rejected() -> None:
    schema = {
        "schemas": [
            {
                "name": "main",
                "tables": [
                    {"name": "atla_clients", "columns": [{"name": "id", "type": "INTEGER"}]}
                ],
            }
        ]
    }
    with pytest.raises(AtlasParseError, match="AtlaClient"):
        parse_atlas_json(schema, dialect="sqlite")


@pytest.mark.parametrize("table_name", ["columns", "table_refs", "decimals", "str_enums"])
def test_generated_import_colliding_model_name_is_rejected(table_name: str) -> None:
    schema = {
        "schemas": [
            {
                "name": "main",
                "tables": [{"name": table_name, "columns": [{"name": "id", "type": "INTEGER"}]}],
            }
        ]
    }
    with pytest.raises(AtlasParseError, match=r"collides|reserved"):
        parse_atlas_json(schema, dialect="sqlite")


def test_builder_symbol_collision_is_rejected() -> None:
    schema = {
        "schemas": [
            {
                "name": "main",
                "tables": [
                    {"name": "users", "columns": [{"name": "id", "type": "INTEGER"}]},
                    {
                        "name": "user_insert_builders",
                        "columns": [{"name": "id", "type": "INTEGER"}],
                    },
                ],
            }
        ]
    }
    with pytest.raises(AtlasParseError, match="InsertBuilder"):
        parse_atlas_json(schema, dialect="sqlite")


def test_unused_table_override_key_is_config_error() -> None:
    schema = {
        "schemas": [
            {
                "name": "main",
                "tables": [{"name": "users", "columns": [{"name": "id", "type": "INTEGER"}]}],
            }
        ]
    }
    with pytest.raises(ConfigError, match="nope"):
        parse_atlas_json(
            schema,
            dialect="sqlite",
            table_overrides={"users": {"model_name": "Person"}, "nope": {"model_name": "X"}},
        )


def test_unknown_override_keys_are_config_error() -> None:
    with pytest.raises(ConfigError, match="modle_name"):
        _parse_table_overrides({"users": {"modle_name": "Person"}})
