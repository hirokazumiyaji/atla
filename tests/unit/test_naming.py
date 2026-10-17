"""atlagen naming / singularize / reserved-attr tests."""

from __future__ import annotations

import pytest

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
    ],
)
def test_singularize_cases(plural: str, singular: str) -> None:
    assert singularize(plural) == singular


def test_snake_case_and_accessor_escaping() -> None:
    assert to_snake_case("UserAccounts") == "user_accounts"
    assert table_model_name("UserAccounts") == "UserAccount"
    assert table_accessor_name("UserAccounts") == "user_account"
    assert table_accessor_name("transactions") == "transaction_"
    assert table_accessor_name("classes") == "class_"
    assert table_accessor_name("returns") == "return_"
    assert table_model_name("movies") == "Movie"
    assert table_model_name("cookies") == "Cookie"
    assert table_model_name("buses") == "Bus"
    assert table_model_name("people") == "Person"
    assert singularize_identifier("user_accounts") == "user_account"


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
