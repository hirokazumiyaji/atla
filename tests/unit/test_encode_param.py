"""Dialect bind-parameter encoding (write codec)."""

from __future__ import annotations

import json
import warnings
from datetime import date, datetime, time
from decimal import Decimal
from enum import StrEnum
from uuid import UUID

import pytest

from atla.compiler import SQLCompiler
from atla.dialect import MySQLDialect, PostgresDialect, SQLiteDialect
from atla.expressions import Column, Literal


class Role(StrEnum):
    ADMIN = "admin"
    USER = "user"


@pytest.mark.parametrize(
    ("dialect", "value", "expected"),
    [
        (PostgresDialect(), UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"), UUID),
        (SQLiteDialect(), UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"), str),
        (MySQLDialect(), UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"), str),
        (PostgresDialect(), Decimal("1.25"), Decimal),
        (SQLiteDialect(), Decimal("1.25"), str),
        (MySQLDialect(), Decimal("1.25"), str),
        (PostgresDialect(), datetime(2024, 1, 2, 3, 4, 5), datetime),
        (SQLiteDialect(), datetime(2024, 1, 2, 3, 4, 5), str),
        (MySQLDialect(), datetime(2024, 1, 2, 3, 4, 5), datetime),
        (SQLiteDialect(), date(2024, 1, 2), str),
        (SQLiteDialect(), time(3, 4, 5), str),
        (PostgresDialect(), {"a": 1}, str),
        (SQLiteDialect(), [1, 2], str),
        (PostgresDialect(), Role.ADMIN, str),
        (SQLiteDialect(), 42, int),
        (SQLiteDialect(), "plain", str),
        (SQLiteDialect(), None, type(None)),
    ],
)
def test_encode_param_types(
    dialect: PostgresDialect | SQLiteDialect | MySQLDialect,
    value: object,
    expected: type,
) -> None:
    encoded = dialect.encode_param(value)
    assert type(encoded) is expected or (expected is type(None) and encoded is None)
    if isinstance(value, UUID) and expected is str:
        assert encoded == str(value)
    if isinstance(value, Decimal) and expected is str:
        assert encoded == "1.25"
    if isinstance(value, (dict, list)):
        assert json.loads(str(encoded)) == value
    if isinstance(value, Role):
        assert encoded == "admin"


def test_sqlite_datetime_bind_has_no_deprecation_warning() -> None:
    dialect = SQLiteDialect()
    compiler = SQLCompiler(dialect)
    col = Column[datetime]("events", "created_at")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compiler.visit(col == datetime(2024, 5, 6, 7, 8, 9))
        sql, params = compiler.build()
    assert not any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert params == ["2024-05-06T07:08:09"]
    assert "?" in sql


def test_literal_json_dict_is_serialized() -> None:
    compiler = SQLCompiler(PostgresDialect())
    compiler.visit(Literal({"role": "admin"}))
    _, params = compiler.build()
    assert params == ['{"role":"admin"}']
