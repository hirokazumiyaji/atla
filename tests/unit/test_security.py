"""Security hardening: injection guards, LIKE escaping, DSN masking."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from atla.compiler import SQLCompiler
from atla.dialect import PostgresDialect, SQLiteDialect
from atla.expressions import Column, FuncCall, Literal, OrderByClause, escape_like
from atlagen.inspect import InspectError, load_schema_json, mask_credentials


def test_order_by_rejects_injected_direction_and_nulls() -> None:
    col = Column[int]("users", "id")
    with pytest.raises(ValueError, match="direction"):
        OrderByClause(col, direction="ASC; DROP TABLE users")
    with pytest.raises(ValueError, match="nulls"):
        col.asc(nulls="LAST; --")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nulls"):
        OrderByClause(col, direction="ASC", nulls="LAST; --")


def test_contains_escapes_like_wildcards() -> None:
    name = Column[str]("users", "name")
    expr = name.contains("100%_real!x")
    assert isinstance(expr.right, Literal)
    assert expr.right.value == f"%{escape_like('100%_real!x')}%"
    assert expr.escape == "!"
    assert expr.op == "LIKE"

    dialect = PostgresDialect()
    compiler = SQLCompiler(dialect)
    compiler.visit(expr)
    sql, params = compiler.build()
    assert " ESCAPE '!'" in sql
    assert params == ["%100!%!_real!!x%"]


def test_icontains_compiles_with_escape() -> None:
    name = Column[str]("users", "name")
    dialect = SQLiteDialect()
    compiler = SQLCompiler(dialect)
    compiler.visit(name.icontains("A_B"))
    sql, params = compiler.build()
    assert "LOWER(" in sql
    assert " ESCAPE '!'" in sql
    assert params == ["%A!_B%"]


def test_contains_escape_clause_portable_on_mysql() -> None:
    from atla.dialect import MySQLDialect

    name = Column[str]("users", "name")
    compiler = SQLCompiler(MySQLDialect())
    compiler.visit(name.contains("a%b"))
    sql, params = compiler.build()
    assert " ESCAPE '!'" in sql
    assert "\\" not in sql.split("ESCAPE", 1)[1]
    assert params == ["%a!%b%"]


def test_quote_rejects_empty_and_nul() -> None:
    dialect = PostgresDialect()
    with pytest.raises(ValueError, match="empty"):
        dialect.quote("")
    with pytest.raises(ValueError, match="NUL"):
        dialect.quote("users\x00")


def test_func_call_rejects_unsafe_names() -> None:
    with pytest.raises(ValueError, match="function name"):
        FuncCall("count; drop", ())
    with pytest.raises(ValueError, match="function name"):
        FuncCall("pg_sleep\x00", ())


def test_mask_credentials_redacts_password() -> None:
    raw = "postgres://alice:s3cret-pass@db.example:5432/app"
    assert mask_credentials(raw) == "postgres://alice:***@db.example:5432/app"
    assert "s3cret" not in mask_credentials(f"atlas failed: could not connect to {raw}")


def test_inspect_error_masks_password_from_atlas_stderr() -> None:
    fake = MagicMock()
    fake.returncode = 1
    fake.stderr = "cannot connect to postgres://bob:topsecret@localhost/db"
    fake.stdout = ""

    with (
        patch("atlagen.inspect.shutil.which", return_value="/usr/bin/atlas"),
        patch("atlagen.inspect.subprocess.run", return_value=fake),
    ):
        with pytest.raises(InspectError) as exc_info:
            load_schema_json("postgres://bob:topsecret@localhost/db")
    message = str(exc_info.value)
    assert "topsecret" not in message
    assert "bob:***@" in message
