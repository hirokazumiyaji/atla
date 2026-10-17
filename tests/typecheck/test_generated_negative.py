"""Negative type tests: invalid generated-API usage fails pyright/mypy --strict."""

from __future__ import annotations

from pathlib import Path

from typecheck._harness import (
    FIXTURES,
    assert_typechecks_fail,
    generate_package,
    write_text,
    write_typecheck_project,
)

TYPECHECK_SCHEMA = FIXTURES / "typecheck_users.json"


def _setup(tmp_path: Path, source: str, *, filename: str) -> tuple[Path, Path]:
    package = tmp_path / "appdb"
    generate_package(package, schema=TYPECHECK_SCHEMA, dialect="postgres")
    app = write_text(tmp_path / filename, source)
    write_typecheck_project(tmp_path, package_dir=package, sources=[app])
    return package, app


def test_column_age_eq_str_is_rejected(tmp_path: Path) -> None:
    source = """\
from __future__ import annotations

from atla.expressions import Expr

from appdb import User


def _need_bool_expr(clause: Expr[bool]) -> None:
    raise NotImplementedError


_need_bool_expr(User.age == "invalid")
"""
    package, app = _setup(tmp_path, source, filename="neg_age_eq.py")
    assert_typechecks_fail(tmp_path, package, app, must_mention=("age", "str", "bool"))


def test_insert_missing_required_email_is_rejected(tmp_path: Path) -> None:
    source = """\
from __future__ import annotations

from typing import cast

from atla.adapter.base import DBAdapter

from appdb import AtlaClient


def bad(db: AtlaClient) -> None:
    # Missing required columns ``email`` and ``age``.
    _ = db.user.insert(name="Alice")


bad(AtlaClient(cast(DBAdapter, object())))
"""
    package, app = _setup(tmp_path, source, filename="neg_insert.py")
    assert_typechecks_fail(tmp_path, package, app, must_mention=("email", "insert", "argument"))


def test_update_set_age_str_is_rejected(tmp_path: Path) -> None:
    source = """\
from __future__ import annotations

from typing import cast

from atla.adapter.base import DBAdapter

from appdb import AtlaClient


def bad(db: AtlaClient) -> None:
    _ = db.user.update().set(age="invalid")


bad(AtlaClient(cast(DBAdapter, object())))
"""
    package, app = _setup(tmp_path, source, filename="neg_update.py")
    assert_typechecks_fail(tmp_path, package, app, must_mention=("age", "str"))
