"""Positive type tests: generated client APIs pass pyright/mypy --strict."""

from __future__ import annotations

from pathlib import Path

import pytest

from typecheck._harness import (
    DIALECT_SCHEMAS,
    FIXTURES,
    assert_typechecks_clean,
    generate_package,
    write_text,
    write_typecheck_project,
)

POSITIVE_APP = '''\
"""Positive usage of generated AtlaClient / User / UserRecord APIs."""

from __future__ import annotations

from typing import assert_type, cast

from atla.adapter.base import DBAdapter

from appdb import AtlaClient, User, UserRecord


async def exercise(db: AtlaClient) -> None:
    query = db.user.query().where(User.email == "alice@example.com")
    rows = await query.all()
    assert_type(rows, list[UserRecord])
    first = await query.first()
    assert_type(first, UserRecord | None)
    one = await query.one()
    assert_type(one, UserRecord)

    _ = db.user.insert(email="alice@example.com")
    _ = db.user.update().set(email="bob@example.com").where(User.id == 1)
    _ = db.user.delete().where(User.id == 1)

    async with db.transaction() as tx:
        assert_type(tx, AtlaClient)
        _ = await tx.user.query().where(User.id == 1).first()


async def _entry() -> None:
    await exercise(AtlaClient(cast(DBAdapter, object())))
'''


@pytest.mark.parametrize(("dialect", "fixture_name"), DIALECT_SCHEMAS)
def test_generated_package_passes_strict_typecheckers(
    tmp_path: Path,
    dialect: str,
    fixture_name: str,
) -> None:
    package = tmp_path / "appdb"
    generate_package(package, schema=FIXTURES / fixture_name, dialect=dialect)
    app = write_text(tmp_path / "positive_app.py", POSITIVE_APP)
    write_typecheck_project(tmp_path, package_dir=package, sources=[app])
    assert_typechecks_clean(tmp_path, package, app)
