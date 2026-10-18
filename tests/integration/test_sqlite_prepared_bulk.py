"""SQLite end-to-end coverage for prepared queries and bulk insert paths."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import aiosqlite
import pytest

from atla.adapter.aiosqlite import AiosqliteAdapter
from atla.expressions import Param
from atlagen.cli import main

SCHEMA = """
{
  "schemas": [{
    "name": "main",
    "tables": [{
      "name": "users",
      "columns": [
        {"name": "id", "type": "INTEGER", "null": false, "attrs": [{"name": "auto_increment"}]},
        {"name": "email", "type": "TEXT", "null": false},
        {"name": "name", "type": "TEXT", "null": false}
      ],
      "primary_key": {"columns": [{"name": "id"}]},
      "indexes": [{
        "name": "users_email_key",
        "unique": true,
        "parts": [{"column": {"name": "email"}}]
      }]
    }]
  }]
}
"""


async def _generate_and_connect(
    tmp_path: Path,
    load_appdb: Callable[[Path], Any],
) -> tuple[Any, aiosqlite.Connection]:
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(SCHEMA, encoding="utf-8")
    out = tmp_path / "appdb"
    assert (
        main(
            [
                "generate",
                "--dialect",
                "sqlite",
                "--schema",
                str(schema_path),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    appdb = load_appdb(tmp_path)
    conn = await aiosqlite.connect(tmp_path / "test.db")
    await conn.execute(
        """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL
        )
        """
    )
    await conn.commit()
    return appdb, conn


@pytest.mark.asyncio
async def test_sqlite_prepared_query_rebinds_against_real_db(
    tmp_path: Path,
    load_appdb: Callable[[Path], Any],
) -> None:
    appdb, conn = await _generate_and_connect(tmp_path, load_appdb)
    try:
        db = appdb.AtlaClient(AiosqliteAdapter(conn))
        User = appdb.User

        await db.user.insert_many(
            [
                {"email": "a@example.com", "name": "Alice"},
                {"email": "b@example.com", "name": "Bob"},
            ]
        ).execute()

        prepared = db.user.query().where(User.email == Param("email")).prepare()
        assert prepared.param_names == ("email",)
        assert prepared.sql.count("?") == 1

        alice = await prepared.first(email="a@example.com")
        assert alice is not None and alice.name == "Alice"
        bob = await prepared.first(email="b@example.com")
        assert bob is not None and bob.name == "Bob"
        assert await prepared.first(email="missing@example.com") is None

        prepared_ids = (
            db.user.query().where(User.id > Param[int]("after")).order_by(User.id.asc()).prepare()
        )
        rows = await prepared_ids.all(after=0)
        assert [row.email for row in rows] == ["a@example.com", "b@example.com"]
        rows = await prepared_ids.all(after=rows[0].id)
        assert [row.email for row in rows] == ["b@example.com"]
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_sqlite_insert_many_execute_many_against_real_db(
    tmp_path: Path,
    load_appdb: Callable[[Path], Any],
) -> None:
    appdb, conn = await _generate_and_connect(tmp_path, load_appdb)
    try:
        db = appdb.AtlaClient(AiosqliteAdapter(conn))

        submitted = await db.user.insert_many(
            [{"email": f"u{i}@example.com", "name": f"User {i}"} for i in range(50)]
        ).execute_many()

        assert submitted == 50
        count = await db.user.query().count()
        assert count == 50

        # RETURNING cannot ride the driver bulk path.
        with pytest.raises(ValueError, match="does not support RETURNING"):
            await (
                db.user.insert_many([{"email": "x@example.com", "name": "X"}])
                .returning()
                .execute_many()
            )

        # aiosqlite has no COPY capability.
        with pytest.raises(TypeError, match="CopyAdapter"):
            await db.user.insert_many([{"email": "y@example.com", "name": "Y"}]).copy()
    finally:
        await conn.close()
