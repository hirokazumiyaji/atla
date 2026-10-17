"""SQLite end-to-end integration using aiosqlite + generated client."""

from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest

from atla.adapter.aiosqlite import AiosqliteAdapter
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


@pytest.mark.asyncio
async def test_sqlite_insert_many_upsert_select(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    monkeypatch.syspath_prepend(str(tmp_path))
    from appdb import AtlaClient, User  # type: ignore[import-not-found]

    db_path = tmp_path / "test.db"
    conn = await aiosqlite.connect(db_path)
    await conn.execute("PRAGMA foreign_keys = ON")
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
    adapter = AiosqliteAdapter(conn)
    client = AtlaClient(adapter)

    await client.user.insert_many(
        [
            {"email": "a@example.com", "name": "Alice"},
            {"email": "b@example.com", "name": "Bob"},
        ]
    ).execute()

    await (
        client.user.insert(email="a@example.com", name="Alice2")
        .on_conflict("email")
        .do_update(name="Alice2")
        .execute()
    )

    projected = (
        await client.user.query().select(User.email, User.name).order_by(User.email.asc()).all()
    )
    assert projected == [("a@example.com", "Alice2"), ("b@example.com", "Bob")]

    rows = await client.user.query().where(User.email == "a@example.com").all()
    assert len(rows) == 1
    assert rows[0].name == "Alice2"
    await conn.close()

    # Re-open to prove writes were committed (not rolled back on close).
    conn2 = await aiosqlite.connect(db_path)
    try:
        adapter2 = AiosqliteAdapter(conn2)
        client2 = AtlaClient(adapter2)
        persisted = await client2.user.query().order_by(User.email.asc()).all()
        assert [(row.email, row.name) for row in persisted] == [
            ("a@example.com", "Alice2"),
            ("b@example.com", "Bob"),
        ]
    finally:
        await conn2.close()
