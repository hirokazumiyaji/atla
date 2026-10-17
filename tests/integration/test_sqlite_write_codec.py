"""SQLite round-trip for UUID / Decimal / datetime / JSON / StrEnum bind encoding."""

from __future__ import annotations

import importlib
import sys
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from uuid import UUID

import aiosqlite
import pytest

from atla.adapter.aiosqlite import AiosqliteAdapter
from atlagen.cli import main

SCHEMA = """
{
  "schemas": [{
    "name": "main",
    "tables": [{
      "name": "items",
      "columns": [
        {"name": "id", "type": "INTEGER", "null": false, "attrs": [{"name": "auto_increment"}]},
        {"name": "sku", "type": "UUID", "null": false},
        {"name": "price", "type": "DECIMAL", "null": false},
        {"name": "created_at", "type": "DATETIME", "null": false},
        {"name": "ship_on", "type": "DATE", "null": false},
        {"name": "meta", "type": "JSON", "null": false},
        {"name": "status", "type": "TEXT", "null": false}
      ],
      "primary_key": {"columns": [{"name": "id"}]}
    }]
  }]
}
"""


class Status(StrEnum):
    OPEN = "open"
    DONE = "done"


@pytest.mark.asyncio
async def test_sqlite_write_codec_round_trip(
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
    # Other generated-client tests also use the package name ``appdb``.
    for name in [m for m in sys.modules if m == "appdb" or m.startswith("appdb.")]:
        monkeypatch.delitem(sys.modules, name)
    importlib.invalidate_caches()
    from appdb import AtlaClient, Item  # type: ignore[import-not-found]

    db_path = tmp_path / "codec.db"
    conn = await aiosqlite.connect(db_path)
    await conn.execute(
        """
        CREATE TABLE items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sku TEXT NOT NULL,
            price TEXT NOT NULL,
            created_at TEXT NOT NULL,
            ship_on TEXT NOT NULL,
            meta TEXT NOT NULL,
            status TEXT NOT NULL
        )
        """
    )
    await conn.commit()
    client = AtlaClient(AiosqliteAdapter(conn))

    sku = UUID("12345678-1234-5678-1234-567812345678")
    price = Decimal("19.99")
    created = datetime(2024, 3, 4, 5, 6, 7)
    ship = date(2024, 3, 10)
    meta = {"tags": ["a", "b"], "n": 1}

    await client.item.insert_many(
        [
            {
                "sku": sku,
                "price": price,
                "created_at": created,
                "ship_on": ship,
                "meta": meta,
                "status": Status.OPEN,
            }
        ]
    ).execute()

    row = await client.item.query().where(Item.sku == sku).one()
    assert row.sku == sku
    assert row.price == price
    assert row.created_at == created
    assert row.ship_on == ship
    assert row.meta == meta
    assert row.status == "open"

    await (
        client.item.update()
        .set(meta={"tags": ["c"]}, status=Status.DONE, price=Decimal("20.00"))
        .where(Item.sku == sku)
        .execute()
    )
    updated = await client.item.query().where(Item.sku == sku).one()
    assert updated.meta == {"tags": ["c"]}
    assert updated.status == "done"
    assert updated.price == Decimal("20.00")
    await conn.close()
