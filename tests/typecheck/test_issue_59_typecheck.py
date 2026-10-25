"""Typecheck tests for Issue #59: NOT NULL array codegen, prepared insert, and codecs."""

from __future__ import annotations

import json
from pathlib import Path

from typecheck._harness import (
    assert_typechecks_clean,
    generate_package,
    write_text,
    write_typecheck_project,
)

SCHEMA_JSON = {
    "schemas": [
        {
            "name": "public",
            "enums": [
                {
                    "name": "account_role",
                    "values": ["admin", "member", "guest"],
                }
            ],
            "tables": [
                {
                    "name": "accounts",
                    "columns": [
                        {
                            "name": "id",
                            "type": "bigserial",
                            "null": False,
                        },
                        {
                            "name": "roles",
                            "type": "account_role[]",
                            "null": False,
                        },
                        {
                            "name": "payloads",
                            "type": "jsonb[]",
                            "null": False,
                        },
                        {
                            "name": "tags",
                            "type": "text[]",
                            "null": False,
                        },
                        {
                            "name": "opt_roles",
                            "type": "account_role[]",
                            "null": True,
                        },
                    ],
                    "primary_key": {"columns": [{"name": "id"}]},
                    "indexes": [],
                    "foreign_keys": [],
                }
            ],
        }
    ]
}

APP_PY = """\
from __future__ import annotations

from typing import Any, assert_type

from atla import Param

from appdb import AccountRecord, AccountRole, AtlaClient


async def test_app(db: AtlaClient) -> None:
    # 1. Exercise NOT NULL array record and nullable array record
    row: list[Any] = [1, ["admin"], [{"meta": 1}], ["t1", "t2"], None]
    record = AccountRecord._from_row(row)  # pyright: ignore[reportPrivateUsage]
    assert_type(record.roles, list[AccountRole])
    assert_type(record.payloads, list[Any])
    assert_type(record.tags, list[str])
    assert_type(record.opt_roles, list[AccountRole] | None)
    assert record.opt_roles is None

    # 2. Exercise insert with Param (Prepared Query)
    insert_prepared = db.account.insert(
        roles=Param("roles"),
        payloads=Param("payloads"),
        tags=Param("tags"),
    ).prepare()
    _ = await insert_prepared.execute(
        roles=[AccountRole.ADMIN],
        payloads=[{"k": "v"}],
        tags=["tag1"],
    )

    # 3. Exercise normal insert
    _ = db.account.insert(
        roles=[AccountRole.MEMBER],
        payloads=[{"a": 1}],
        tags=["test"],
    )
"""


def test_not_null_arrays_and_prepared_insert_pass_strict_typecheck(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(SCHEMA_JSON), encoding="utf-8")
    package = tmp_path / "appdb"
    generate_package(package, schema=schema_path, dialect="postgres")
    app = write_text(tmp_path / "app.py", APP_PY)
    write_typecheck_project(tmp_path, package_dir=package, sources=[app])
    assert_typechecks_clean(tmp_path, app)
