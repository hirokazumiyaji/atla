"""Generated composite-PK ``get_many`` issues batched row-value IN queries."""

from __future__ import annotations

import importlib
import sqlite3
import sys
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any

import pytest

from atla.adapter.base import DBAdapter
from atla.dialect import SQLiteDialect
from atlagen.emitter import generate_files
from atlagen.ir import ColumnIR, SchemaIR, TableIR


def _column(name: str, *, pk: bool, py: str = "int") -> ColumnIR:
    return ColumnIR(
        name,
        name,
        py,
        py,
        py,
        None,
        nullable=False,
        schema_nullable=False,
        has_default=False,
        is_primary_key=pk,
        row_codec=None,
    )


class SqliteAdapter:
    """Minimal adapter backed by stdlib sqlite3 (validates the emitted SQL)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.dialect = SQLiteDialect()
        self.conn = conn
        self.queries: list[tuple[str, list[Any]]] = []

    async def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Sequence[Any]]:
        self.queries.append((sql, list(params)))
        return self.conn.execute(sql, list(params)).fetchall()

    async def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Any] | None:
        row: Sequence[Any] | None = self.conn.execute(sql, list(params)).fetchone()
        return row

    async def fetch_val(self, sql: str, params: Sequence[Any] = ()) -> Any:
        return None

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        return 0

    async def execute_many(self, sql: str, params_seq: Sequence[Sequence[Any]]) -> None:
        return None

    def transaction(self) -> AbstractAsyncContextManager[DBAdapter]:
        raise NotImplementedError


@pytest.mark.asyncio
async def test_composite_get_many_uses_row_value_in_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = TableIR(
        name="memberships",
        model_name="Membership",
        accessor_name="membership",
        columns=(
            _column("tenant_id", pk=True),
            _column("id", pk=True),
            _column("name", pk=False, py="str"),
        ),
        primary_key=("tenant_id", "id"),
        foreign_keys=(),
    )
    files = generate_files(SchemaIR(name="s", dialect="sqlite", tables=(table,)))
    pkg = tmp_path / "gen_composite_pk"
    pkg.mkdir()
    for rel, text in files.items():
        (pkg / rel).write_text(text)
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])
    try:
        query_mod = importlib.import_module("gen_composite_pk.query")
        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE TABLE memberships (tenant_id INTEGER, id INTEGER, name TEXT,"
            " PRIMARY KEY (tenant_id, id))"
        )
        conn.executemany(
            "INSERT INTO memberships VALUES (?, ?, ?)",
            [(t, i, f"m{t}-{i}") for t in (1, 2) for i in range(300)],
        )
        adapter = SqliteAdapter(conn)
        repo = query_mod.MembershipRepository(adapter)

        keys = [(1, 5), (2, 7), (9, 9)] + [(1, i) for i in range(10, 300)]
        rows = await repo.get_many(keys)

        assert {(r.tenant_id, r.id) for r in rows} == {(1, 5), (2, 7)} | {
            (1, i) for i in range(10, 300)
        }
        # 293 keys * 2 params => batch_size 250 => two queries, none per-key.
        assert len(adapter.queries) == 2
        sql, params = adapter.queries[0]
        assert ") IN (" in sql
        assert '("memberships"."tenant_id", "memberships"."id") IN ((?, ?)' in sql
        assert len(params) == 500
        assert await repo.get_many([]) == []
    finally:
        for name in [m for m in sys.modules if m.startswith("gen_composite_pk")]:
            del sys.modules[name]


def test_insert_many_builder_carries_raw_copy_table() -> None:
    table = TableIR(
        name="memberships",
        model_name="Membership",
        accessor_name="membership",
        columns=(
            _column("tenant_id", pk=True),
            _column("id", pk=True),
            _column("name", pk=False, py="str"),
        ),
        primary_key=("tenant_id", "id"),
        foreign_keys=(),
    )
    files = generate_files(SchemaIR(name="s", dialect="postgres", tables=(table,)))
    query_src = files["query.py"]

    assert "_COPY_TABLE = 'memberships'" in query_src
    assert "class MembershipInsertManyBuilder" in query_src
    # Raw name (not the quoted SQL identifier) is what asyncpg COPY needs.
    assert "_COPY_TABLE = '\"memberships\"'" not in query_src


def test_copy_table_mirrors_table_name_for_qualified_names() -> None:
    """COPY must target exactly what the generated INSERT targets."""
    table = TableIR(
        name="other.users",
        model_name="User",
        accessor_name="user",
        columns=(_column("id", pk=True), _column("email", pk=False, py="str")),
        primary_key=("id",),
        foreign_keys=(),
    )
    query_src = generate_files(SchemaIR(name="other", dialect="postgres", tables=(table,)))[
        "query.py"
    ]

    assert '    _TABLE_NAME = \'"other"."users"\'' in query_src
    # Raw dotted name: the asyncpg COPY adapter re-quotes each part itself.
    assert "    _COPY_TABLE = 'other.users'" in query_src


def test_copy_table_matches_sql_interpretation_of_dotted_names() -> None:
    """A dot is a schema separator everywhere, including in generated SQL.

    ``_quote_table`` splits on dots, so ``orders.v2`` is emitted as
    ``"orders"."v2"``; ``_COPY_TABLE`` keeps the raw dotted form and the COPY
    adapter splits it the same way, so both paths target one relation.
    """
    table = TableIR(
        name="orders.v2",
        model_name="Order",
        accessor_name="order",
        columns=(_column("id", pk=True),),
        primary_key=("id",),
        foreign_keys=(),
    )
    query_src = generate_files(SchemaIR(name="main", dialect="postgres", tables=(table,)))[
        "query.py"
    ]

    assert '    _TABLE_NAME = \'"orders"."v2"\'' in query_src
    assert "    _COPY_TABLE = 'orders.v2'" in query_src
