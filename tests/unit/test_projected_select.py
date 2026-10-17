"""ProjectedSelectQuery codecs, aliases, aggregates, and scalars."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from typing import Any
from uuid import UUID

import pytest

from atla.adapter.base import DBAdapter, IsolationLevel
from atla.dialect import Dialect, SQLiteDialect
from atla.expressions import Column, count
from atla.query import BaseSelectQuery


class _FakeAdapter:
    def __init__(self, rows: list[Sequence[Any]] | None = None) -> None:
        self.rows = rows or []
        self.calls: list[tuple[str, str, list[Any]]] = []
        self.dialect: Dialect = SQLiteDialect()

    async def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Sequence[Any]]:
        self.calls.append(("fetch_all", sql, list(params)))
        return self.rows

    async def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Any] | None:
        self.calls.append(("fetch_one", sql, list(params)))
        return self.rows[0] if self.rows else None

    async def fetch_val(self, sql: str, params: Sequence[Any] = ()) -> Any:
        row = await self.fetch_one(sql, params)
        return None if row is None else row[0]

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        self.calls.append(("execute", sql, list(params)))
        return 1

    async def execute_many(self, sql: str, params_seq: Sequence[Sequence[Any]]) -> None:
        return None

    def transaction(
        self,
        *,
        isolation: IsolationLevel | None = None,
        readonly: bool = False,
        deferrable: bool = False,
    ) -> AbstractAsyncContextManager[DBAdapter]:
        raise NotImplementedError


def _map_item_row(row: Sequence[Any]) -> Sequence[Any]:
    return row


class ItemQuery(BaseSelectQuery[Any]):
    _FROM_SQL = '"items"'
    _SELECT_SQL = 'SELECT "items"."id" FROM "items"'
    _ROW_MAPPER = staticmethod(_map_item_row)


@pytest.mark.asyncio
async def test_projected_select_applies_column_decoder() -> None:
    adapter = _FakeAdapter(rows=[("12345678-1234-5678-1234-567812345678", "2024-01-02T03:04:05")])
    sku = Column[UUID](
        "items",
        "sku",
        decoder=lambda v: UUID(v) if isinstance(v, str) else v,
    )
    created = Column[datetime](
        "items",
        "created_at",
        decoder=lambda v: datetime.fromisoformat(v) if isinstance(v, str) else v,
    )
    rows = await ItemQuery(adapter).select(sku, created).all()
    assert rows == [
        (
            UUID("12345678-1234-5678-1234-567812345678"),
            datetime(2024, 1, 2, 3, 4, 5),
        )
    ]


def test_projected_select_uses_column_alias() -> None:
    adapter = _FakeAdapter()
    col = Column[int]("items", "id", alias="i")
    sql, _ = ItemQuery(adapter).select(col).to_sql()
    assert sql.startswith('SELECT "i"."id" FROM "items"')


def test_projected_select_compiles_aggregates() -> None:
    adapter = _FakeAdapter()
    age = Column[int]("items", "age")
    sql, params = ItemQuery(adapter).select(count(), age).to_sql()
    assert sql.startswith('SELECT COUNT(*), "items"."age" FROM "items"')
    assert params == []


@pytest.mark.asyncio
async def test_projected_select_scalars() -> None:
    adapter = _FakeAdapter(rows=[(1,), (2,), (3,)])
    id_col = Column[int]("items", "id")
    values: list[int] = await ItemQuery(adapter).select(id_col).scalars()
    assert values == [1, 2, 3]
    first: int | None = await ItemQuery(adapter).select(id_col).scalar_first()
    assert first == 1


@pytest.mark.asyncio
async def test_projected_select_scalar_one_and_multi_column_error() -> None:
    adapter = _FakeAdapter(rows=[(7,)])
    id_col = Column[int]("items", "id")
    only: int = await ItemQuery(adapter).select(id_col).scalar_one()
    assert only == 7
    with pytest.raises(ValueError, match="single-column"):
        await ItemQuery(adapter).select(id_col, id_col).scalars()  # type: ignore[misc]
