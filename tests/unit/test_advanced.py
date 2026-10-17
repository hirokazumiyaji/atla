"""Tests for insert_many, upsert, and projected select."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any, NamedTuple

import pytest

from atla.adapter.base import DBAdapter
from atla.advanced import BaseInsertManyQuery
from atla.dialect import MySQLDialect, PostgresDialect, SQLiteDialect
from atla.expressions import Column
from atla.query import BaseInsertQuery, BaseSelectQuery


class UserRecord(NamedTuple):
    id: int
    email: str

    @staticmethod
    def from_row(row: Sequence[Any]) -> UserRecord:
        return UserRecord(int(row[0]), str(row[1]))


class FakeAdapter:
    def __init__(self, dialect: Any = None) -> None:
        self.dialect = dialect or PostgresDialect()
        self.calls: list[tuple[str, str, list[Any]]] = []
        self.rows: list[Sequence[Any]] = []
        self.execute_count = 2

    async def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Sequence[Any]]:
        self.calls.append(("fetch_all", sql, list(params)))
        return list(self.rows)

    async def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Any] | None:
        self.calls.append(("fetch_one", sql, list(params)))
        return self.rows[0] if self.rows else None

    async def fetch_val(self, sql: str, params: Sequence[Any] = ()) -> Any:
        return 0

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        self.calls.append(("execute", sql, list(params)))
        return self.execute_count

    async def execute_many(self, sql: str, params_seq: Sequence[Sequence[Any]]) -> None:
        self.calls.append(("execute_many", sql, [list(p) for p in params_seq]))

    def transaction(self) -> AbstractAsyncContextManager[DBAdapter]:
        raise NotImplementedError


class UserQuery(BaseSelectQuery[UserRecord]):
    __slots__ = ()
    _TABLE_NAME = '"users"'
    _SELECT_COLS = '"users"."id", "users"."email"'
    _FROM_SQL = '"users"'
    _COLUMN_COUNT = 2
    _SELECT_SQL = 'SELECT "users"."id", "users"."email" FROM "users"'
    _COUNT_SQL = 'SELECT COUNT(*) FROM "users"'
    _ROW_MAPPER = UserRecord.from_row


class UserInsert(BaseInsertQuery[UserRecord]):
    __slots__ = ()
    _TABLE_NAME = '"users"'
    _RETURNING_COLS = '"id", "email"'
    _ROW_MAPPER = UserRecord.from_row


class UserInsertMany(BaseInsertManyQuery[UserRecord]):
    __slots__ = ()
    _RETURNING_COLS = '"id", "email"'
    _ROW_MAPPER = UserRecord.from_row


id_col: Column[int] = Column("users", "id")
email_col: Column[str] = Column("users", "email")


@pytest.mark.asyncio
async def test_insert_many_builds_multi_values() -> None:
    adapter = FakeAdapter()
    q = UserInsertMany(
        adapter,
        table_name='"users"',
        rows=(
            (("email", "a@example.com"), ("id", 1)),
            (("email", "b@example.com"), ("id", 2)),
        ),
    )
    sql, params = q.to_sql()
    assert sql.startswith('INSERT INTO "users" ("email", "id") VALUES')
    assert sql.count("VALUES") == 1
    assert params == ["a@example.com", 1, "b@example.com", 2]
    assert await q.execute() == 2


@pytest.mark.asyncio
async def test_upsert_on_conflict_do_update_postgres() -> None:
    adapter = FakeAdapter(PostgresDialect())
    sql, params = (
        UserInsert(adapter)
        .values(email="a@example.com", id=1)
        .on_conflict("email")
        .do_update(email="a@example.com")
        .to_sql()
    )
    assert 'ON CONFLICT ("email") DO UPDATE SET "email" = $3' in sql
    assert params == ["a@example.com", 1, "a@example.com"]


@pytest.mark.asyncio
async def test_upsert_on_conflict_do_nothing_sqlite() -> None:
    adapter = FakeAdapter(SQLiteDialect())
    sql, params = (
        UserInsert(adapter).values(email="a@example.com").on_conflict("email").do_nothing().to_sql()
    )
    assert 'ON CONFLICT ("email") DO NOTHING' in sql
    assert params == ["a@example.com"]


@pytest.mark.asyncio
async def test_upsert_on_duplicate_key_mysql() -> None:
    adapter = FakeAdapter(MySQLDialect())
    sql, params = (
        UserInsert(adapter)
        .values(email="a@example.com", id=1)
        .on_conflict()
        .do_update(email="b@example.com")
        .to_sql()
    )
    assert "ON DUPLICATE KEY UPDATE" in sql
    assert "`email` = %s" in sql
    assert params == ["a@example.com", 1, "b@example.com"]


@pytest.mark.asyncio
async def test_projected_select_types_and_sql() -> None:
    adapter = FakeAdapter()
    adapter.rows = [(1, "a@example.com")]
    rows = await UserQuery(adapter).select(id_col, email_col).all()
    assert rows == [(1, "a@example.com")]
    sql = adapter.calls[0][1]
    assert sql.startswith('SELECT "users"."id", "users"."email" FROM "users"')


@pytest.mark.asyncio
async def test_projected_select_quotes_reserved_column_names() -> None:
    adapter = FakeAdapter()
    order_col: Column[str] = Column("users", "order")
    sql, _params = UserQuery(adapter).select(order_col).to_sql()
    assert sql.startswith('SELECT "users"."order" FROM "users"')
