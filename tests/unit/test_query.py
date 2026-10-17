"""Tests for immutable BaseQuery / Mutation builders."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any, NamedTuple

import pytest

from atla.adapter.base import DBAdapter
from atla.dialect import PostgresDialect
from atla.errors import MultipleResultsError, NotFoundError, UnsafeMutationError
from atla.expressions import Column
from atla.query import (
    BaseDeleteQuery,
    BaseInsertQuery,
    BaseSelectQuery,
    BaseUpdateQuery,
)
from atla.types import UNSET


class UserRecord(NamedTuple):
    id: int
    email: str

    @staticmethod
    def from_row(row: Sequence[Any]) -> UserRecord:
        return UserRecord(id=int(row[0]), email=str(row[1]))


class FakeAdapter:
    def __init__(self) -> None:
        self.dialect = PostgresDialect()
        self.calls: list[tuple[str, str, list[Any]]] = []
        self.rows: list[Sequence[Any]] = [(1, "alice@example.com"), (2, "bob@example.com")]
        self.scalar: Any = 2
        self.execute_count = 1

    async def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Sequence[Any]]:
        self.calls.append(("fetch_all", sql, list(params)))
        rows = list(self.rows)
        offset = 0
        if " OFFSET $" in sql:
            index = int(sql.split(" OFFSET $", 1)[1].split(None, 1)[0]) - 1
            offset = int(params[index])
        elif " OFFSET ?" in sql or " OFFSET %s" in sql:
            offset = int(params[-1])
        if offset:
            rows = rows[offset:]
        if " LIMIT $" in sql:
            index = int(sql.split(" LIMIT $", 1)[1].split(None, 1)[0]) - 1
            rows = rows[: int(params[index])]
        elif " LIMIT ?" in sql or " LIMIT %s" in sql:
            limit_value = params[-2] if " OFFSET " in sql else params[-1]
            rows = rows[: int(limit_value)]
        return rows

    async def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Any] | None:
        self.calls.append(("fetch_one", sql, list(params)))
        return self.rows[0] if self.rows else None

    async def fetch_val(self, sql: str, params: Sequence[Any] = ()) -> Any:
        self.calls.append(("fetch_val", sql, list(params)))
        return self.scalar

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


class UserInsertBuilder(BaseInsertQuery[UserRecord]):
    __slots__ = ()
    _TABLE_NAME = '"users"'
    _RETURNING_COLS = '"id", "email"'
    _ROW_MAPPER = UserRecord.from_row


class UserUpdateBuilder(BaseUpdateQuery[UserRecord]):
    __slots__ = ()
    _TABLE_NAME = '"users"'
    _RETURNING_COLS = '"id", "email"'
    _ROW_MAPPER = UserRecord.from_row

    def set(self, **columns: Any) -> UserUpdateBuilder:
        return self._set_values(**columns)


id_col: Column[int] = Column("users", "id")
email_col: Column[str] = Column("users", "email")
age_col: Column[int] = Column("users", "age")


def test_select_where_is_immutable() -> None:
    adapter = FakeAdapter()
    base = UserQuery(adapter)
    active = base.where(id_col > 0)
    alice = base.where(email_col == "alice@example.com")

    base_sql, base_params = base.to_sql()
    active_sql, active_params = active.to_sql()
    alice_sql, alice_params = alice.to_sql()

    assert "WHERE" not in base_sql
    assert base_params == []
    assert "WHERE" in active_sql
    assert "WHERE" in alice_sql
    assert active_params == [0]
    assert alice_params == ["alice@example.com"]
    assert active_sql != alice_sql
    # Base query remains unchanged after branching.
    assert base.to_sql() == (base_sql, base_params)


@pytest.mark.asyncio
async def test_first_and_one_preserve_existing_limit() -> None:
    adapter = FakeAdapter()
    adapter.rows = [(1, "alice@example.com"), (2, "bob@example.com")]

    assert await UserQuery(adapter).limit(0).first() is None
    with pytest.raises(NotFoundError):
        await UserQuery(adapter).limit(0).one()

    # Existing limit(1) must not be widened to 2 (would raise MultipleResultsError).
    one = await UserQuery(adapter).limit(1).one()
    assert one == UserRecord(1, "alice@example.com")
    last_fetch = next(call for call in reversed(adapter.calls) if call[0] == "fetch_all")
    assert "LIMIT $1" in last_fetch[1]
    assert last_fetch[2] == [1]


@pytest.mark.asyncio
async def test_first_and_one_preserve_existing_offset() -> None:
    adapter = FakeAdapter()
    adapter.rows = [
        (1, "a@example.com"),
        (2, "b@example.com"),
        (3, "c@example.com"),
        (4, "d@example.com"),
        (5, "e@example.com"),
        (6, "f@example.com"),
    ]
    first = await UserQuery(adapter).offset(5).first()
    assert first == UserRecord(6, "f@example.com")
    last_fetch = next(call for call in reversed(adapter.calls) if call[0] == "fetch_all")
    assert "OFFSET $2" in last_fetch[1] or "OFFSET $1" in last_fetch[1]
    assert 5 in last_fetch[2]
    assert "LIMIT $" in last_fetch[1]

    one = await UserQuery(adapter).offset(5).limit(1).one()
    assert one == UserRecord(6, "f@example.com")


@pytest.mark.asyncio
async def test_count_ignores_limit_offset() -> None:
    adapter = FakeAdapter()
    q = UserQuery(adapter).where(id_col > 0).limit(10).offset(5)
    adapter.scalar = 42
    assert await q.count() == 42
    assert adapter.calls[-1][0] == "fetch_val"
    sql = adapter.calls[-1][1]
    assert "LIMIT" not in sql
    assert "OFFSET" not in sql
    assert "WHERE" in sql


def test_select_to_sql_order_limit_offset() -> None:
    adapter = FakeAdapter()
    q = UserQuery(adapter).where(id_col > 1).order_by(email_col.asc()).limit(10).offset(5)
    sql, params = q.to_sql()
    assert sql.startswith('SELECT "users"."id", "users"."email" FROM "users" WHERE ')
    assert "ORDER BY" in sql
    assert "LIMIT $2" in sql
    assert "OFFSET $3" in sql
    assert params == [1, 10, 5]


@pytest.mark.asyncio
async def test_select_terminal_methods() -> None:
    adapter = FakeAdapter()
    q = UserQuery(adapter)

    rows = await q.all()
    assert rows == [UserRecord(1, "alice@example.com"), UserRecord(2, "bob@example.com")]

    adapter.rows = [(1, "alice@example.com")]
    first = await q.first()
    assert first == UserRecord(1, "alice@example.com")
    one = await q.where(id_col == 1).one()
    assert one == UserRecord(1, "alice@example.com")

    adapter.rows = []
    with pytest.raises(NotFoundError):
        await UserQuery(adapter).one()

    adapter.rows = [(1, "a"), (2, "b")]
    with pytest.raises(MultipleResultsError):
        await UserQuery(adapter).one()

    adapter.scalar = 3
    assert await UserQuery(adapter).count() == 3
    assert await UserQuery(adapter).exists() is True
    adapter.rows = []
    assert await UserQuery(adapter).exists() is False


def test_insert_skips_unset_and_builds_sql() -> None:
    adapter = FakeAdapter()
    q = UserInsertBuilder(adapter).values(
        email="alice@example.com",
        id=UNSET,
        name="alice",
    )
    sql, params = q.to_sql()
    # Bare kwargs are quoted exactly once (not ""email"").
    assert sql == 'INSERT INTO "users" ("email", "name") VALUES ($1, $2)'
    assert '""email""' not in sql
    assert params == ["alice@example.com", "alice"]


def test_insert_returning_requires_support() -> None:
    adapter = FakeAdapter()
    sql, _ = UserInsertBuilder(adapter).values(email="a@example.com").returning().to_sql()
    assert sql.endswith('RETURNING "id", "email"')


@pytest.mark.asyncio
async def test_insert_execute_and_one() -> None:
    adapter = FakeAdapter()
    adapter.rows = [(1, "alice@example.com")]
    builder = UserInsertBuilder(adapter).values(email="a@example.com").returning()
    assert await builder.execute() == 1
    record = await builder.one()
    assert record == UserRecord(1, "alice@example.com")


@pytest.mark.asyncio
async def test_insert_one_raises_on_multiple_returning_rows() -> None:
    adapter = FakeAdapter()
    adapter.rows = [(1, "a@example.com"), (2, "b@example.com")]
    builder = UserInsertBuilder(adapter).values(email="a@example.com").returning()
    with pytest.raises(MultipleResultsError):
        await builder.one()


@pytest.mark.asyncio
async def test_update_returning_all_first_one() -> None:
    adapter = FakeAdapter()
    adapter.rows = [(1, "a@example.com"), (2, "b@example.com")]
    q = UserUpdateBuilder(adapter).set(email="x").where(id_col > 0).returning()
    rows = await q.all()
    assert rows == [UserRecord(1, "a@example.com"), UserRecord(2, "b@example.com")]
    assert await q.first() == UserRecord(1, "a@example.com")
    with pytest.raises(MultipleResultsError):
        await q.one()
    adapter.rows = [(1, "a@example.com")]
    assert await q.one() == UserRecord(1, "a@example.com")


class UserDeleteBuilder(BaseDeleteQuery[UserRecord]):
    __slots__ = ()
    _RETURNING_COLS = '"id", "email"'
    _ROW_MAPPER = UserRecord.from_row


def test_delete_returning_sql() -> None:
    adapter = FakeAdapter()
    sql, params = (
        UserDeleteBuilder(adapter, table_name='"users"').where(id_col == 1).returning().to_sql()
    )
    assert sql == 'DELETE FROM "users" WHERE "users"."id" = $1 RETURNING "id", "email"'
    assert params == [1]


@pytest.mark.asyncio
async def test_delete_returning_one() -> None:
    adapter = FakeAdapter()
    adapter.rows = [(1, "a@example.com")]
    record = (
        await UserDeleteBuilder(adapter, table_name='"users"').where(id_col == 1).returning().one()
    )
    assert record == UserRecord(1, "a@example.com")


def test_update_accepts_expr_and_literal() -> None:
    adapter = FakeAdapter()
    q = UserUpdateBuilder(adapter).set(email="new@example.com", age=age_col + 1).where(id_col == 1)
    sql, params = q.to_sql()
    assert 'UPDATE "users" SET "email" = $1, "age" = "users"."age" + $2 WHERE' in sql
    assert params == ["new@example.com", 1, 1]


def test_update_without_where_raises() -> None:
    adapter = FakeAdapter()
    q = UserUpdateBuilder(adapter).set(email="x@example.com")
    with pytest.raises(UnsafeMutationError):
        q.to_sql()


def test_update_allow_full_table() -> None:
    adapter = FakeAdapter()
    sql, params = UserUpdateBuilder(adapter).set(email="x@example.com").allow_full_table().to_sql()
    assert sql == 'UPDATE "users" SET "email" = $1'
    assert params == ["x@example.com"]


def test_delete_without_where_raises() -> None:
    adapter = FakeAdapter()
    q = BaseDeleteQuery(adapter, table_name='"users"')
    with pytest.raises(UnsafeMutationError):
        q.to_sql()


def test_delete_allow_full_table_and_where() -> None:
    adapter = FakeAdapter()
    full_sql, full_params = (
        BaseDeleteQuery(adapter, table_name='"users"').allow_full_table().to_sql()
    )
    assert full_sql == 'DELETE FROM "users"'
    assert full_params == []

    sql, params = BaseDeleteQuery(adapter, table_name='"users"').where(id_col == 9).to_sql()
    assert sql.startswith('DELETE FROM "users" WHERE ')
    assert params == [9]


def test_update_delete_where_is_immutable() -> None:
    adapter = FakeAdapter()
    base_update = UserUpdateBuilder(adapter).set(email="a")
    with pytest.raises(UnsafeMutationError):
        base_update.to_sql()
    u1_sql, u1_params = base_update.where(id_col == 1).to_sql()
    u2_sql, u2_params = base_update.where(id_col == 2).to_sql()
    assert u1_params == ["a", 1]
    assert u2_params == ["a", 2]
    assert u1_params != u2_params
    assert u1_sql == u2_sql  # placeholders match; bind values diverge
    with pytest.raises(UnsafeMutationError):
        base_update.to_sql()

    base_delete = BaseDeleteQuery(adapter, table_name='"users"')
    with pytest.raises(UnsafeMutationError):
        base_delete.to_sql()
    d1_sql, d1_params = base_delete.where(id_col == 1).to_sql()
    d2_sql, d2_params = base_delete.where(id_col == 2).to_sql()
    assert d1_params == [1]
    assert d2_params == [2]
    assert d1_params != d2_params
    assert d1_sql == d2_sql
    with pytest.raises(UnsafeMutationError):
        base_delete.to_sql()
