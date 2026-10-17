"""GROUP BY / HAVING / DISTINCT / FOR UPDATE / subquery / stream coverage."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any, NamedTuple

import pytest

from atla.adapter.base import DBAdapter, IsolationLevel
from atla.dialect import MySQLDialect, PostgresDialect, SQLiteDialect, TiDBDialect
from atla.expressions import Column, count, exists
from atla.query import BaseSelectQuery


class UserRecord(NamedTuple):
    id: int
    role: str

    @staticmethod
    def from_row(row: Sequence[Any]) -> UserRecord:
        return UserRecord(id=int(row[0]), role=str(row[1]))


class FakeAdapter:
    def __init__(self, dialect: Any | None = None) -> None:
        self.dialect = dialect or PostgresDialect()
        self.calls: list[tuple[str, str, list[Any]]] = []
        self.rows: list[Sequence[Any]] = [
            (1, "admin"),
            (2, "user"),
            (3, "user"),
            (4, "guest"),
        ]

    async def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Sequence[Any]]:
        self.calls.append(("fetch_all", sql, list(params)))
        rows = list(self.rows)
        if " LIMIT $" in sql:
            index = int(sql.split(" LIMIT $", 1)[1].split(None, 1)[0]) - 1
            limit = int(params[index])
            offset = 0
            if " OFFSET $" in sql:
                oindex = int(sql.split(" OFFSET $", 1)[1].split(None, 1)[0]) - 1
                offset = int(params[oindex])
            return rows[offset : offset + limit]
        return rows

    async def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Any] | None:
        return self.rows[0] if self.rows else None

    async def fetch_val(self, sql: str, params: Sequence[Any] = ()) -> Any:
        return len(self.rows)

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        return 0

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


class UserQuery(BaseSelectQuery[UserRecord]):
    __slots__ = ()
    _TABLE_NAME = '"users"'
    _FROM_SQL = '"users"'
    _SELECT_COLS = '"users"."id", "users"."role"'
    _COLUMN_COUNT = 2
    _SELECT_SQL = 'SELECT "users"."id", "users"."role" FROM "users"'
    _COUNT_SQL = 'SELECT COUNT(*) FROM "users"'
    _ROW_MAPPER = staticmethod(UserRecord.from_row)


ID = Column[int]("users", "id")
ROLE = Column[str]("users", "role")
POST_USER = Column[int]("posts", "user_id")
PUBLISHED = Column[bool]("posts", "published")


def test_group_by_having_sql() -> None:
    adapter = FakeAdapter()
    sql, params = (
        UserQuery(adapter).select(ROLE, count()).group_by(ROLE).having(count() > 10).to_sql()
    )
    assert sql.startswith('SELECT "users"."role", COUNT(*) FROM "users"')
    assert " GROUP BY " in sql
    assert " HAVING " in sql
    assert "COUNT(*) > $1" in sql
    assert params == [10]


def test_for_update_skip_locked() -> None:
    adapter = FakeAdapter()
    sql, _ = UserQuery(adapter).where(ID == 1).for_update(skip_locked=True).to_sql()
    assert sql.endswith(" FOR UPDATE SKIP LOCKED")
    assert "WHERE" in sql


def test_for_update_rejected_on_sqlite() -> None:
    adapter = FakeAdapter(SQLiteDialect())
    with pytest.raises(ValueError, match="does not support FOR UPDATE"):
        UserQuery(adapter).for_update().to_sql()


def test_tidb_share_lock_and_rejects_skip_locked() -> None:
    adapter = FakeAdapter(TiDBDialect())
    sql, _ = UserQuery(adapter).for_share().to_sql()
    assert sql.endswith(" LOCK IN SHARE MODE")
    with pytest.raises(ValueError, match="SKIP LOCKED"):
        UserQuery(adapter).for_update(skip_locked=True).to_sql()
    sql, _ = UserQuery(adapter).for_update().to_sql()
    assert sql.endswith(" FOR UPDATE")


def test_distinct_and_distinct_on() -> None:
    adapter = FakeAdapter()
    sql, _ = UserQuery(adapter).distinct().to_sql()
    assert sql.startswith('SELECT DISTINCT "users"."id"')
    sql, _ = UserQuery(adapter).distinct(ROLE).order_by(ROLE.asc()).to_sql()
    assert "DISTINCT ON (" in sql
    mysql = FakeAdapter(MySQLDialect())
    with pytest.raises(ValueError, match="DISTINCT ON"):
        UserQuery(mysql).distinct(ROLE).to_sql()


def test_subquery_in_rebinding() -> None:
    adapter = FakeAdapter()

    class PostQuery(BaseSelectQuery[UserRecord]):
        __slots__ = ()
        _TABLE_NAME = '"posts"'
        _FROM_SQL = '"posts"'
        _SELECT_COLS = '"posts"."user_id"'
        _COLUMN_COUNT = 1
        _SELECT_SQL = 'SELECT "posts"."user_id" FROM "posts"'
        _COUNT_SQL = 'SELECT COUNT(*) FROM "posts"'
        _ROW_MAPPER = staticmethod(UserRecord.from_row)

    sub = PostQuery(adapter).select(POST_USER).where(PUBLISHED.eq(True))
    sql, params = UserQuery(adapter).where(ID.in_(sub)).to_sql()
    assert 'IN (SELECT "posts"."user_id" FROM "posts" WHERE' in sql
    assert params == [True]
    # Outer WHERE reuses rebound $1 from the subquery.
    assert "$1" in sql


def test_exists_predicate() -> None:
    adapter = FakeAdapter()
    sub = UserQuery(adapter).select(ID).where(ROLE.eq("admin"))
    sql, params = UserQuery(adapter).where(exists(sub)).to_sql()
    assert "EXISTS (SELECT" in sql
    assert params == ["admin"]


@pytest.mark.asyncio
async def test_stream_batches() -> None:
    adapter = FakeAdapter()
    rows = [row async for row in UserQuery(adapter).order_by(ID.asc()).stream(batch_size=2)]
    assert len(rows) == 4
    assert rows[0].role == "admin"
    assert any("LIMIT" in call[1] for call in adapter.calls)


@pytest.mark.asyncio
async def test_stream_requires_order_by_after_first_batch() -> None:
    adapter = FakeAdapter()
    with pytest.raises(ValueError, match="ORDER BY"):
        _ = [row async for row in UserQuery(adapter).stream(batch_size=2)]
