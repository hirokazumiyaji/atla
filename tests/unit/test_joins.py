"""Tests for type-safe joins, self-join aliases, and batch preload."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any

import pytest

from atla.adapter.base import DBAdapter, IsolationLevel
from atla.dialect import PostgresDialect
from atla.expressions import Column, Param
from atla.query import BaseSelectQuery
from atla.table import TableRef


@dataclass(slots=True, frozen=True)
class UserRecord:
    id: int
    email: str

    @classmethod
    def _from_row(cls, row: Sequence[Any]) -> UserRecord:
        return cls(int(row[0]), str(row[1]))


@dataclass(slots=True, frozen=True)
class PostRecord:
    id: int
    user_id: int
    title: str

    @classmethod
    def _from_row(cls, row: Sequence[Any]) -> PostRecord:
        return cls(int(row[0]), int(row[1]), str(row[2]))


class UsersTable(TableRef[UserRecord]):
    __slots__ = ("email", "id")
    __table_name__ = "users"
    _COLUMNS = ("id", "email")
    _ROW_MAPPER = UserRecord._from_row  # pyright: ignore[reportPrivateUsage]

    def __init__(self, alias: str | None = None) -> None:
        super().__init__(alias)
        t = alias or "users"
        self.id: Column[int] = Column(t, "id")
        self.email: Column[str] = Column(t, "email")

    def as_(self, alias: str) -> UsersTable:
        return UsersTable(alias=alias)


class PostsTable(TableRef[PostRecord]):
    __slots__ = ("id", "title", "user_id")
    __table_name__ = "posts"
    _COLUMNS = ("id", "user_id", "title")
    _ROW_MAPPER = PostRecord._from_row  # pyright: ignore[reportPrivateUsage]

    def __init__(self, alias: str | None = None) -> None:
        super().__init__(alias)
        t = alias or "posts"
        self.id: Column[int] = Column(t, "id")
        self.user_id: Column[int] = Column(t, "user_id")
        self.title: Column[str] = Column(t, "title")


User = UsersTable()
Post = PostsTable()


class UserQuery(BaseSelectQuery[UserRecord]):
    __slots__ = ()
    _TABLE_NAME = '"users"'
    _SELECT_COLS = '"users"."id", "users"."email"'
    _FROM_SQL = '"users"'
    _COLUMN_COUNT = 2
    _SELECT_SQL = 'SELECT "users"."id", "users"."email" FROM "users"'
    _COUNT_SQL = 'SELECT COUNT(*) FROM "users"'
    _ROW_MAPPER = UserRecord._from_row  # pyright: ignore[reportPrivateUsage]


class FakeAdapter:
    def __init__(self) -> None:
        self.dialect = PostgresDialect()
        self.calls: list[tuple[str, str, list[Any]]] = []
        self.rows: list[Sequence[Any]] = []
        self.fetch_all_responses: list[list[Sequence[Any]]] = []

    async def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Sequence[Any]]:
        self.calls.append(("fetch_all", sql, list(params)))
        if self.fetch_all_responses:
            return self.fetch_all_responses.pop(0)
        return list(self.rows)

    async def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Any] | None:
        self.calls.append(("fetch_one", sql, list(params)))
        return self.rows[0] if self.rows else None

    async def fetch_val(self, sql: str, params: Sequence[Any] = ()) -> Any:
        self.calls.append(("fetch_val", sql, list(params)))
        return 0

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        self.calls.append(("execute", sql, list(params)))
        return 0

    async def execute_many(self, sql: str, params_seq: Sequence[Sequence[Any]]) -> None:
        self.calls.append(("execute_many", sql, [list(p) for p in params_seq]))

    def transaction(
        self,
        *,
        isolation: IsolationLevel | None = None,
        readonly: bool = False,
        deferrable: bool = False,
    ) -> AbstractAsyncContextManager[DBAdapter]:
        raise NotImplementedError


@pytest.mark.asyncio
async def test_inner_join_hydrates_tuple_rows() -> None:
    adapter = FakeAdapter()
    adapter.rows = [(1, "a@example.com", 10, 1, "hello")]
    rows = await UserQuery(adapter).join(Post, Post.user_id == User.id).where(User.id == 1).all()
    assert rows == [(UserRecord(1, "a@example.com"), PostRecord(10, 1, "hello"))]
    sql = adapter.calls[0][1]
    assert 'INNER JOIN "posts" ON' in sql
    assert '"posts"."user_id" = "users"."id"' in sql
    assert sql.startswith(
        'SELECT "users"."id", "users"."email", "posts"."id", "posts"."user_id", "posts"."title"'
    )


@pytest.mark.asyncio
async def test_prepared_join_reuses_template_and_rebinds() -> None:
    adapter = FakeAdapter()
    adapter.rows = [(1, "a@example.com", 10, 1, "hello")]
    prepared = (
        UserQuery(adapter)
        .join(Post, Post.user_id == User.id)
        .where(User.email == Param("email"))
        .prepare()
    )

    assert prepared.param_names == ("email",)
    assert '"users"."email" = $1' in prepared.sql

    first = await prepared.all(email="a@example.com")
    second = await prepared.all(email="b@example.com")

    assert first == [(UserRecord(1, "a@example.com"), PostRecord(10, 1, "hello"))]
    assert second == first
    assert [call[2] for call in adapter.calls] == [["a@example.com"], ["b@example.com"]]
    assert {call[1] for call in adapter.calls} == {prepared.sql}


@pytest.mark.asyncio
async def test_left_join_maps_all_null_right_side_to_none() -> None:
    adapter = FakeAdapter()
    adapter.rows = [(1, "a@example.com", None, None, None)]
    rows = await UserQuery(adapter).left_join(Post, Post.user_id == User.id).all()
    assert rows == [(UserRecord(1, "a@example.com"), None)]
    assert "LEFT JOIN" in adapter.calls[0][1]


@pytest.mark.asyncio
async def test_self_join_uses_table_alias() -> None:
    adapter = FakeAdapter()
    manager = User.as_("manager")
    adapter.rows = [(1, "a@example.com", 2, "boss@example.com")]
    sql, _params = UserQuery(adapter).join(manager, manager.id == User.id).to_sql()
    assert 'INNER JOIN "users" AS "manager" ON' in sql
    assert '"manager"."id"' in sql
    assert '"manager"."email"' in sql


@pytest.mark.asyncio
async def test_preload_issues_single_in_query_without_n_plus_one() -> None:
    adapter = FakeAdapter()
    adapter.fetch_all_responses = [
        [(1, "a@example.com"), (2, "b@example.com")],
        [(10, 1, "p1"), (11, 1, "p2"), (20, 2, "p3")],
    ]
    rows = await (
        UserQuery(adapter)
        .preload(Post, fk=Post.user_id, key=User.id, fk_attr="user_id", key_attr="id")
        .all()
    )
    assert rows == [
        (UserRecord(1, "a@example.com"), [PostRecord(10, 1, "p1"), PostRecord(11, 1, "p2")]),
        (UserRecord(2, "b@example.com"), [PostRecord(20, 2, "p3")]),
    ]
    assert len(adapter.calls) == 2
    assert all(call[0] == "fetch_all" for call in adapter.calls)
    assert " IN (" in adapter.calls[1][1]
    assert adapter.calls[1][2] == [1, 2]
