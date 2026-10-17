"""Enhanced preload / belongs-to / multi-relation coverage."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any

import pytest

from atla.adapter.base import DBAdapter, IsolationLevel
from atla.dialect import PostgresDialect
from atla.errors import MultipleResultsError, NotFoundError
from atla.expressions import Column
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
    published: bool = True

    @classmethod
    def _from_row(cls, row: Sequence[Any]) -> PostRecord:
        published = True if len(row) < 4 else bool(row[3])
        return cls(int(row[0]), int(row[1]), str(row[2]), published)


@dataclass(slots=True, frozen=True)
class CommentRecord:
    id: int
    user_id: int
    body: str

    @classmethod
    def _from_row(cls, row: Sequence[Any]) -> CommentRecord:
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


class PostsTable(TableRef[PostRecord]):
    __slots__ = ("id", "published", "title", "user_id")
    __table_name__ = "posts"
    _COLUMNS = ("id", "user_id", "title", "published")
    _ROW_MAPPER = PostRecord._from_row  # pyright: ignore[reportPrivateUsage]

    def __init__(self, alias: str | None = None) -> None:
        super().__init__(alias)
        t = alias or "posts"
        self.id: Column[int] = Column(t, "id")
        self.user_id: Column[int] = Column(t, "user_id")
        self.title: Column[str] = Column(t, "title")
        self.published: Column[bool] = Column(t, "published")


class CommentsTable(TableRef[CommentRecord]):
    __slots__ = ("body", "id", "user_id")
    __table_name__ = "comments"
    _COLUMNS = ("id", "user_id", "body")
    _ROW_MAPPER = CommentRecord._from_row  # pyright: ignore[reportPrivateUsage]

    def __init__(self, alias: str | None = None) -> None:
        super().__init__(alias)
        t = alias or "comments"
        self.id: Column[int] = Column(t, "id")
        self.user_id: Column[int] = Column(t, "user_id")
        self.body: Column[str] = Column(t, "body")


User = UsersTable()
Post = PostsTable()
Comment = CommentsTable()


class UserQuery(BaseSelectQuery[UserRecord]):
    __slots__ = ()
    _TABLE_NAME = '"users"'
    _FROM_SQL = '"users"'
    _SELECT_COLS = '"users"."id", "users"."email"'
    _COLUMN_COUNT = 2
    _SELECT_SQL = 'SELECT "users"."id", "users"."email" FROM "users"'
    _COUNT_SQL = 'SELECT COUNT(*) FROM "users"'
    _ROW_MAPPER = UserRecord._from_row  # pyright: ignore[reportPrivateUsage]


class PostQuery(BaseSelectQuery[PostRecord]):
    __slots__ = ()
    _TABLE_NAME = '"posts"'
    _FROM_SQL = '"posts"'
    _SELECT_COLS = '"posts"."id", "posts"."user_id", "posts"."title", "posts"."published"'
    _COLUMN_COUNT = 4
    _SELECT_SQL = (
        'SELECT "posts"."id", "posts"."user_id", "posts"."title", "posts"."published" FROM "posts"'
    )
    _COUNT_SQL = 'SELECT COUNT(*) FROM "posts"'
    _ROW_MAPPER = PostRecord._from_row  # pyright: ignore[reportPrivateUsage]


class FakeAdapter:
    def __init__(self) -> None:
        self.dialect = PostgresDialect()
        self.calls: list[tuple[str, str, list[Any]]] = []
        self.fetch_all_responses: list[list[Sequence[Any]]] = []
        self.scalar: Any = 0

    async def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Sequence[Any]]:
        self.calls.append(("fetch_all", sql, list(params)))
        if self.fetch_all_responses:
            return self.fetch_all_responses.pop(0)
        return []

    async def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Any] | None:
        rows = await self.fetch_all(sql, params)
        return rows[0] if rows else None

    async def fetch_val(self, sql: str, params: Sequence[Any] = ()) -> Any:
        self.calls.append(("fetch_val", sql, list(params)))
        return self.scalar

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


@pytest.mark.asyncio
async def test_preload_related_filters_and_one() -> None:
    adapter = FakeAdapter()
    adapter.fetch_all_responses = [
        [(1, "a@example.com")],
        [(10, 1, "pub", True)],
    ]
    row = await (
        UserQuery(adapter)
        .where(User.id == 1)
        .preload(
            Post,
            Post.published.eq(True),
            fk=Post.user_id,
            key=User.id,
            fk_attr="user_id",
            key_attr="id",
            order_by=[Post.id.asc()],
        )
        .one()
    )
    assert row[0] == UserRecord(1, "a@example.com")
    assert row[1] == [PostRecord(10, 1, "pub", True)]
    assert len(adapter.calls) == 2
    assert True in adapter.calls[1][2]


@pytest.mark.asyncio
async def test_preload_one_raises_on_empty() -> None:
    adapter = FakeAdapter()
    adapter.fetch_all_responses = [[]]
    with pytest.raises(NotFoundError):
        await (
            UserQuery(adapter)
            .preload(Post, fk=Post.user_id, key=User.id, fk_attr="user_id", key_attr="id")
            .one()
        )


@pytest.mark.asyncio
async def test_preload_one_raises_on_multiple() -> None:
    adapter = FakeAdapter()
    adapter.fetch_all_responses = [
        [(1, "a@example.com"), (2, "b@example.com")],
        [],
    ]
    with pytest.raises(MultipleResultsError):
        await (
            UserQuery(adapter)
            .preload(Post, fk=Post.user_id, key=User.id, fk_attr="user_id", key_attr="id")
            .one()
        )


@pytest.mark.asyncio
async def test_belongs_to_preload() -> None:
    adapter = FakeAdapter()
    adapter.fetch_all_responses = [
        [(10, 1, "p1", True), (11, 2, "p2", True)],
        [(1, "a@example.com"), (2, "b@example.com")],
    ]
    rows = await (
        PostQuery(adapter)
        .preload_one(
            User,
            fk=Post.user_id,
            key=User.id,
            fk_attr="user_id",
            key_attr="id",
        )
        .all()
    )
    assert rows == [
        (PostRecord(10, 1, "p1", True), UserRecord(1, "a@example.com")),
        (PostRecord(11, 2, "p2", True), UserRecord(2, "b@example.com")),
    ]
    assert len(adapter.calls) == 2
    assert " IN (" in adapter.calls[1][1]


@pytest.mark.asyncio
async def test_belongs_to_missing_non_null_parent_raises() -> None:
    adapter = FakeAdapter()
    adapter.fetch_all_responses = [
        [(10, 99, "orphan", True)],
        [],
    ]
    with pytest.raises(NotFoundError, match="missing related row"):
        await (
            PostQuery(adapter)
            .preload_one(
                User,
                fk=Post.user_id,
                key=User.id,
                fk_attr="user_id",
                key_attr="id",
                nullable=False,
            )
            .all()
        )


@pytest.mark.asyncio
async def test_belongs_to_nullable_missing_parent() -> None:
    adapter = FakeAdapter()
    adapter.fetch_all_responses = [
        [(10, 99, "orphan", True)],
        [],
    ]
    rows = await (
        PostQuery(adapter)
        .preload_one(
            User,
            fk=Post.user_id,
            key=User.id,
            fk_attr="user_id",
            key_attr="id",
            nullable=True,
        )
        .all()
    )
    assert rows == [(PostRecord(10, 99, "orphan", True), None)]


@pytest.mark.asyncio
async def test_belongs_to_filtered_missing_parent_returns_none() -> None:
    """Parent-side predicates can exclude a real parent; that is not NotFoundError."""
    adapter = FakeAdapter()
    adapter.fetch_all_responses = [
        [(10, 1, "p1", True)],
        [],  # filter excluded the user row
    ]
    rows = await (
        PostQuery(adapter)
        .preload_one(
            User,
            User.email.eq("nobody@example.com"),
            fk=Post.user_id,
            key=User.id,
            fk_attr="user_id",
            key_attr="id",
            nullable=False,
        )
        .all()
    )
    assert rows == [(PostRecord(10, 1, "p1", True), None)]


@pytest.mark.asyncio
async def test_multi_relation_preload() -> None:
    adapter = FakeAdapter()
    adapter.fetch_all_responses = [
        [(1, "a@example.com")],
        [(10, 1, "p1", True)],
        [(100, 1, "c1")],
    ]
    rows = await (
        UserQuery(adapter)
        .preload(Post, fk=Post.user_id, key=User.id, fk_attr="user_id", key_attr="id")
        .preload(Comment, fk=Comment.user_id, key=User.id, fk_attr="user_id", key_attr="id")
        .all()
    )
    assert len(rows) == 1
    parent, posts, comments = rows[0]
    assert parent.id == 1
    assert len(posts) == 1
    assert len(comments) == 1
    assert len(adapter.calls) == 3


@pytest.mark.asyncio
async def test_preload_count_and_exists() -> None:
    adapter = FakeAdapter()
    adapter.scalar = 3
    adapter.fetch_all_responses = [[(1, "a@example.com")]]
    query = UserQuery(adapter).preload(
        Post, fk=Post.user_id, key=User.id, fk_attr="user_id", key_attr="id"
    )
    assert await query.count() == 3
    assert await query.exists() is True
