"""Unit tests for Issue #60: SQLCompiler fixes and Query Builder consistency."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, cast

import pytest

from atla.adapter.base import DBAdapter, IsolationLevel
from atla.compiler import SQLCompiler, compile_expr
from atla.dialect import (
    CockroachDialect,
    Dialect,
    MySQLDialect,
    PostgresDialect,
    SQLiteDialect,
    TiDBDialect,
)
from atla.expressions import Column
from atla.query import (
    BaseInsertQuery,
    BaseSelectQuery,
    BaseUpdateQuery,
    PreloadOneSelectQuery,
    PreloadSelectQuery,
)
from atla.select_clauses import write_order_limit_offset
from atla.table import TableRef

DIALECTS: list[Dialect] = [
    PostgresDialect(),
    CockroachDialect(),
    MySQLDialect(),
    TiDBDialect(),
    SQLiteDialect(),
]


@dataclass(slots=True, frozen=True)
class UserRecord:
    id: int
    name: str
    org_id: int | None = None

    @classmethod
    def _from_row(cls, row: Sequence[Any]) -> UserRecord:
        org_id = None if len(row) < 3 else (int(row[2]) if row[2] is not None else None)
        return cls(int(row[0]), str(row[1]), org_id)


@dataclass(slots=True, frozen=True)
class PostRecord:
    id: int
    user_id: int
    title: str

    @classmethod
    def _from_row(cls, row: Sequence[Any]) -> PostRecord:
        return cls(int(row[0]), int(row[1]), str(row[2]))


@dataclass(slots=True, frozen=True)
class OrgRecord:
    id: int
    name: str

    @classmethod
    def _from_row(cls, row: Sequence[Any]) -> OrgRecord:
        return cls(int(row[0]), str(row[1]))


class UsersTable(TableRef[UserRecord]):
    __slots__ = ("id", "name", "org_id")
    __table_name__ = "users"
    _COLUMNS = ("id", "name", "org_id")
    _ROW_MAPPER = UserRecord._from_row

    def __init__(self, alias: str | None = None) -> None:
        super().__init__(alias)
        t = alias or "users"
        self.id: Column[int] = Column(t, "id")
        self.name: Column[str] = Column(t, "name")
        self.org_id: Column[int | None] = Column(t, "org_id")


class PostsTable(TableRef[PostRecord]):
    __slots__ = ("id", "title", "user_id")
    __table_name__ = "posts"
    _COLUMNS = ("id", "user_id", "title")
    _ROW_MAPPER = PostRecord._from_row

    def __init__(self, alias: str | None = None) -> None:
        super().__init__(alias)
        t = alias or "posts"
        self.id: Column[int] = Column(t, "id")
        self.user_id: Column[int] = Column(t, "user_id")
        self.title: Column[str] = Column(t, "title")


class OrgsTable(TableRef[OrgRecord]):
    __slots__ = ("id", "name")
    __table_name__ = "orgs"
    _COLUMNS = ("id", "name")
    _ROW_MAPPER = OrgRecord._from_row

    def __init__(self, alias: str | None = None) -> None:
        super().__init__(alias)
        t = alias or "orgs"
        self.id: Column[int] = Column(t, "id")
        self.name: Column[str] = Column(t, "name")


User = UsersTable()
Post = PostsTable()
Org = OrgsTable()


class FakeAdapter:
    def __init__(self, dialect: Dialect | None = None) -> None:
        self.dialect = dialect or PostgresDialect()
        self.calls: list[tuple[str, str, list[Any]]] = []
        self.fetch_all_responses: list[list[Sequence[Any]]] = []
        self.scalar: Any = 1

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


class UserQuery(BaseSelectQuery[UserRecord]):
    __slots__ = ()
    _TABLE_NAME = '"users"'
    _FROM_SQL = '"users"'
    _SELECT_COLS = '"users"."id", "users"."name", "users"."org_id"'
    _COLUMN_COUNT = 3
    _SELECT_SQL = 'SELECT "users"."id", "users"."name", "users"."org_id" FROM "users"'
    _COUNT_SQL = 'SELECT COUNT(*) FROM "users"'
    _ROW_MAPPER = UserRecord._from_row

    def with_posts(self) -> PreloadSelectQuery[UserRecord, PostRecord]:
        return self.preload(
            Post,
            fk=Post.user_id,
            key=User.id,
            fk_attr="user_id",
            key_attr="id",
        )

    def with_org(self) -> PreloadOneSelectQuery[UserRecord, OrgRecord | None]:
        return self.preload_one(
            Org,
            fk=User.org_id,
            key=Org.id,
            fk_attr="org_id",
            key_attr="id",
            nullable=True,
        )


class InsertQuery(BaseInsertQuery[UserRecord]):
    __slots__ = ()
    _TABLE_NAME = '"users"'
    _RETURNING_COLS = '"id", "name"'
    _ROW_MAPPER = UserRecord._from_row


class UpdateQuery(BaseUpdateQuery[UserRecord]):
    __slots__ = ()
    _TABLE_NAME = '"users"'
    _RETURNING_COLS = '"id", "name"'
    _ROW_MAPPER = UserRecord._from_row

    def set(self, **columns: Any) -> UpdateQuery:
        return self._set_values(**columns)


@pytest.mark.parametrize("dialect", DIALECTS, ids=lambda d: d.name)
def test_double_unary_minus_avoids_sql_comment(dialect: Dialect) -> None:
    expr = -(-User.id) == 10  # noqa: B002
    sql, params = compile_expr(dialect, expr)
    assert "--" not in sql
    assert params == [10]
    expected_col = dialect.quote("users") + "." + dialect.quote("id")
    expected = f"-(-{expected_col}) = {dialect.placeholder(1)}"
    assert sql == expected


@pytest.mark.parametrize("dialect", DIALECTS, ids=lambda d: d.name)
def test_postfix_unary_and_compound_binary_children_parens(dialect: Dialect) -> None:
    q = dialect.quote
    col = f"{q('users')}.{q('id')}"

    # Postfix unary IS NULL on a comparison operand must enclose operand in parens
    sql1, _ = compile_expr(dialect, (User.id == 1).is_null())
    assert sql1 == f"({col} = {dialect.placeholder(1)}) IS NULL"

    # Postfix unary IS NULL as child of binary op must be enclosed in parens
    sql2, _ = compile_expr(dialect, User.id.is_null() == False)  # noqa: E712
    assert sql2 == f"({col} IS NULL) = {dialect.placeholder(1)}"

    # BetweenOp as child of binary op must be enclosed in parens
    sql3, _ = compile_expr(dialect, User.id.between(0, 20) == False)  # noqa: E712
    expected_between = (
        f"({col} BETWEEN {dialect.placeholder(1)} AND {dialect.placeholder(2)}) "
        f"= {dialect.placeholder(3)}"
    )
    assert sql3 == expected_between

    # InOp as child of binary op must be enclosed in parens
    sql4, _ = compile_expr(dialect, User.id.in_([1, 2]) == True)  # noqa: E712
    p1, p2, p3 = dialect.placeholder(1), dialect.placeholder(2), dialect.placeholder(3)
    expected_in = f"({col} IN ({p1}, {p2})) = {p3}"
    assert sql4 == expected_in


def test_limit_offset_validation() -> None:
    adapter = FakeAdapter()
    base_query = UserQuery(adapter)
    joined_query = base_query.join(Post, Post.user_id == User.id)
    preload_query = base_query.with_posts()
    preload_one_query = base_query.with_org()
    multi_preload_query = preload_query.preload(
        Post,
        fk=Post.user_id,
        key=User.id,
        fk_attr="user_id",
        key_attr="id",
    )
    projected_query = base_query.select(User.id)

    queries: list[Any] = [
        base_query,
        joined_query,
        preload_query,
        preload_one_query,
        multi_preload_query,
        projected_query,
    ]

    for q in queries:
        with pytest.raises(ValueError, match="limit must be a non-negative integer"):
            q.limit(-1)
        with pytest.raises(ValueError, match="limit must be a non-negative integer"):
            q.limit(cast(int, True))
        with pytest.raises(ValueError, match="offset must be a non-negative integer"):
            q.offset(-1)
        with pytest.raises(ValueError, match="offset must be a non-negative integer"):
            q.offset(cast(int, False))

        # Non-negative integers must succeed
        valid_q = q.limit(0).offset(0)
        assert valid_q is not None

    compiler = SQLCompiler(PostgresDialect())
    with pytest.raises(ValueError, match="limit must be a non-negative integer"):
        write_order_limit_offset(compiler, order_by=(), limit=-5, offset=None)
    with pytest.raises(ValueError, match="offset must be a non-negative integer"):
        write_order_limit_offset(compiler, order_by=(), limit=None, offset=cast(int, True))


@pytest.mark.asyncio
async def test_preload_first_one_and_exists_respect_limit() -> None:
    adapter = FakeAdapter()
    # Adapter returns 1 user with id=1
    adapter.fetch_all_responses = [
        [(1, "Alice", 10)],
        [(100, 1, "First Post")],
    ]
    query = UserQuery(adapter).limit(1).with_posts()
    row = await query.one()
    assert row[0] == UserRecord(1, "Alice", 10)
    assert row[1] == [PostRecord(100, 1, "First Post")]
    # The parent query fetch must have LIMIT 1, not LIMIT 2
    assert " LIMIT $1" in adapter.calls[0][1]
    assert adapter.calls[0][2] == [1]

    # Test exists() delegates to parent.exists() (SELECT 1 ... LIMIT 1)
    adapter.calls.clear()
    adapter.scalar = 1
    exists_result = await query.exists()
    assert exists_result is True
    assert adapter.calls[0][0] == "fetch_val"
    assert "SELECT 1 FROM" in adapter.calls[0][1]


@pytest.mark.asyncio
async def test_preload_chaining_and_dynamic_with() -> None:
    adapter = FakeAdapter()
    # 1 User, posts for that user, org for that user
    adapter.fetch_all_responses = [
        [(1, "Alice", 10)],
        [(100, 1, "P1")],
        [(10, "Acme Corp")],
    ]

    # Test chaining with_posts().with_org() dynamically via __getattr__
    query = UserQuery(adapter).with_posts().with_org()
    results = await query.all()
    assert len(results) == 1
    user, posts, org = results[0]
    assert user == UserRecord(1, "Alice", 10)
    assert posts == [PostRecord(100, 1, "P1")]
    assert org == OrgRecord(10, "Acme Corp")

    # Test chaining reverse order: with_org().with_posts()
    adapter.fetch_all_responses = [
        [(1, "Alice", 10)],
        [(10, "Acme Corp")],
        [(100, 1, "P1")],
    ]
    reverse_query = UserQuery(adapter).with_org().with_posts()
    rev_results = await reverse_query.all()
    assert len(rev_results) == 1
    rev_user, rev_org, rev_posts = rev_results[0]
    assert rev_user == UserRecord(1, "Alice", 10)
    assert rev_org == OrgRecord(10, "Acme Corp")
    assert rev_posts == [PostRecord(100, 1, "P1")]


def test_column_deduplication_last_write_wins() -> None:
    adapter = FakeAdapter()

    # 1. BaseInsertQuery.values
    insert = InsertQuery(adapter).values(id=1, name="Alice").values(name="Bob")
    sql, params = insert.to_sql()
    assert sql == 'INSERT INTO "users" ("id", "name") VALUES ($1, $2)'
    assert params == [1, "Bob"]

    # 2. BaseUpdateQuery._set_values
    update = UpdateQuery(adapter).where(User.id == 1).set(id=1, name="Alice").set(name="Bob")
    sql, params = update.to_sql()
    assert sql == 'UPDATE "users" SET "id" = $1, "name" = $2 WHERE "users"."id" = $3'
    assert params == [1, "Bob", 1]

    # 3. UpsertQuery.do_update
    upsert = (
        InsertQuery(adapter)
        .values(id=1, name="Alice")
        .on_conflict(User.id)
        .do_update(name="Alice")
        .do_update(name="Bob")
    )
    sql, params = upsert.to_sql()
    assert 'DO UPDATE SET "name" = $3' in sql
    assert params == [1, "Alice", "Bob"]

    # Accumulate different columns
    upsert2 = (
        InsertQuery(adapter)
        .values(id=1, name="Alice")
        .on_conflict(User.id)
        .do_update(name="Alice")
        .do_update(id=2)
    )
    sql2, params2 = upsert2.to_sql()
    assert 'DO UPDATE SET "name" = $3, "id" = $4' in sql2
    assert params2 == [1, "Alice", "Alice", 2]


@pytest.mark.asyncio
async def test_count_and_exists_on_joined_and_projected_queries() -> None:
    adapter = FakeAdapter()
    base = UserQuery(adapter)
    joined = base.join(Post, Post.user_id == User.id).where(User.id == 1)
    projected = base.select(User.id, User.name).where(User.id == 1)

    # Joined count
    adapter.scalar = 5
    count_val = await joined.count()
    assert count_val == 5
    assert 'SELECT COUNT(*) FROM "users" INNER JOIN "posts"' in adapter.calls[-1][1]
    assert 'WHERE "users"."id" = $1' in adapter.calls[-1][1]

    # Joined exists
    adapter.scalar = 1
    exists_val = await joined.exists()
    assert exists_val is True
    assert 'SELECT 1 FROM "users" INNER JOIN "posts"' in adapter.calls[-1][1]
    assert "LIMIT $2" in adapter.calls[-1][1]

    # Projected count
    adapter.scalar = 3
    p_count = await projected.count()
    assert p_count == 3
    assert 'SELECT COUNT(*) FROM "users"' in adapter.calls[-1][1]

    # Projected exists
    adapter.scalar = 1
    p_exists = await projected.exists()
    assert p_exists is True
    assert 'SELECT 1 FROM "users"' in adapter.calls[-1][1]
    assert "LIMIT $2" in adapter.calls[-1][1]

    # Rejection of group_by/having/distinct on count & exists
    grouped_joined = joined.group_by(User.id)
    with pytest.raises(ValueError, match="count\\(\\) ignores group_by"):
        await grouped_joined.count()
    with pytest.raises(ValueError, match="exists\\(\\) ignores group_by"):
        await grouped_joined.exists()

    grouped_projected = projected.group_by(User.id)
    with pytest.raises(ValueError, match="count\\(\\) ignores group_by"):
        await grouped_projected.count()
    with pytest.raises(ValueError, match="exists\\(\\) ignores group_by"):
        await grouped_projected.exists()
