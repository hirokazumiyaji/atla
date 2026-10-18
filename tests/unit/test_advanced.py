"""Tests for insert_many, upsert, and projected select."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any, ClassVar, NamedTuple

import pytest

from atla.adapter.base import DBAdapter, IsolationLevel
from atla.advanced import BaseInsertManyQuery, UpsertQuery, insert_many_rows
from atla.dialect import MySQLDialect, PostgresDialect, SQLiteDialect
from atla.errors import MultipleResultsError
from atla.expressions import Column, excluded
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


class CopyFakeAdapter(FakeAdapter):
    """Adapter exposing the optional PostgreSQL binary COPY capability."""

    def __init__(self, dialect: Any = None) -> None:
        super().__init__(dialect or PostgresDialect())
        self.copies: list[tuple[str, list[str], list[list[Any]]]] = []

    async def copy_records_to_table(
        self,
        table: str,
        *,
        columns: Sequence[str],
        records: Sequence[Sequence[Any]],
    ) -> int:
        rows = [list(row) for row in records]
        self.copies.append((table, list(columns), rows))
        return len(rows)


class CopyUserInsertMany(UserInsertMany):
    __slots__ = ()
    _COPY_TABLE = "users"


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


def test_insert_many_rows_normalizes_key_order() -> None:
    rows = insert_many_rows(
        [{"email": "a", "name": "A"}, {"name": "B", "email": "b"}],
        attr_to_sql={},
    )
    assert rows == (
        (("email", "a"), ("name", "A")),
        (("email", "b"), ("name", "B")),
    )


def test_insert_many_rows_rejects_unknown_attr() -> None:
    with pytest.raises(ValueError, match="unknown column"):
        insert_many_rows(
            [{"email": "a", "emial": "typo"}],
            attr_to_sql={"email": "email", "name": "name"},
        )


@pytest.mark.asyncio
async def test_insert_many_chunks_large_batches() -> None:
    adapter = FakeAdapter(SQLiteDialect())
    # Large batches use atomic multi-VALUES INSERT chunks (bind-parameter limit).
    rows = tuple((("a", i), ("b", i)) for i in range(20_000))
    q = UserInsertMany(adapter, table_name='"t"', rows=rows)
    assert await q.execute() == 4  # FakeAdapter.execute_count * 2 chunks
    executes = [c for c in adapter.calls if c[0] == "execute"]
    assert len(executes) == 2
    assert all("VALUES" in c[1] for c in executes)
    assert not [c for c in adapter.calls if c[0] == "execute_many"]


@pytest.mark.asyncio
async def test_projected_select_quotes_reserved_column_names() -> None:
    adapter = FakeAdapter()
    order_col: Column[str] = Column("users", "order")
    sql, _params = UserQuery(adapter).select(order_col).to_sql()
    assert sql.startswith('SELECT "users"."order" FROM "users"')


def test_on_conflict_accepts_column_and_attr_mapping() -> None:
    adapter = FakeAdapter(PostgresDialect())

    class MappedUpsert(UpsertQuery[UserRecord]):
        __slots__ = ()
        _RETURNING_COLS = '"id", "email"'
        _ROW_MAPPER = UserRecord.from_row
        _ATTR_TO_SQL: ClassVar[Mapping[str, str]] = {"from_": "from"}

    class MappedInsert(BaseInsertQuery[UserRecord]):
        __slots__ = ()
        _TABLE_NAME = '"users"'
        _RETURNING_COLS = '"id", "email"'
        _ROW_MAPPER = UserRecord.from_row
        _ATTR_TO_SQL: ClassVar[Mapping[str, str]] = {"from_": "from"}
        _UPSERT_CLS = MappedUpsert

    sql, params = (
        MappedInsert(adapter)
        .values(**{"from": "x"})
        .on_conflict(email_col)
        .do_update_attrs(from_="y")
        .to_sql()
    )
    assert 'ON CONFLICT ("email") DO UPDATE SET "from" = $2' in sql
    assert params == ["x", "y"]


def test_excluded_compiles_for_conflict_and_duplicate_dialects() -> None:
    pg = FakeAdapter(PostgresDialect())
    sql, params = (
        UserInsert(pg)
        .values(email="a@example.com")
        .on_conflict("email")
        .do_update(email=excluded(email_col))
        .to_sql()
    )
    assert 'DO UPDATE SET "email" = EXCLUDED."email"' in sql
    assert params == ["a@example.com"]

    mysql = FakeAdapter(MySQLDialect())
    sql, params = (
        UserInsert(mysql)
        .values(email="a@example.com", id=1)
        .on_conflict()
        .do_update(email=excluded(email_col))
        .to_sql()
    )
    assert "ON DUPLICATE KEY UPDATE `email` = VALUES(`email`)" in sql
    assert params == ["a@example.com", 1]


@pytest.mark.asyncio
async def test_upsert_returning_all_and_one() -> None:
    adapter = FakeAdapter(PostgresDialect())
    adapter.rows = [(1, "a@example.com"), (2, "b@example.com")]
    q = (
        UserInsert(adapter)
        .values(email="a@example.com")
        .on_conflict("email")
        .do_update(email="a@example.com")
        .returning()
    )
    rows = await q.all()
    assert len(rows) == 2
    with pytest.raises(MultipleResultsError):
        await q.one()
    adapter.rows = [(1, "a@example.com")]
    assert await q.one() == UserRecord(1, "a@example.com")


@pytest.mark.asyncio
async def test_mutation_one_skips_mapper_when_multiple_rows() -> None:
    mapped: list[Sequence[Any]] = []

    def counting_mapper(row: Sequence[Any]) -> UserRecord:
        mapped.append(row)
        return UserRecord.from_row(row)

    class CountingInsert(BaseInsertQuery[UserRecord]):
        __slots__ = ()
        _TABLE_NAME = '"users"'
        _RETURNING_COLS = '"id", "email"'
        _ROW_MAPPER = staticmethod(counting_mapper)

    adapter = FakeAdapter(PostgresDialect())
    adapter.rows = [(1, "a@example.com"), (2, "b@example.com")]
    q = CountingInsert(adapter).values(email="a@example.com").returning()
    with pytest.raises(MultipleResultsError):
        await q.one()
    assert mapped == []
    adapter.rows = [(1, "a@example.com")]
    assert await q.one() == UserRecord(1, "a@example.com")
    assert mapped == [(1, "a@example.com")]


@pytest.mark.asyncio
async def test_insert_many_execute_many_uses_single_row_driver_bulk() -> None:
    adapter = FakeAdapter(PostgresDialect())
    rows = tuple((("a", f"v{i}"), ("b", i)) for i in range(3))
    q = UserInsertMany(adapter, table_name='"t"', rows=rows)

    assert await q.execute_many() == 3

    calls = [call for call in adapter.calls if call[0] == "execute_many"]
    assert len(calls) == 1
    assert calls[0][1] == 'INSERT INTO "t" ("a", "b") VALUES ($1, $2)'
    assert calls[0][2] == [["v0", 0], ["v1", 1], ["v2", 2]]
    assert not [call for call in adapter.calls if call[0] == "execute"]


@pytest.mark.asyncio
async def test_insert_many_execute_many_rejects_returning_and_expr() -> None:
    adapter = FakeAdapter()
    rows = ((("a", 1),),)
    with pytest.raises(ValueError, match="does not support RETURNING"):
        await UserInsertMany(adapter, table_name='"t"', rows=rows).returning().execute_many()

    expr_rows = ((("a", id_col),),)
    with pytest.raises(ValueError, match="requires literal values"):
        await UserInsertMany(adapter, table_name='"t"', rows=expr_rows).execute_many()


def test_insert_many_multi_values_uses_one_placeholder_template() -> None:
    adapter = FakeAdapter(SQLiteDialect())
    rows = tuple((("a", i), ("b", i)) for i in range(4))
    sql, params = UserInsertMany(adapter, table_name='"t"', rows=rows).to_sql()

    assert sql == 'INSERT INTO "t" ("a", "b") VALUES (?, ?), (?, ?), (?, ?), (?, ?)'
    assert params == [0, 0, 1, 1, 2, 2, 3, 3]


@pytest.mark.asyncio
async def test_insert_many_copy_uses_copy_adapter() -> None:
    adapter = CopyFakeAdapter()
    rows = ((("a", "x"), ("b", 1)), (("a", "y"), ("b", 2)))
    q = CopyUserInsertMany(adapter, table_name='"users"', rows=rows)

    assert await q.copy() == 2

    assert adapter.copies == [("users", ["a", "b"], [["x", 1], ["y", 2]])]
    assert not adapter.calls


@pytest.mark.asyncio
async def test_insert_many_copy_requires_copy_capable_adapter() -> None:
    adapter = FakeAdapter()
    q = CopyUserInsertMany(adapter, table_name='"users"', rows=((("a", 1),),))
    with pytest.raises(TypeError, match="CopyAdapter"):
        await q.copy()


@pytest.mark.asyncio
async def test_insert_many_copy_requires_raw_table_name() -> None:
    adapter = CopyFakeAdapter()
    q = UserInsertMany(adapter, table_name='"users"', rows=((("a", 1),),))
    with pytest.raises(ValueError, match="known table name"):
        await q.copy()


@pytest.mark.asyncio
async def test_insert_many_copy_rejects_returning_and_expr() -> None:
    adapter = CopyFakeAdapter()
    rows = ((("a", 1),),)
    with pytest.raises(ValueError, match="does not support RETURNING"):
        await CopyUserInsertMany(adapter, table_name='"users"', rows=rows).returning().copy()

    expr_rows = ((("a", id_col),),)
    with pytest.raises(ValueError, match="requires literal values"):
        await CopyUserInsertMany(adapter, table_name='"users"', rows=expr_rows).copy()


def test_insert_many_dollar_placeholders_advance_per_row() -> None:
    """Dollar placeholders embed the position, so rows must not share numbering."""
    adapter = FakeAdapter(PostgresDialect())
    rows = ((("a", 1), ("b", "x")), (("a", 2), ("b", "y")))
    sql, params = UserInsertMany(adapter, table_name='"t"', rows=rows).to_sql()

    assert sql == 'INSERT INTO "t" ("a", "b") VALUES ($1, $2), ($3, $4)'
    assert params == [1, "x", 2, "y"]


def test_insert_many_rows_reports_column_set_mismatch_after_empty_first_row() -> None:
    with pytest.raises(ValueError, match="same column set"):
        insert_many_rows([{}, {"a": 1}], attr_to_sql={})
