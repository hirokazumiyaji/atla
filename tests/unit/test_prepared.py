"""Tests for ``Param`` slots and ``prepare()`` / ``CompiledQuery``."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any, NamedTuple
from uuid import UUID, uuid4

import pytest

from atla.adapter.base import DBAdapter, IsolationLevel
from atla.dialect import PostgresDialect, SQLiteDialect
from atla.expressions import Column, Param, count
from atla.prepared import CompiledQuery
from atla.query import BaseDeleteQuery, BaseInsertQuery, BaseSelectQuery, BaseUpdateQuery


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
        self.rows: list[Sequence[Any]] = [(1, "a@example.com"), (2, "b@example.com")]
        self.scalar: Any = 7
        self.execute_count = 3

    async def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Sequence[Any]]:
        self.calls.append(("fetch_all", sql, list(params)))
        return list(self.rows)

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
        self.calls.append(("execute_many", sql, [list(row) for row in params_seq]))

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
    _ROW_CTOR = UserRecord


class UserInsert(BaseInsertQuery[UserRecord]):
    __slots__ = ()
    _TABLE_NAME = '"users"'
    _RETURNING_COLS = '"id", "email"'
    _ROW_MAPPER = UserRecord.from_row


class UserUpdate(BaseUpdateQuery[UserRecord]):
    __slots__ = ()
    _TABLE_NAME = '"users"'
    _RETURNING_COLS = '"id", "email"'
    _ROW_MAPPER = UserRecord.from_row


id_col: Column[int] = Column("users", "id")
email_col: Column[str] = Column("users", "email")


def test_param_validates_name() -> None:
    assert Param("email").name == "email"
    with pytest.raises(TypeError, match="must be a string"):
        Param(1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="valid Python identifier"):
        Param("")
    with pytest.raises(ValueError, match="valid Python identifier"):
        Param("not a name")
    with pytest.raises(ValueError, match="valid Python identifier"):
        Param("class")
    # Soft keywords remain usable as keyword arguments.
    assert Param("match").name == "match"


def test_param_requires_prepare() -> None:
    query = UserQuery(FakeAdapter()).where(id_col == Param("id"))
    with pytest.raises(ValueError, match="requires a precompiled query"):
        query.to_sql()


@pytest.mark.asyncio
async def test_prepare_rebinds_without_recompiling(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = FakeAdapter()
    prepared = UserQuery(adapter).where(id_col == Param("id")).limit(5).prepare()

    assert (
        prepared.sql
        == 'SELECT "users"."id", "users"."email" FROM "users" WHERE "users"."id" = $1 LIMIT $2'
    )
    assert prepared.param_names == ("id",)

    compiles = 0
    original = __import__("atla.query", fromlist=["SQLCompiler"]).SQLCompiler

    def counting_compiler(*args: Any, **kwargs: Any) -> Any:
        nonlocal compiles
        compiles += 1
        return original(*args, **kwargs)

    monkeypatch.setattr("atla.query.SQLCompiler", counting_compiler)

    assert [row.id for row in await prepared.all(id=1)] == [1, 2]
    assert [row.id for row in await prepared.all(id=2)] == [1, 2]
    assert compiles == 0
    assert [call[2] for call in adapter.calls] == [[1, 5], [2, 5]]
    assert {call[1] for call in adapter.calls} == {prepared.sql}


@pytest.mark.asyncio
async def test_prepare_binds_by_name_and_position() -> None:
    adapter = FakeAdapter()
    prepared = (
        UserQuery(adapter).where(id_col == Param("id"), email_col == Param("email")).prepare()
    )
    assert prepared.param_names == ("id", "email")

    await prepared.all(id=1, email="a@example.com")
    await prepared.all(2, "b@example.com")

    assert [call[2] for call in adapter.calls] == [
        [1, "a@example.com"],
        [2, "b@example.com"],
    ]


@pytest.mark.asyncio
async def test_prepare_repeated_slot_name_fills_every_occurrence() -> None:
    adapter = FakeAdapter()
    prepared = UserQuery(adapter).where(id_col == Param("key"), email_col == Param("key")).prepare()
    assert prepared.param_names == ("key",)

    await prepared.all(key=9)

    # Each occurrence gets its own placeholder (portable across `%s` / `?`
    # dialects); one bound value fills them all.
    assert prepared.sql.count("$1") == 1
    assert prepared.sql.count("$2") == 1
    assert adapter.calls[0][2] == [9, 9]


@pytest.mark.asyncio
async def test_prepare_binding_errors() -> None:
    adapter = FakeAdapter()
    prepared = (
        UserQuery(adapter).where(id_col == Param("id"), email_col == Param("email")).prepare()
    )

    with pytest.raises(TypeError, match="missing parameter"):
        await prepared.all(id=1)
    with pytest.raises(TypeError, match="unknown parameter"):
        await prepared.all(id=1, email="a", extra=2)
    with pytest.raises(TypeError, match="given twice"):
        await prepared.all(1, id=1, email="a")
    with pytest.raises(TypeError, match="at most 2 positional"):
        await prepared.all(1, "a", "b")


@pytest.mark.asyncio
async def test_prepare_has_no_slots_when_no_params() -> None:
    adapter = FakeAdapter()
    prepared = UserQuery(adapter).prepare()
    assert prepared.param_names == ()
    assert prepared.sql == UserQuery._SELECT_SQL  # pyright: ignore[reportPrivateUsage]
    await prepared.all()
    assert adapter.calls[0][2] == []


@pytest.mark.asyncio
async def test_prepare_encodes_bound_values_per_dialect() -> None:
    adapter = FakeAdapter(SQLiteDialect())
    prepared = UserQuery(adapter).where(id_col == Param("id")).prepare()
    value = uuid4()

    await prepared.all(id=value)
    await prepared.all(id=UUID(int=0))

    assert adapter.calls[0][2] == [str(value)]
    assert adapter.calls[1][2] == [str(UUID(int=0))]


@pytest.mark.asyncio
async def test_prepared_first_uses_fetch_one_without_limit() -> None:
    adapter = FakeAdapter()
    prepared = UserQuery(adapter).where(id_col == Param("id")).prepare()

    row = await prepared.first(id=1)

    assert row == UserRecord(1, "a@example.com")
    assert adapter.calls[0][0] == "fetch_one"
    assert "LIMIT" not in prepared.sql


@pytest.mark.asyncio
async def test_prepared_first_returns_none_without_rows() -> None:
    adapter = FakeAdapter()
    adapter.rows = []
    prepared = UserQuery(adapter).where(id_col == Param("id")).prepare()

    assert await prepared.first(id=1) is None


@pytest.mark.asyncio
async def test_prepared_projection_scalar_fetch_val() -> None:
    adapter = FakeAdapter()
    prepared = UserQuery(adapter).select(count()).prepare()

    assert await prepared.fetch_val() == 7
    assert adapter.calls[0][0] == "fetch_val"


@pytest.mark.asyncio
async def test_prepared_projection_two_columns_rejects_fetch_val() -> None:
    adapter = FakeAdapter()
    prepared = UserQuery(adapter).select(id_col, email_col).prepare()

    rows = await prepared.all()
    assert rows == [(1, "a@example.com"), (2, "b@example.com")]
    with pytest.raises(ValueError, match="single-column projection"):
        await prepared.fetch_val()


@pytest.mark.asyncio
async def test_prepared_record_query_rejects_fetch_val_and_execute() -> None:
    adapter = FakeAdapter()
    prepared = UserQuery(adapter).prepare()

    with pytest.raises(ValueError, match="single-column projection"):
        await prepared.fetch_val()
    with pytest.raises(ValueError, match="returns rows"):
        await prepared.execute()


@pytest.mark.asyncio
async def test_prepared_update_binds_set_and_where() -> None:
    adapter = FakeAdapter()
    prepared = (
        UserUpdate(adapter)
        ._set_values(email=Param("email"))  # pyright: ignore[reportPrivateUsage]
        .where(id_col == Param("id"))
        .prepare()
    )

    assert prepared.sql == 'UPDATE "users" SET "email" = $1 WHERE "users"."id" = $2'
    assert prepared.param_names == ("email", "id")
    assert await prepared.execute(email="new@example.com", id=4) == 3
    assert adapter.calls[0][2] == ["new@example.com", 4]


@pytest.mark.asyncio
async def test_prepared_insert_without_returning_requires_execute() -> None:
    adapter = FakeAdapter()
    prepared = UserInsert(adapter).values(email=Param("email"), id=1).prepare()

    with pytest.raises(ValueError, match="does not return rows"):
        await prepared.all()
    assert await prepared.execute(email="a@example.com") == 3
    assert adapter.calls[0][2] == ["a@example.com", 1]


@pytest.mark.asyncio
async def test_prepared_insert_returning_hydrates_rows() -> None:
    adapter = FakeAdapter()
    prepared = (
        UserInsert(adapter).values(email=Param("email"), id=Param("id")).returning().prepare()
    )

    rows = await prepared.all(email="a@example.com", id=1)
    assert rows == [UserRecord(1, "a@example.com"), UserRecord(2, "b@example.com")]
    with pytest.raises(ValueError, match="returns rows"):
        await prepared.execute(email="a@example.com", id=1)


def test_compiled_query_is_not_exported_from_builder_by_accident() -> None:
    adapter = FakeAdapter()
    prepared = UserQuery(adapter).prepare()
    assert isinstance(prepared, CompiledQuery)


@pytest.mark.asyncio
async def test_prepare_binds_fixed_arity_in_list_slots() -> None:
    adapter = FakeAdapter()
    prepared = (
        UserQuery(adapter).where(id_col.in_([Param[int]("first"), Param[int]("second")])).prepare()
    )

    assert prepared.param_names == ("first", "second")
    assert prepared.sql.endswith('"users"."id" IN ($1, $2)')
    await prepared.all(first=1, second=2)
    assert adapter.calls[0][2] == [1, 2]


def test_param_inside_in_list_requires_prepare() -> None:
    adapter = FakeAdapter()
    query = UserQuery(adapter).where(id_col.in_([Param[int]("first")]))
    with pytest.raises(ValueError, match="requires a precompiled query"):
        query.to_sql()


class UserDelete(BaseDeleteQuery[UserRecord]):
    __slots__ = ()
    _RETURNING_COLS = '"id", "email"'
    _ROW_MAPPER = UserRecord.from_row


@pytest.mark.asyncio
async def test_prepared_delete_binds_where_values() -> None:
    adapter = FakeAdapter()
    prepared = UserDelete(adapter, table_name='"users"').where(id_col == Param("id")).prepare()

    assert prepared.sql == 'DELETE FROM "users" WHERE "users"."id" = $1'
    assert prepared.param_names == ("id",)
    assert await prepared.execute(id=7) == 3
    assert adapter.calls[0][2] == [7]


@pytest.mark.asyncio
async def test_prepared_upsert_binds_values_and_conflict_clause() -> None:
    adapter = FakeAdapter(PostgresDialect())
    prepared = (
        UserInsert(adapter)
        .values(email=Param("email"), id=Param("id"))
        .on_conflict("email")
        .do_update(email=Param("email"))
        .prepare()
    )

    assert 'ON CONFLICT ("email") DO UPDATE SET "email" = $3' in prepared.sql
    assert prepared.param_names == ("email", "id")
    assert await prepared.execute(email="a@example.com", id=1) == 3
    assert adapter.calls[0][2] == ["a@example.com", 1, "a@example.com"]


@pytest.mark.asyncio
async def test_prepared_upsert_without_returning_rejects_all() -> None:
    adapter = FakeAdapter(PostgresDialect())
    prepared = (
        UserInsert(adapter).values(email=Param("email")).on_conflict("email").do_nothing().prepare()
    )

    with pytest.raises(ValueError, match="does not return rows"):
        await prepared.all()


def _upper(value: Any) -> Any:
    return str(value).upper()


@pytest.mark.asyncio
async def test_prepared_scalar_fetch_val_applies_column_decoder() -> None:
    adapter = FakeAdapter()
    adapter.scalar = "abc"
    decoded_col: Column[str] = Column("users", "email", decoder=_upper)
    prepared = UserQuery(adapter).select(decoded_col).prepare()

    assert await prepared.fetch_val() == "ABC"


@pytest.mark.asyncio
async def test_prepared_scalar_fetch_val_skips_decoder_for_none() -> None:
    adapter = FakeAdapter()
    adapter.scalar = None
    decoded_col: Column[str] = Column("users", "email", decoder=_upper)
    prepared = UserQuery(adapter).select(decoded_col).prepare()

    assert await prepared.fetch_val() is None


@pytest.mark.asyncio
async def test_compiled_query_without_row_mapper_rejects_row_fetches() -> None:
    adapter = FakeAdapter()
    compiled: CompiledQuery[UserRecord] = CompiledQuery(
        adapter,
        sql="SELECT 1",
        params=[],
        slot_occurrences=[],
        rows_mapper=lambda rows: [],
    )

    with pytest.raises(ValueError, match="does not return rows"):
        await compiled.first()
    with pytest.raises(ValueError, match="returns rows"):
        await compiled.execute()
    assert await compiled.all() == []
