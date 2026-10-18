"""PostgreSQL / MySQL E2E tests (skipped unless DSNs are provided).

These tests DROP and recreate ``users`` / ``posts`` in the configured database.
Only dedicated test database names are accepted (see ``_DEDICATED_TEST_DBS``).
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import pytest

from atla.expressions import Param
from atlagen.cli import main

POSTGRES_DSN = os.environ.get("ATLA_TEST_POSTGRES_DSN")
MYSQL_DSN = os.environ.get("ATLA_TEST_MYSQL_DSN")

# Database names that are safe to wipe when running destructive E2E setup.
_DEDICATED_TEST_DBS = frozenset({"atla", "atla_test"})


def _require_dedicated_test_db(dsn: str, *, label: str) -> str:
    """Return the DB name, or skip if the DSN does not target a dedicated test DB."""
    parsed = urlparse(dsn)
    database = unquote((parsed.path or "").lstrip("/")).split("?", 1)[0]
    if database not in _DEDICATED_TEST_DBS:
        pytest.skip(
            f"{label} refuses database {database!r}; "
            f"use a dedicated test DB named one of {sorted(_DEDICATED_TEST_DBS)}"
        )
    return database


SCHEMA = """
{
  "schemas": [{
    "name": "public",
    "tables": [
      {
        "name": "users",
        "columns": [
          {"name": "id", "type": "bigserial", "null": false},
          {"name": "email", "type": "text", "null": false},
          {"name": "name", "type": "text", "null": false}
        ],
        "primary_key": {"columns": [{"name": "id"}]},
        "indexes": [{
          "name": "users_email_key",
          "unique": true,
          "columns": [{"name": "email"}]
        }],
        "foreign_keys": []
      },
      {
        "name": "posts",
        "columns": [
          {"name": "id", "type": "bigserial", "null": false},
          {"name": "user_id", "type": "bigint", "null": false},
          {"name": "title", "type": "text", "null": false}
        ],
        "primary_key": {"columns": [{"name": "id"}]},
        "foreign_keys": [{
          "name": "posts_user_id_fkey",
          "columns": [{"name": "user_id"}],
          "ref_columns": [{"name": "id"}],
          "ref_table": {"name": "users"}
        }]
      }
    ]
  }]
}
"""


def _generate(tmp_path: Path, *, dialect: str) -> None:
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(SCHEMA, encoding="utf-8")
    out = tmp_path / "appdb"
    assert (
        main(
            [
                "generate",
                "--dialect",
                dialect,
                "--schema",
                str(schema_path),
                "--out",
                str(out),
            ]
        )
        == 0
    )


async def _reset_postgres_schema(conn: Any) -> None:
    await conn.execute("DROP TABLE IF EXISTS posts")
    await conn.execute("DROP TABLE IF EXISTS users")
    await conn.execute(
        """
        CREATE TABLE users (
            id BIGSERIAL PRIMARY KEY,
            email TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL
        )
        """
    )
    await conn.execute(
        """
        CREATE TABLE posts (
            id BIGSERIAL PRIMARY KEY,
            user_id BIGINT NOT NULL REFERENCES users(id),
            title TEXT NOT NULL
        )
        """
    )


def _postgres_dsn() -> str:
    assert POSTGRES_DSN is not None
    _require_dedicated_test_db(POSTGRES_DSN, label="ATLA_TEST_POSTGRES_DSN")
    return POSTGRES_DSN


@asynccontextmanager
async def _postgres_adapter() -> AsyncGenerator[Any]:
    import asyncpg  # type: ignore[import-untyped]

    from atla.adapter.asyncpg import AsyncpgAdapter

    conn = await asyncpg.connect(_postgres_dsn())
    try:
        await _reset_postgres_schema(conn)
        yield AsyncpgAdapter(conn)
    finally:
        await conn.close()


@asynccontextmanager
async def _postgres_pool_adapter() -> AsyncGenerator[Any]:
    """Pool-backed adapter: the documented production setup for asyncpg."""
    import asyncpg  # pyright: ignore[reportMissingTypeStubs]

    from atla.adapter.asyncpg import AsyncpgAdapter

    pool = await asyncpg.create_pool(_postgres_dsn(), min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            await _reset_postgres_schema(conn)
        yield AsyncpgAdapter(pool)
    finally:
        await pool.close()


@asynccontextmanager
async def _mysql_adapter() -> AsyncGenerator[Any]:
    import asyncmy  # type: ignore[import-untyped, unused-ignore]

    from atla.adapter.asyncmy import AsyncmyAdapter

    assert MYSQL_DSN is not None
    parsed = urlparse(MYSQL_DSN)
    user = unquote(parsed.username or "")
    password = unquote(parsed.password or "")
    host = parsed.hostname
    database = _require_dedicated_test_db(MYSQL_DSN, label="ATLA_TEST_MYSQL_DSN")
    if not user or not host or not database:
        pytest.skip(f"invalid ATLA_TEST_MYSQL_DSN: {MYSQL_DSN!r}")
    conn = await asyncmy.connect(
        host=host,
        port=parsed.port or 3306,
        user=user,
        password=password,
        database=database,
        autocommit=True,
    )
    try:
        async with conn.cursor() as cursor:
            await cursor.execute("DROP TABLE IF EXISTS posts")
            await cursor.execute("DROP TABLE IF EXISTS users")
            await cursor.execute(
                """
                CREATE TABLE users (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    email VARCHAR(255) NOT NULL,
                    name TEXT NOT NULL,
                    UNIQUE KEY users_email_key (email)
                )
                """
            )
            await cursor.execute(
                """
                CREATE TABLE posts (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    title TEXT NOT NULL,
                    CONSTRAINT posts_user_id_fkey FOREIGN KEY (user_id) REFERENCES users(id)
                )
                """
            )
        yield AsyncmyAdapter(conn)
    finally:
        await conn.ensure_closed()


async def _exercise_client(client: Any, *, models: Any, supports_returning: bool) -> None:
    user_model = models.User
    post_model = models.Post

    if supports_returning:
        inserted = await client.user.insert(email="a@example.com", name="Alice").returning().one()
        user_id = inserted.id
    else:
        await client.user.insert(email="a@example.com", name="Alice").execute()
        inserted = await client.user.where(user_model.email == "a@example.com").one()
        user_id = inserted.id

    assert inserted.email == "a@example.com"

    await client.user.insert_many([{"email": "b@example.com", "name": "Bob"}]).execute()
    await client.post.insert(user_id=user_id, title="hello").execute()

    rows = await client.user.query().join(post_model, post_model.user_id == user_model.id).all()
    assert len(rows) >= 1
    assert rows[0][1].title == "hello"

    preloaded = await client.user.where(user_model.id == user_id).with_posts().one()
    assert preloaded[0].id == user_id
    assert len(preloaded[1]) == 1

    prepared = client.user.where(user_model.email == Param("email")).prepare()
    assert prepared.param_names == ("email",)
    fetched = await prepared.first(email="b@example.com")
    assert fetched is not None and fetched.name == "Bob"
    assert await prepared.first(email="missing@example.com") is None

    async with client.transaction():
        await client.user.update().set(name="Alicia").where(user_model.id == user_id).execute()
        updated = await client.user.where(user_model.id == user_id).one()
        assert updated.name == "Alicia"

    await client.post.delete().where(post_model.user_id == user_id).execute()
    await client.user.delete().where(user_model.id == user_id).execute()


@pytest.mark.skipif(not POSTGRES_DSN, reason="Set ATLA_TEST_POSTGRES_DSN to enable")
@pytest.mark.asyncio
async def test_postgres_e2e(
    tmp_path: Path,
    load_appdb: Callable[[Path], Any],
) -> None:
    _generate(tmp_path, dialect="postgres")
    appdb = load_appdb(tmp_path)

    async with _postgres_adapter() as adapter:
        await _exercise_client(
            appdb.AtlaClient(adapter),
            models=appdb,
            supports_returning=True,
        )


@pytest.mark.skipif(not POSTGRES_DSN, reason="Set ATLA_TEST_POSTGRES_DSN to enable")
@pytest.mark.asyncio
async def test_postgres_insert_many_copy(
    tmp_path: Path,
    load_appdb: Callable[[Path], Any],
) -> None:
    _generate(tmp_path, dialect="postgres")
    appdb = load_appdb(tmp_path)

    async with _postgres_adapter() as adapter:
        client = appdb.AtlaClient(adapter)
        copied = await client.user.insert_many(
            [{"email": f"copy{i}@example.com", "name": f"Copy {i}"} for i in range(500)]
        ).copy()

        assert copied == 500
        assert await client.user.query().count() == 500
        last = await client.user.where(appdb.User.email == "copy499@example.com").one()
        assert last.name == "Copy 499"


@pytest.mark.skipif(not POSTGRES_DSN, reason="Set ATLA_TEST_POSTGRES_DSN to enable")
@pytest.mark.asyncio
async def test_postgres_insert_many_copy_through_pool(
    tmp_path: Path,
    load_appdb: Callable[[Path], Any],
) -> None:
    """asyncpg.Pool proxies copy_records_to_table; the pool path must work too."""
    _generate(tmp_path, dialect="postgres")
    appdb = load_appdb(tmp_path)

    async with _postgres_pool_adapter() as adapter:
        client = appdb.AtlaClient(adapter)
        copied = await client.user.insert_many(
            [{"email": f"pool{i}@example.com", "name": f"Pool {i}"} for i in range(100)]
        ).copy()

        assert copied == 100
        assert await client.user.query().count() == 100


@pytest.mark.skipif(not POSTGRES_DSN, reason="Set ATLA_TEST_POSTGRES_DSN to enable")
@pytest.mark.asyncio
async def test_postgres_copy_targets_the_same_relation_as_insert(
    tmp_path: Path,
    load_appdb: Callable[[Path], Any],
) -> None:
    """COPY must hit the relation the generated (unqualified) SQL resolves to.

    The generated client emits unqualified names, so both ``insert_many().execute()``
    and ``copy()`` resolve through ``search_path``. Point ``search_path`` at a
    non-default schema holding a decoy table of the same name in ``public`` and
    check where each path actually wrote.
    """
    import asyncpg  # pyright: ignore[reportMissingTypeStubs]

    from atla.adapter.asyncpg import AsyncpgAdapter

    _generate(tmp_path, dialect="postgres")
    appdb = load_appdb(tmp_path)

    conn = await asyncpg.connect(_postgres_dsn())
    try:
        await conn.execute("DROP SCHEMA IF EXISTS atla_other CASCADE")
        # posts references users, so drop it first (see _reset_postgres_schema).
        await conn.execute("DROP TABLE IF EXISTS public.posts")
        await conn.execute("DROP TABLE IF EXISTS public.users")
        await conn.execute("CREATE SCHEMA atla_other")
        for schema in ("public", "atla_other"):
            await conn.execute(
                f"""
                CREATE TABLE {schema}.users (
                    id BIGSERIAL PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL
                )
                """
            )
        await conn.execute("SET search_path TO atla_other")
        adapter = AsyncpgAdapter(conn)
        client = appdb.AtlaClient(adapter)

        await client.user.insert_many([{"email": "insert@example.com", "name": "Insert"}]).execute()
        copied = await client.user.insert_many(
            [{"email": "copy@example.com", "name": "Copy"}]
        ).copy()

        assert copied == 1
        other_rows = await conn.fetch("SELECT email FROM atla_other.users ORDER BY email")
        public_rows = await conn.fetch("SELECT email FROM public.users")
        assert [row["email"] for row in other_rows] == ["copy@example.com", "insert@example.com"]
        assert public_rows == []
    finally:
        await conn.execute("DROP SCHEMA IF EXISTS atla_other CASCADE")
        await conn.execute("DROP TABLE IF EXISTS public.users")
        await conn.close()


@pytest.mark.skipif(not MYSQL_DSN, reason="Set ATLA_TEST_MYSQL_DSN to enable")
@pytest.mark.asyncio
async def test_mysql_e2e(
    tmp_path: Path,
    load_appdb: Callable[[Path], Any],
) -> None:
    _generate(tmp_path, dialect="mysql")
    appdb = load_appdb(tmp_path)

    async with _mysql_adapter() as adapter:
        await _exercise_client(
            appdb.AtlaClient(adapter),
            models=appdb,
            supports_returning=False,
        )
