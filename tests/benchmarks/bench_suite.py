#!/usr/bin/env python3
"""Benchmark suite for issue #33: atla vs raw driver vs SQLAlchemy 2.0.

Both ORMs run on the *same* aiosqlite driver, so the numbers isolate ORM /
hydration / compilation overhead instead of driver differences.

Scenarios (best wall time over ``--repeats`` samples, after warm-up; see
``_best_of`` for why the fastest sample is the estimator):

1. ``pk_lookup``      — AOT ``db.user.get(id)`` vs query builder vs ``session.get``
2. ``filtered_query`` — compile-only (builder ``to_sql`` vs prepared template) and
                        fetch (builder vs prepared vs ORM ``select``)
3. ``bulk_fetch``     — full-table hydration vs ORM ``select``
4. ``materialize``    — record construction vs ORM object construction (no DB)
5. ``join_fetch``     — ``join`` vs prepared join vs raw SQL vs ORM joined select
6. ``bulk_insert``    — 10,000 rows: multi-``VALUES``, driver ``executemany``,
                        raw ``executemany``, SQLAlchemy 2.0 insertmanyvalues, and
                        the SQLAlchemy ORM unit-of-work path

Usage:
  uv run python tests/benchmarks/bench_suite.py
  uv run python tests/benchmarks/bench_suite.py --rows 50000 --lookups 2000
  uv run python tests/benchmarks/bench_suite.py --strict   # non-zero exit on a missed target

Design targets (issue #33): atla stays within ~15% of the raw driver and beats
SQLAlchemy 2.0 by 5x-20x on bulk fetch and 5x on the ORM unit-of-work bulk insert.
Round-trip-bound scenarios (PK lookup, a 500-row page) and join hydration cannot
reach 5x against a matched driver, so they are printed as measured ratios.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import importlib
import json
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from itertools import starmap
from pathlib import Path
from typing import Any, cast

import aiosqlite

from atla.adapter.aiosqlite import AiosqliteAdapter
from atla.expressions import Param
from atlagen.cli import main as atlagen_main

SCHEMA = {
    "schemas": [
        {
            "name": "main",
            "tables": [
                {
                    "name": "users",
                    "columns": [
                        {
                            "name": "id",
                            "type": "INTEGER",
                            "null": False,
                            "attrs": [{"name": "auto_increment"}],
                        },
                        {"name": "email", "type": "TEXT", "null": False},
                        {"name": "name", "type": "TEXT", "null": False},
                    ],
                    "primary_key": {"columns": [{"name": "id"}]},
                    "indexes": [
                        {
                            "name": "users_email_key",
                            "unique": True,
                            "parts": [{"column": {"name": "email"}}],
                        }
                    ],
                },
                {
                    "name": "posts",
                    "columns": [
                        {
                            "name": "id",
                            "type": "INTEGER",
                            "null": False,
                            "attrs": [{"name": "auto_increment"}],
                        },
                        {"name": "user_id", "type": "INTEGER", "null": False},
                        {"name": "title", "type": "TEXT", "null": False},
                    ],
                    "primary_key": {"columns": [{"name": "id"}]},
                    "foreign_keys": [
                        {
                            "name": "posts_user_id_fkey",
                            "columns": [{"name": "user_id"}],
                            "ref_columns": [{"name": "id"}],
                            "ref_table": {"name": "users"},
                        }
                    ],
                },
            ],
        }
    ]
}


@dataclass(slots=True, frozen=True)
class Timing:
    label: str
    seconds: float


def _best_of(samples: Sequence[float]) -> float:
    """Best sample, the standard estimator when collector pauses can intrude.

    The garbage collector stays enabled (ORM object churn is part of what these
    scenarios compare), so a single generational collection can land inside one
    sample and inflate a median by tens of percent. The fastest run is the one
    least disturbed by that interference, and it is applied to every contender
    identically.
    """
    return min(samples)


def _time_sync(fn: Callable[[], object], repeats: int) -> float:
    # Clear the previous scenario's garbage once so the first sample does not pay
    # for it.
    gc.collect()
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return _best_of(samples)


async def _time_async(fn: Callable[[], Awaitable[object]], repeats: int) -> float:
    gc.collect()
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        await fn()
        samples.append(time.perf_counter() - start)
    return _best_of(samples)


def _ratio(numerator: float, denominator: float) -> float:
    """Ratio that reports infinity instead of raising on a zero measurement."""
    return numerator / denominator if denominator else float("inf")


def _report(title: str, timings: Sequence[Timing], baseline: str) -> None:
    base = _seconds(timings, baseline)
    print(f"\n[{title}]")
    for timing in sorted(timings, key=lambda item: item.seconds):
        factor = _ratio(base, timing.seconds)
        print(f"  {timing.label:<28} {timing.seconds:10.6f}s  {factor:6.2f}x vs {baseline}")


def _seconds(timings: Sequence[Timing], label: str) -> float:
    return next(t.seconds for t in timings if t.label == label)


def _check_slower_than(
    timings: Sequence[Timing],
    *,
    reference: str,
    contender: str,
    max_factor: float,
    failures: list[str],
) -> None:
    """Report whether ``contender`` stays within ``max_factor`` of ``reference``."""
    ratio = _ratio(_seconds(timings, contender), _seconds(timings, reference))
    passed = ratio <= max_factor
    print(
        f"  target {contender} within {max_factor:g}x of {reference}: "
        f"{ratio:.2f}x {'PASS' if passed else 'FAIL'}"
    )
    if not passed:
        failures.append(f"{contender} was {ratio:.2f}x {reference} (want <= {max_factor:g}x)")


def _check_faster_than(
    timings: Sequence[Timing],
    *,
    faster: str,
    slower: str,
    min_speedup: float,
    failures: list[str],
) -> None:
    """Report whether ``faster`` beats ``slower`` by at least ``min_speedup``."""
    ratio = _ratio(_seconds(timings, slower), _seconds(timings, faster))
    passed = ratio >= min_speedup
    print(
        f"  target {faster} vs {slower}: {ratio:.2f}x faster "
        f"(want >= {min_speedup:g}) {'PASS' if passed else 'FAIL'}"
    )
    if not passed:
        failures.append(f"{faster} was only {ratio:.2f}x faster than {slower}")


def _note(
    timings: Sequence[Timing],
    *,
    baseline: str,
    contender: str,
    comment: str = "",
) -> None:
    """Print ``contender / baseline`` for context that is not a design target."""
    ratio = _ratio(_seconds(timings, contender), _seconds(timings, baseline))
    suffix = f" ({comment})" if comment else ""
    print(f"  note {contender} vs {baseline}: {ratio:.2f}x{suffix}")


def _check_same_rows(
    left: Sequence[Any],
    right: Sequence[Any],
    *,
    left_label: str,
    right_label: str,
    left_key: Callable[[Any], Any],
    right_key: Callable[[Any], Any],
) -> None:
    """Fail loudly when two contenders did not fetch the same rows.

    Timings alone cannot distinguish "faster" from "fetched less", so every fetch
    scenario compares a key projection of the atla and baseline results once
    during warm-up. Contenders return different shapes (atla records vs ORM
    tuples), so each side projects its own keys.
    """
    left_keys = [left_key(row) for row in left]
    right_keys = [right_key(row) for row in right]
    if left_keys != right_keys:
        msg = (
            f"{left_label} returned {len(left_keys)} row(s) and {right_label} "
            f"returned {len(right_keys)}; keys differ"
        )
        raise RuntimeError(msg)


def _load_generated(tmp_dir: Path) -> Any:
    """Generate the atla package for the benchmark schema and import it."""
    schema_path = tmp_dir / "schema.json"
    schema_path.write_text(json.dumps(SCHEMA), encoding="utf-8")
    out = tmp_dir / "benchapp"
    code = atlagen_main(
        [
            "generate",
            "--dialect",
            "sqlite",
            "--schema",
            str(schema_path),
            "--out",
            str(out),
        ]
    )
    if code != 0:
        msg = "atlagen generate failed"
        raise RuntimeError(msg)
    sys.path.insert(0, str(tmp_dir))
    try:
        # A second run() in the same process must load the freshly generated
        # package, not the one cached by the previous call.
        stale = [name for name in sys.modules if name == "benchapp" or name.startswith("benchapp.")]
        for module_name in stale:
            del sys.modules[module_name]
        return importlib.import_module("benchapp")
    finally:
        sys.path.remove(str(tmp_dir))


class SqlAlchemyModels:
    """SQLAlchemy 2.0 async ORM models on the same aiosqlite driver as atla."""

    def __init__(self, db_path: Path) -> None:
        from sqlalchemy import Column, ForeignKey, Integer, String, insert, select
        from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
        from sqlalchemy.orm import declarative_base

        self.select = select
        self.insert = insert
        base = declarative_base()

        class User(base):  # type: ignore[misc, valid-type]
            __tablename__ = "users"
            id = Column(Integer, primary_key=True)
            email = Column(String)
            name = Column(String)

        class Post(base):  # type: ignore[misc, valid-type]
            __tablename__ = "posts"
            id = Column(Integer, primary_key=True)
            user_id = Column(Integer, ForeignKey("users.id"))
            title = Column(String)

        # Any-typed handles keep the ORM class attributes usable in expressions
        # under mypy --strict (declarative attributes are dynamically generated).
        self.User: Any = User
        self.Post: Any = Post
        self.engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
        self._session_factory = AsyncSession

    def session(self) -> Any:
        return self._session_factory(self.engine, expire_on_commit=False)

    async def dispose(self) -> None:
        await self.engine.dispose()


async def _prepare_db(db_path: Path, rows: int) -> aiosqlite.Connection:
    conn = await aiosqlite.connect(db_path)
    await conn.execute(
        """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL
        )
        """
    )
    await conn.execute(
        """
        CREATE TABLE posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id),
            title TEXT NOT NULL
        )
        """
    )
    await conn.executemany(
        "INSERT INTO users (id, email, name) VALUES (?, ?, ?)",
        [(i, f"user{i}@example.com", f"User {i % 100}") for i in range(rows)],
    )
    await conn.executemany(
        "INSERT INTO posts (id, user_id, title) VALUES (?, ?, ?)",
        [(i, i // 2, f"Post {i}") for i in range(rows * 2)],
    )
    await conn.commit()
    return conn


async def _scenario_pk_lookup(
    db: Any,
    app: Any,
    conn: aiosqlite.Connection,
    sa: SqlAlchemyModels,
    *,
    rows: int,
    lookups: int,
    repeats: int,
    failures: list[str],
) -> None:
    user_model = app.User
    # Keys past the seeded rows would hydrate as None on the atla paths (``get``
    # returns ``UserRecord | None``) while the raw and ORM baselines drop missing
    # rows, so clamp the workload to rows that exist.
    keys = list(range(min(lookups, rows)))

    async def raw() -> list[Any]:
        out: list[Any] = []
        for key in keys:
            cursor = await conn.execute("SELECT id, email, name FROM users WHERE id = ?", (key,))
            try:
                row = await cursor.fetchone()
            finally:
                await cursor.close()
            if row is not None:
                out.append(app.UserRecord(*row))
        return out

    async def atla_get() -> list[Any]:
        return [await db.user.get(key) for key in keys]

    async def atla_builder() -> list[Any]:
        return [await db.user.where(user_model.id == key).first() for key in keys]

    async def sqlalchemy_get() -> list[Any]:
        async with sa.session() as session:
            found = [await session.get(sa.User, key) for key in keys]
            return [(row.id, row.email, row.name) for row in found if row is not None]

    for round_index in range(2):
        raw_rows = await raw()
        atla_rows = await atla_get()
        builder_rows = await atla_builder()
        sa_rows = await sqlalchemy_get()
        if round_index == 0:
            _check_same_rows(
                raw_rows,
                atla_rows,
                left_label="raw_aiosqlite",
                right_label="atla_get",
                left_key=lambda row: row.id,
                right_key=lambda row: row.id,
            )
            _check_same_rows(
                raw_rows,
                builder_rows,
                left_label="raw_aiosqlite",
                right_label="atla_builder",
                left_key=lambda row: row.id,
                right_key=lambda row: row.id,
            )
            _check_same_rows(
                raw_rows,
                sa_rows,
                left_label="raw_aiosqlite",
                right_label="sqlalchemy_get",
                left_key=lambda row: row.id,
                right_key=lambda row: row[0],
            )

    timings = [
        Timing("raw_aiosqlite", await _time_async(raw, repeats)),
        Timing("atla_get", await _time_async(atla_get, repeats)),
        Timing("atla_builder", await _time_async(atla_builder, repeats)),
        Timing("sqlalchemy_get", await _time_async(sqlalchemy_get, repeats)),
    ]
    _report(f"pk_lookup rows={rows} lookups={lookups}", timings, "raw_aiosqlite")
    _check_slower_than(
        timings,
        reference="raw_aiosqlite",
        contender="atla_get",
        max_factor=1.15,
        failures=failures,
    )
    # A PK lookup is one round trip; no layer can be 5x faster than the driver,
    # so this stays a measured ratio rather than an asserted target.
    _note(timings, baseline="atla_get", contender="sqlalchemy_get", comment="round-trip bound")


async def _scenario_filtered_query(
    db: Any,
    app: Any,
    sa: SqlAlchemyModels,
    *,
    repeats: int,
    failures: list[str],
) -> None:
    user_model = app.User
    limit = 500

    def build_and_compile() -> str:
        query = (
            db.user.where(user_model.name == "User 1", user_model.id > 0)
            .order_by(user_model.email.asc())
            .limit(limit)
        )
        sql, _ = query.to_sql()
        return cast(str, sql)

    prepared = (
        db.user.where(user_model.name == Param("name"), user_model.id > Param("min_id"))
        .order_by(user_model.email.asc())
        .limit(limit)
        .prepare()
    )

    def bind_only() -> list[Any]:
        # ``bind`` returns the driver-ready parameter tuple, reused as the
        # "template reuse only" workload below.
        return list(prepared.bind(name="User 1", min_id=0))

    async def atla_builder() -> list[Any]:
        return list(
            await db.user.where(user_model.name == "User 1", user_model.id > 0)
            .order_by(user_model.email.asc())
            .limit(limit)
            .all()
        )

    async def atla_prepared() -> list[Any]:
        return list(await prepared.all(name="User 1", min_id=0))

    async def sqlalchemy_orm() -> list[Any]:
        async with sa.session() as session:
            result = await session.scalars(
                sa.select(sa.User)
                .where(sa.User.name == "User 1", sa.User.id > 0)
                .order_by(sa.User.email.asc())
                .limit(limit)
            )
            return [(row.id, row.email, row.name) for row in result.all()]

    for round_index in range(3):
        build_and_compile()
        bind_only()
        builder_rows = await atla_builder()
        prepared_rows = await atla_prepared()
        sa_rows = await sqlalchemy_orm()
        if round_index == 0:
            _check_same_rows(
                prepared_rows,
                builder_rows,
                left_label="atla_prepared",
                right_label="atla_builder",
                left_key=lambda row: row.id,
                right_key=lambda row: row.id,
            )
            _check_same_rows(
                prepared_rows,
                sa_rows,
                left_label="atla_prepared",
                right_label="sqlalchemy_orm",
                left_key=lambda row: row.id,
                right_key=lambda row: row[0],
            )

    compile_timings = [
        Timing("builder_to_sql", _time_sync(build_and_compile, repeats * 50)),
        Timing("prepared_bind", _time_sync(bind_only, repeats * 50)),
    ]
    _report("filtered_query compile-only", compile_timings, "builder_to_sql")
    _check_faster_than(
        compile_timings,
        faster="prepared_bind",
        slower="builder_to_sql",
        min_speedup=5.0,
        failures=failures,
    )

    fetch_timings = [
        Timing("atla_builder", await _time_async(atla_builder, repeats)),
        Timing("atla_prepared", await _time_async(atla_prepared, repeats)),
        Timing("sqlalchemy_orm", await _time_async(sqlalchemy_orm, repeats)),
    ]
    _report(f"filtered_query fetch (where+order_by+limit {limit})", fetch_timings, "atla_prepared")
    _note(
        fetch_timings,
        baseline="atla_prepared",
        contender="sqlalchemy_orm",
        comment="small page: scan/transfer bound",
    )


async def _scenario_bulk_fetch(
    db: Any,
    app: Any,
    conn: aiosqlite.Connection,
    sa: SqlAlchemyModels,
    *,
    rows: int,
    repeats: int,
    failures: list[str],
) -> None:
    async def raw() -> list[Any]:
        fetched = await conn.execute_fetchall("SELECT id, email, name FROM users")
        return list(starmap(app.UserRecord, fetched))

    async def atla() -> list[Any]:
        return list(await db.user.query().all())

    async def sqlalchemy_orm() -> list[Any]:
        async with sa.session() as session:
            result = await session.scalars(sa.select(sa.User))
            return [(row.id, row.email, row.name) for row in result.all()]

    for round_index in range(2):
        raw_rows = await raw()
        atla_rows = await atla()
        sa_rows = await sqlalchemy_orm()
        if round_index == 0:
            _check_same_rows(
                raw_rows,
                atla_rows,
                left_label="raw_aiosqlite",
                right_label="atla_query_all",
                left_key=lambda row: row.id,
                right_key=lambda row: row.id,
            )
            _check_same_rows(
                raw_rows,
                sa_rows,
                left_label="raw_aiosqlite",
                right_label="sqlalchemy_orm",
                left_key=lambda row: row.id,
                right_key=lambda row: row[0],
            )

    timings = [
        Timing("raw_aiosqlite", await _time_async(raw, repeats)),
        Timing("atla_query_all", await _time_async(atla, repeats)),
        Timing("sqlalchemy_orm", await _time_async(sqlalchemy_orm, repeats)),
    ]
    _report(f"bulk_fetch rows={rows}", timings, "raw_aiosqlite")
    _check_slower_than(
        timings,
        reference="raw_aiosqlite",
        contender="atla_query_all",
        max_factor=1.15,
        failures=failures,
    )
    _check_faster_than(
        timings,
        faster="atla_query_all",
        slower="sqlalchemy_orm",
        min_speedup=5.0,
        failures=failures,
    )


async def _scenario_materialize(
    app: Any,
    sa: SqlAlchemyModels,
    *,
    rows: int,
    repeats: int,
    failures: list[str],
) -> None:
    """Hydration-only comparison: no database round trip at all."""
    source = [(i, f"user{i}@example.com", f"User {i % 100}") for i in range(rows)]

    def atla_materialize() -> list[Any]:
        return list(starmap(app.UserRecord, source))

    def sqlalchemy_materialize() -> list[Any]:
        return [sa.User(id=row[0], email=row[1], name=row[2]) for row in source]

    for _ in range(3):
        atla_materialize()
        sqlalchemy_materialize()

    timings = [
        Timing("atla_materialize", _time_sync(atla_materialize, repeats)),
        Timing("sqlalchemy_materialize", _time_sync(sqlalchemy_materialize, repeats)),
    ]
    _report(f"materialize rows={rows} (no DB round trip)", timings, "atla_materialize")
    _check_faster_than(
        timings,
        faster="atla_materialize",
        slower="sqlalchemy_materialize",
        min_speedup=5.0,
        failures=failures,
    )


async def _scenario_join_fetch(
    db: Any,
    app: Any,
    conn: aiosqlite.Connection,
    sa: SqlAlchemyModels,
    *,
    repeats: int,
    failures: list[str],
) -> None:
    user_model = app.User
    post_model = app.Post
    join_sql = (
        "SELECT users.id, users.email, users.name, posts.id, posts.user_id, posts.title "
        "FROM users JOIN posts ON posts.user_id = users.id"
    )
    prepared = db.user.query().join(post_model, post_model.user_id == user_model.id).prepare()

    async def raw() -> list[Any]:
        fetched = await conn.execute_fetchall(join_sql)
        return [
            (app.UserRecord(row[0], row[1], row[2]), app.PostRecord(row[3], row[4], row[5]))
            for row in fetched
        ]

    async def atla_join() -> list[Any]:
        return list(
            await db.user.query().join(post_model, post_model.user_id == user_model.id).all()
        )

    async def atla_join_prepared() -> list[Any]:
        return list(await prepared.all())

    async def sqlalchemy_orm() -> list[Any]:
        async with sa.session() as session:
            result = await session.execute(
                sa.select(sa.User, sa.Post).join(sa.Post, sa.Post.user_id == sa.User.id)
            )
            return [(user.id, user.email, post.id, post.title) for user, post in result.all()]

    for round_index in range(2):
        raw_rows = await raw()
        join_rows = await atla_join()
        prepared_rows = await atla_join_prepared()
        sa_rows = await sqlalchemy_orm()
        if round_index == 0:
            _check_same_rows(
                raw_rows,
                join_rows,
                left_label="raw_aiosqlite",
                right_label="atla_join",
                left_key=lambda row: (row[0].id, row[1].id),
                right_key=lambda row: (row[0].id, row[1].id),
            )
            _check_same_rows(
                join_rows,
                prepared_rows,
                left_label="atla_join",
                right_label="atla_join_prepared",
                left_key=lambda row: (row[0].id, row[1].id),
                right_key=lambda row: (row[0].id, row[1].id),
            )
            _check_same_rows(
                join_rows,
                sa_rows,
                left_label="atla_join",
                right_label="sqlalchemy_orm",
                left_key=lambda row: (row[0].id, row[1].id),
                right_key=lambda row: (row[0], row[2]),
            )

    timings = [
        Timing("raw_aiosqlite", await _time_async(raw, repeats)),
        Timing("atla_join", await _time_async(atla_join, repeats)),
        Timing("atla_join_prepared", await _time_async(atla_join_prepared, repeats)),
        Timing("sqlalchemy_orm", await _time_async(sqlalchemy_orm, repeats)),
    ]
    _report("join_fetch", timings, "raw_aiosqlite")
    # Join hydration is allocation-bound for every layer and swings between ~2.4x
    # and ~3.6x against the ORM, so it is reported as a measured ratio: JOIN is not
    # one of issue #33's acceptance scenarios and a floor here would be noise.
    _note(
        timings,
        baseline="atla_join_prepared",
        contender="sqlalchemy_orm",
        comment="allocation-bound",
    )


async def _scenario_bulk_insert(
    db: Any,
    conn: aiosqlite.Connection,
    sa: SqlAlchemyModels,
    *,
    rows: int,
    repeats: int,
    failures: list[str],
) -> None:
    payload = [{"id": i, "email": f"bulk{i}@example.com", "name": f"Bulk {i}"} for i in range(rows)]
    sa_payload = [dict(row) for row in payload]

    async def reset() -> None:
        # posts reference users; clear both regardless of FK enforcement.
        await conn.execute("DELETE FROM posts")
        await conn.execute("DELETE FROM users")
        await conn.commit()

    async def raw_executemany() -> None:
        await conn.execute("BEGIN")
        try:
            await conn.executemany(
                "INSERT INTO users (id, email, name) VALUES (?, ?, ?)",
                [(row["id"], row["email"], row["name"]) for row in payload],
            )
            await conn.commit()
        except BaseException:
            await conn.rollback()
            raise
        return None

    async def atla_execute() -> None:
        # Use the client yielded by transaction(): it is pinned to the tx connection.
        async with db.transaction() as tx:
            await tx.user.insert_many(payload).execute()
        return None

    async def atla_execute_many() -> None:
        async with db.transaction() as tx:
            await tx.user.insert_many(payload).execute_many()
        return None

    async def sqlalchemy_bulk() -> None:
        async with sa.session() as session:
            await session.execute(sa.insert(sa.User), sa_payload)
            await session.commit()
        return None

    async def sqlalchemy_orm_flush() -> None:
        async with sa.session() as session:
            session.add_all(
                [sa.User(id=row["id"], email=row["email"], name=row["name"]) for row in payload]
            )
            await session.commit()
        return None

    runners: dict[str, Callable[[], Awaitable[object]]] = {
        "raw_executemany": raw_executemany,
        "atla_execute": atla_execute,
        "atla_execute_many": atla_execute_many,
        "sqlalchemy_insertmanyvalues": sqlalchemy_bulk,
        "sqlalchemy_orm_flush": sqlalchemy_orm_flush,
    }

    gc.collect()

    async def stored_rows() -> int:
        cursor = await conn.execute("SELECT COUNT(*) FROM users")
        try:
            row = await cursor.fetchone()
        finally:
            await cursor.close()
        if row is None:
            msg = "SELECT COUNT(*) returned no row"
            raise RuntimeError(msg)
        return int(row[0])

    samples: dict[str, list[float]] = {label: [] for label in runners}
    for label, runner in runners.items():
        for _ in range(2):
            await reset()
            await runner()
        for _ in range(repeats):
            await reset()
            start = time.perf_counter()
            await runner()
            samples[label].append(time.perf_counter() - start)
            # Count the rows in the table rather than trusting each runner's own
            # (sometimes constant) return value, so a partial insert is caught.
            stored = await stored_rows()
            if stored != rows:
                msg = f"{label} stored {stored} of {rows} rows"
                raise RuntimeError(msg)

    timings = [Timing(label, _best_of(values)) for label, values in samples.items()]
    _report(f"bulk_insert rows={rows} (single transaction each)", timings, "raw_executemany")
    # SQLAlchemy 2.0's tuned bulk path (insertmanyvalues) is within ~1.5x of raw;
    # the ORM unit-of-work path is where the order-of-magnitude gap shows up.
    _note(
        timings,
        baseline="atla_execute",
        contender="sqlalchemy_insertmanyvalues",
        comment="SQLAlchemy tuned bulk path",
    )
    _check_faster_than(
        timings,
        faster="atla_execute",
        slower="sqlalchemy_orm_flush",
        min_speedup=5.0,
        failures=failures,
    )


async def run(rows: int, lookups: int, repeats: int) -> list[str]:
    print(
        f"atla benchmark suite: rows={rows} lookups={lookups} repeats={repeats} "
        "(SQLAlchemy 2.0 async ORM on the same aiosqlite driver; best of "
        f"{repeats} samples per figure)"
    )
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        db_path = tmp_dir / "bench.db"
        conn = await _prepare_db(db_path, rows)
        sa = SqlAlchemyModels(db_path)
        adapter = AiosqliteAdapter(conn)
        try:
            app = _load_generated(tmp_dir)
            db = app.AtlaClient(adapter)
            await _scenario_pk_lookup(
                db,
                app,
                conn,
                sa,
                rows=rows,
                lookups=lookups,
                repeats=repeats,
                failures=failures,
            )
            await _scenario_filtered_query(
                db,
                app,
                sa,
                repeats=repeats,
                failures=failures,
            )
            await _scenario_bulk_fetch(
                db,
                app,
                conn,
                sa,
                rows=rows,
                repeats=repeats,
                failures=failures,
            )
            await _scenario_materialize(
                app,
                sa,
                rows=rows,
                repeats=repeats,
                failures=failures,
            )
            await _scenario_join_fetch(
                db,
                app,
                conn,
                sa,
                repeats=repeats,
                failures=failures,
            )
            await _scenario_bulk_insert(
                db,
                conn,
                sa,
                rows=rows,
                repeats=repeats,
                failures=failures,
            )
        finally:
            await sa.dispose()
            await conn.close()
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    # 60k keeps the hydration-bound ratios clear of the 5x line's run-to-run noise.
    parser.add_argument("--rows", type=int, default=60_000)
    parser.add_argument("--lookups", type=int, default=1_000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when a design target misses (default: report only)",
    )
    args = parser.parse_args()
    if args.rows < 1 or args.lookups < 1 or args.repeats < 1:
        parser.error("--rows, --lookups and --repeats must all be >= 1")
    failures = asyncio.run(run(args.rows, args.lookups, args.repeats))
    if failures:
        print("\nmissed design target(s):")
        for failure in failures:
            print(f"  - {failure}")
        if args.strict:
            raise SystemExit(1)
        print("re-run with --strict to fail the process on these.")


if __name__ == "__main__":
    main()
