#!/usr/bin/env python3
"""Benchmark atla hydration vs raw aiosqlite and optional SQLAlchemy 2.0.

Usage:
  uv run python tests/benchmarks/bench_fetch.py
  uv run python tests/benchmarks/bench_fetch.py --rows 10000

Design targets (Issue #12 / #32):
  - atla within +15% of raw driver CPU time
  - atla at least 5x faster than SQLAlchemy 2.0 ORM fetch+materialize
  - materialize-only path at least 15x faster than SQLAlchemy construction
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import tempfile
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from itertools import starmap
from pathlib import Path
from typing import Any, cast

import aiosqlite

from atla.adapter.aiosqlite import AiosqliteAdapter
from atla.query import BaseSelectQuery


@dataclass(slots=True, frozen=True)
class UserRecord:
    id: int
    email: str
    name: str

    @classmethod
    def from_row(cls, row: Sequence[Any]) -> UserRecord:
        # Match generated codec-free `_from_row`: unpack only, no casts.
        return cls(*row)


class UserQuery(BaseSelectQuery[UserRecord]):
    __slots__ = ()
    _TABLE_NAME = '"users"'
    _SELECT_COLS = '"users"."id", "users"."email", "users"."name"'
    _FROM_SQL = '"users"'
    _COLUMN_COUNT = 3
    _SELECT_SQL = 'SELECT "users"."id", "users"."email", "users"."name" FROM "users"'
    _COUNT_SQL = 'SELECT COUNT(*) FROM "users"'
    _ROW_MAPPER = UserRecord.from_row
    _ROW_CTOR = UserRecord


async def _setup(db_path: Path, rows: int) -> aiosqlite.Connection:
    conn = await aiosqlite.connect(db_path)
    await conn.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT NOT NULL, name TEXT NOT NULL)"
    )
    await conn.executemany(
        "INSERT INTO users (id, email, name) VALUES (?, ?, ?)",
        [(i, f"user{i}@example.com", f"User {i}") for i in range(rows)],
    )
    await conn.commit()
    return conn


async def _time(fn: Callable[[], Awaitable[object]], repeats: int) -> float:
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        await fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


async def run(rows: int, repeats: int) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "bench.db"
        conn = await _setup(db_path, rows)
        adapter = AiosqliteAdapter(conn)

        async def raw() -> object:
            fetched = await conn.execute_fetchall("SELECT id, email, name FROM users")
            return list(starmap(UserRecord, fetched))

        async def atla() -> object:
            return await UserQuery(cast(Any, adapter)).all()

        # Warm both paths (and aiosqlite PRAGMA / thread pool) before sampling.
        for _ in range(3):
            await raw()
            await atla()

        raw_s = await _time(raw, repeats)
        atla_s = await _time(atla, repeats)
        print(f"rows={rows} repeats={repeats}")
        print(f"raw_aiosqlite_median_s={raw_s:.6f}")
        print(f"atla_median_s={atla_s:.6f}")
        print(f"atla_vs_raw_ratio={atla_s / raw_s:.3f} (target <= 1.15)")

        try:
            from sqlalchemy import Column, Integer, String, create_engine, select
            from sqlalchemy.orm import Session, declarative_base
        except ImportError:
            print("sqlalchemy_not_installed=skip")
            await conn.close()
            return

        base = declarative_base()

        class SAUser(base):  # type: ignore[misc, valid-type]
            __tablename__ = "users"
            id = Column(Integer, primary_key=True)
            email = Column(String)
            name = Column(String)

        sa_path = Path(tmp) / "bench_sa.db"
        sa_conn = await _setup(sa_path, rows)
        await sa_conn.close()
        engine = create_engine(f"sqlite:///{sa_path}")

        def sa() -> object:
            with Session(engine) as session:
                users = list(session.scalars(select(SAUser)))
                # Force attribute materialization for every mapped column.
                return [(user.id, user.email, user.name) for user in users]

        sa()
        samples: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            sa()
            samples.append(time.perf_counter() - start)
        sa_s = float(statistics.median(samples))
        ratio = sa_s / atla_s
        print(f"sqlalchemy_median_s={sa_s:.6f}")
        print(f"sqlalchemy_vs_atla_ratio={ratio:.3f} (target >= 5.0)")
        # Pure construction microbench isolates ORM object overhead from DB I/O.
        # Fixed size keeps GC noise from dominating the ratio at large --rows.
        map_rows = 10_000
        raw_rows = [(i, f"user{i}@example.com", f"User {i}") for i in range(map_rows)]

        def atla_map() -> object:
            return list(starmap(UserRecord, raw_rows))

        def sa_map() -> object:
            return [SAUser(id=r[0], email=r[1], name=r[2]) for r in raw_rows]

        for _ in range(3):
            atla_map()
            sa_map()
        atla_samples: list[float] = []
        sa_samples: list[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            atla_map()
            atla_samples.append(time.perf_counter() - t0)
            t1 = time.perf_counter()
            sa_map()
            sa_samples.append(time.perf_counter() - t1)
        map_atla = statistics.median(atla_samples)
        map_sa = statistics.median(sa_samples)
        print(f"materialize_rows={map_rows}")
        print(f"materialize_atla_median_s={map_atla:.6f}")
        print(f"materialize_sqlalchemy_median_s={map_sa:.6f}")
        print(f"materialize_sqlalchemy_vs_atla_ratio={map_sa / map_atla:.3f} (target >= 15.0)")
        engine.dispose()
        await conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    # Larger fetch sets make SQLAlchemy session/materialize overhead dominate I/O.
    parser.add_argument("--rows", type=int, default=40000)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    asyncio.run(run(args.rows, args.repeats))


if __name__ == "__main__":
    main()
