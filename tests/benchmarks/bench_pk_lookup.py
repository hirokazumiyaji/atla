#!/usr/bin/env python3
"""Benchmark AOT ``get`` vs query builder vs raw aiosqlite."""

from __future__ import annotations

import argparse
import asyncio
import statistics
import tempfile
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiosqlite

from atla.adapter.aiosqlite import AiosqliteAdapter
from atla.expressions import Column
from atla.query import BaseSelectQuery


@dataclass(slots=True, frozen=True)
class UserRecord:
    id: int
    email: str

    @classmethod
    def from_row(cls, row: Sequence[Any]) -> UserRecord:
        return cls(*row)


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


class UserRepository:
    __slots__ = ("_adapter",)
    _GET_BY_PK_SQL = (
        'SELECT "users"."id", "users"."email" FROM "users" WHERE "users"."id" = ? LIMIT 1'
    )

    def __init__(self, adapter: AiosqliteAdapter) -> None:
        self._adapter = adapter

    def query(self) -> UserQuery:
        return UserQuery(self._adapter)

    async def get(self, id: int) -> UserRecord | None:
        row = await self._adapter.fetch_one(self._GET_BY_PK_SQL, (id,))
        return None if row is None else UserRecord.from_row(row)


id_col: Column[int] = Column("users", "id")


async def _setup(path: Path, rows: int) -> aiosqlite.Connection:
    conn = await aiosqlite.connect(path)
    await conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT NOT NULL)")
    await conn.executemany(
        "INSERT INTO users (id, email) VALUES (?, ?)",
        [(i, f"u{i}@ex.com") for i in range(rows)],
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


async def run(rows: int, lookups: int, repeats: int) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        conn = await _setup(Path(tmp) / "b.db", rows)
        adapter = AiosqliteAdapter(conn)
        repo = UserRepository(adapter)
        keys = list(range(0, lookups))

        async def raw() -> object:
            out: list[Any] = []
            for key in keys:
                cursor = await conn.execute(
                    "SELECT id, email FROM users WHERE id = ? LIMIT 1", (key,)
                )
                try:
                    row = await cursor.fetchone()
                finally:
                    await cursor.close()
                if row is not None:
                    out.append(UserRecord.from_row(row))
            return out

        async def aot_get() -> object:
            return [await repo.get(key) for key in keys]

        async def builder() -> object:
            return [await repo.query().where(id_col == key).first() for key in keys]

        for _ in range(2):
            await raw()
            await aot_get()
            await builder()

        raw_s = await _time(raw, repeats)
        aot_s = await _time(aot_get, repeats)
        builder_s = await _time(builder, repeats)
        print(f"rows={rows} lookups={lookups} repeats={repeats}")
        print(f"raw_median_s={raw_s:.6f}")
        print(f"aot_get_median_s={aot_s:.6f}")
        print(f"builder_median_s={builder_s:.6f}")
        print(f"aot_vs_raw_ratio={aot_s / raw_s:.3f}")
        print(f"builder_vs_aot_ratio={builder_s / aot_s:.3f}")
        await conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--lookups", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(run(args.rows, args.lookups, args.repeats))


if __name__ == "__main__":
    main()
