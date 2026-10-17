"""aiosqlite adapter for SQLite."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from atla.dialect import Dialect, SQLiteDialect


class AiosqliteAdapter:
    """Adapter over an ``aiosqlite.Connection`` with SAVEPOINT nesting."""

    __slots__ = ("_conn", "_depth", "_dialect", "_foreign_keys", "_needs_configure")

    def __init__(
        self,
        connection: Any,
        *,
        dialect: Dialect | None = None,
        foreign_keys: bool = True,
        _depth: int = 0,
        _configure: bool = True,
    ) -> None:
        self._conn = connection
        self._dialect = dialect or SQLiteDialect()
        if not isinstance(self._dialect, SQLiteDialect):
            msg = "AiosqliteAdapter requires SQLiteDialect"
            raise TypeError(msg)
        self._depth = _depth
        self._foreign_keys = foreign_keys
        self._needs_configure = _configure

    @property
    def dialect(self) -> Dialect:
        return self._dialect

    async def _ensure_configured(self) -> None:
        if not self._needs_configure:
            return
        if self._foreign_keys:
            await self._conn.execute("PRAGMA foreign_keys = ON")
        self._needs_configure = False

    async def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Sequence[Any]]:
        await self._ensure_configured()
        cursor = await self._conn.execute(sql, tuple(params))
        try:
            rows: Sequence[Sequence[Any]] = await cursor.fetchall()
            return rows
        finally:
            await cursor.close()

    async def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Any] | None:
        await self._ensure_configured()
        cursor = await self._conn.execute(sql, tuple(params))
        try:
            row: Sequence[Any] | None = await cursor.fetchone()
            return row
        finally:
            await cursor.close()

    async def fetch_val(self, sql: str, params: Sequence[Any] = ()) -> Any:
        row = await self.fetch_one(sql, params)
        if row is None:
            return None
        return row[0]

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        await self._ensure_configured()
        cursor = await self._conn.execute(sql, tuple(params))
        try:
            return int(cursor.rowcount)
        finally:
            await cursor.close()

    async def execute_many(self, sql: str, params_seq: Sequence[Sequence[Any]]) -> None:
        await self._ensure_configured()
        await self._conn.executemany(sql, [tuple(params) for params in params_seq])

    async def _exec_control(self, sql: str) -> None:
        cursor = await self._conn.execute(sql)
        await cursor.close()

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[AiosqliteAdapter]:
        await self._ensure_configured()
        if self._depth == 0:
            await self._exec_control("BEGIN")
            pinned = AiosqliteAdapter(
                self._conn,
                dialect=self._dialect,
                foreign_keys=self._foreign_keys,
                _depth=1,
                _configure=False,
            )
            try:
                yield pinned
            except BaseException:
                await self._exec_control("ROLLBACK")
                raise
            else:
                await self._exec_control("COMMIT")
            return

        savepoint = f"atla_sp_{self._depth}"
        await self._exec_control(f"SAVEPOINT {savepoint}")
        pinned = AiosqliteAdapter(
            self._conn,
            dialect=self._dialect,
            foreign_keys=self._foreign_keys,
            _depth=self._depth + 1,
            _configure=False,
        )
        try:
            yield pinned
        except BaseException:
            await self._exec_control(f"ROLLBACK TO SAVEPOINT {savepoint}")
            await self._exec_control(f"RELEASE SAVEPOINT {savepoint}")
            raise
        else:
            await self._exec_control(f"RELEASE SAVEPOINT {savepoint}")
