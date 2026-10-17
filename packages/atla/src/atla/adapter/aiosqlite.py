"""aiosqlite adapter for SQLite."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from atla.adapter.base import IsolationLevel
from atla.adapter.translate import reraise_translated, with_translated_errors
from atla.dialect import Dialect, SQLiteDialect


class AiosqliteAdapter:
    """Adapter over an ``aiosqlite.Connection`` with SAVEPOINT nesting.

    Outside ``transaction()``, writes are committed immediately so persistence
    matches asyncpg-style autocommit. Nested calls on the same adapter instance
    while a transaction is active raise ``RuntimeError``; use the yielded
    adapter to nest via SAVEPOINT.

    A single connection is serialized with an ``asyncio.Lock``. Outer-adapter
    operations check ``_tx_active`` under that lock so peer tasks cannot join
    an in-flight transaction.
    """

    __slots__ = (
        "_conn",
        "_depth",
        "_dialect",
        "_foreign_keys",
        "_lock",
        "_needs_configure",
        "_tx_active",
    )

    def __init__(
        self,
        connection: Any,
        *,
        dialect: Dialect | None = None,
        foreign_keys: bool = True,
        _depth: int = 0,
        _configure: bool = True,
        _tx_active: bool = False,
        _lock: asyncio.Lock | None = None,
    ) -> None:
        self._conn = connection
        self._dialect = dialect or SQLiteDialect()
        if not isinstance(self._dialect, SQLiteDialect):
            msg = "AiosqliteAdapter requires SQLiteDialect"
            raise TypeError(msg)
        self._depth = _depth
        self._foreign_keys = foreign_keys
        self._needs_configure = _configure
        self._tx_active = _tx_active
        # Share one lock across outer and transaction-pinned adapters.
        self._lock = _lock if _lock is not None else asyncio.Lock()

    @property
    def dialect(self) -> Dialect:
        return self._dialect

    def _ensure_not_in_active_transaction(self) -> None:
        """Reject ops on the outer adapter while a transaction is open.

        Must be called while holding ``self._lock``. The yielded (pinned)
        adapter has ``_depth > 0`` and is the only allowed entry point for the
        connection during that window.
        """
        if self._depth == 0 and self._tx_active:
            msg = (
                "connection is checked out for an active transaction; "
                "use the adapter yielded by transaction()"
            )
            raise RuntimeError(msg)

    async def _ensure_configured_unlocked(self) -> None:
        """Apply one-time PRAGMA setup. Caller must hold ``self._lock``."""
        if not self._needs_configure:
            return
        if self._foreign_keys:
            await self._conn.execute("PRAGMA foreign_keys = ON")
        self._needs_configure = False

    async def _commit_if_autocommit(self) -> None:
        if self._depth == 0 and not self._tx_active:
            await self._conn.commit()

    @with_translated_errors
    async def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Sequence[Any]]:
        async with self._lock:
            self._ensure_not_in_active_transaction()
            await self._ensure_configured_unlocked()
            encoded = self._dialect.encode_params(params)
            if encoded:
                rows: Sequence[Sequence[Any]] = await self._conn.execute_fetchall(sql, encoded)
            else:
                rows = await self._conn.execute_fetchall(sql)
            return rows

    @with_translated_errors
    async def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Any] | None:
        async with self._lock:
            self._ensure_not_in_active_transaction()
            await self._ensure_configured_unlocked()
            encoded = self._dialect.encode_params(params)
            cursor = await self._conn.execute(sql, encoded)
            try:
                row: Sequence[Any] | None = await cursor.fetchone()
                return row
            finally:
                await cursor.close()

    @with_translated_errors
    async def fetch_val(self, sql: str, params: Sequence[Any] = ()) -> Any:
        row = await self.fetch_one(sql, params)
        if row is None:
            return None
        return row[0]

    @with_translated_errors
    async def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        async with self._lock:
            self._ensure_not_in_active_transaction()
            await self._ensure_configured_unlocked()
            encoded = self._dialect.encode_params(params)
            cursor = await self._conn.execute(sql, encoded)
            try:
                rowcount = int(cursor.rowcount)
            finally:
                await cursor.close()
            await self._commit_if_autocommit()
        return rowcount

    @with_translated_errors
    async def execute_many(self, sql: str, params_seq: Sequence[Sequence[Any]]) -> None:
        async with self._lock:
            self._ensure_not_in_active_transaction()
            await self._ensure_configured_unlocked()
            encoded_seq = self._dialect.encode_params_seq(params_seq)
            await self._conn.executemany(sql, encoded_seq)
            await self._commit_if_autocommit()

    async def _exec_control(self, sql: str) -> None:
        cursor = await self._conn.execute(sql)
        await cursor.close()

    @asynccontextmanager
    async def transaction(
        self,
        *,
        isolation: IsolationLevel | None = None,
        readonly: bool = False,
        deferrable: bool = False,
    ) -> AsyncGenerator[AiosqliteAdapter]:
        if isolation is not None:
            msg = "SQLite does not support isolation/readonly/deferrable transaction options"
            raise ValueError(msg)
        if readonly:
            msg = "SQLite does not support readonly transactions"
            raise ValueError(msg)
        if deferrable:
            msg = "SQLite does not support deferrable transactions"
            raise ValueError(msg)
        if self._depth == 0:
            async with self._lock:
                await self._ensure_configured_unlocked()
                if self._tx_active:
                    msg = (
                        "transaction already active on this adapter; "
                        "use the adapter yielded by transaction() for nesting"
                    )
                    raise RuntimeError(msg)
                self._tx_active = True
                try:
                    await self._exec_control("BEGIN")
                except BaseException:
                    self._tx_active = False
                    raise
            pinned = AiosqliteAdapter(
                self._conn,
                dialect=self._dialect,
                foreign_keys=self._foreign_keys,
                _depth=1,
                _configure=False,
                _lock=self._lock,
            )
            try:
                yield pinned
            except BaseException as exc:
                async with self._lock:
                    try:
                        await self._exec_control("ROLLBACK")
                    finally:
                        self._tx_active = False
                if isinstance(exc, Exception):
                    reraise_translated(exc)
                raise
            else:
                async with self._lock:
                    try:
                        try:
                            await self._exec_control("COMMIT")
                        except Exception as exc:
                            reraise_translated(exc)
                    finally:
                        self._tx_active = False
            return

        savepoint = f"atla_sp_{self._depth}"
        async with self._lock:
            await self._exec_control(f"SAVEPOINT {savepoint}")
        pinned = AiosqliteAdapter(
            self._conn,
            dialect=self._dialect,
            foreign_keys=self._foreign_keys,
            _depth=self._depth + 1,
            _configure=False,
            _lock=self._lock,
        )
        try:
            yield pinned
        except BaseException as exc:
            async with self._lock:
                await self._exec_control(f"ROLLBACK TO SAVEPOINT {savepoint}")
                await self._exec_control(f"RELEASE SAVEPOINT {savepoint}")
            if isinstance(exc, Exception):
                reraise_translated(exc)
            raise
        else:
            async with self._lock:
                await self._exec_control(f"RELEASE SAVEPOINT {savepoint}")
