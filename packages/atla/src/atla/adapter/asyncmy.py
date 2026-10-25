"""asyncmy adapter for MySQL and TiDB."""

from __future__ import annotations

import inspect
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from typing import Any, cast

from atla.adapter._checkout import (
    checkout_scope,
    ensure_not_checked_out,
    new_checkout_owners,
)
from atla.adapter.base import IsolationLevel
from atla.adapter.translate import reraise_translated, with_translated_errors
from atla.dialect import Dialect, MySQLDialect, TiDBDialect
from atla.transaction import begin_sql


class _TxState:
    __slots__ = ("active_depth",)

    def __init__(self, depth: int = 0) -> None:
        self.active_depth = depth


class AsyncmyAdapter:
    """Adapter over ``asyncmy`` pool or connection using tuple cursors.

    Pool-backed instances are safe to share across concurrent tasks. While a
    task holds ``async with adapter.transaction()``, that same task must use
    the yielded adapter (not the outer pool adapter). Peer tasks — including
    children created with ``asyncio.create_task`` — may still acquire from the
    pool.

    .. warning::
        With a small pool (e.g. ``maxsize=1``), a parent that holds a
        transaction and awaits a child task which uses the *outer* adapter can
        hang indefinitely waiting for a free connection. Previously this raised
        ``RuntimeError``; now the child blocks in ``pool.acquire()`` until the
        parent releases its connection. Size the pool for nested concurrency or
        pass the yielded adapter to children instead.
    """

    __slots__ = (
        "_checkout_owners",
        "_conn",
        "_depth",
        "_dialect",
        "_pool",
        "_savepoint",
        "_tx_state",
    )

    def __init__(
        self,
        pool_or_conn: Any,
        *,
        dialect: Dialect | None = None,
        _depth: int = 0,
        _savepoint: str | None = None,
        _tx_state: _TxState | None = None,
    ) -> None:
        self._dialect = dialect or MySQLDialect()
        if not isinstance(self._dialect, (MySQLDialect, TiDBDialect)):
            msg = "AsyncmyAdapter requires MySQLDialect or TiDBDialect"
            raise TypeError(msg)
        self._depth = _depth
        self._checkout_owners = new_checkout_owners()
        self._savepoint = _savepoint
        if hasattr(pool_or_conn, "acquire") and not hasattr(pool_or_conn, "cursor"):
            self._pool = pool_or_conn
            self._conn: Any | None = None
            self._tx_state = _TxState(0)
        else:
            self._pool = None
            self._conn = pool_or_conn
            self._tx_state = _tx_state if _tx_state is not None else _TxState(_depth)

    @property
    def dialect(self) -> Dialect:
        return self._dialect

    async def _acquire(self) -> Any:
        ensure_not_checked_out(self._checkout_owners)
        if self._conn is not None:
            if self._tx_state.active_depth != self._depth:
                msg = (
                    "connection is checked out for an active transaction; "
                    "use the adapter yielded by transaction()"
                )
                raise RuntimeError(msg)
            return self._conn
        assert self._pool is not None
        return await self._pool.acquire()

    async def _release(self, conn: Any) -> None:
        if self._pool is not None and self._conn is None:
            await self._pool.release(conn)

    @with_translated_errors
    async def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Sequence[Any]]:
        encoded = self._dialect.encode_params(params)
        conn = await self._acquire()
        try:
            async with conn.cursor() as cursor:
                await cursor.execute(sql, encoded)
                rows = await cursor.fetchall()
                return cast(Sequence[Sequence[Any]], rows)
        finally:
            await self._release(conn)

    @with_translated_errors
    async def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Any] | None:
        encoded = self._dialect.encode_params(params)
        conn = await self._acquire()
        try:
            async with conn.cursor() as cursor:
                await cursor.execute(sql, encoded)
                return cast(Sequence[Any] | None, await cursor.fetchone())
        finally:
            await self._release(conn)

    @with_translated_errors
    async def fetch_val(self, sql: str, params: Sequence[Any] = ()) -> Any:
        row = await self.fetch_one(sql, params)
        if row is None:
            return None
        return row[0]

    @staticmethod
    async def _close_broken_connection(conn: Any) -> None:
        try:
            if hasattr(conn, "ensure_closed"):
                res = conn.ensure_closed()
                if inspect.isawaitable(res):
                    await res
            if hasattr(conn, "close"):
                res = conn.close()
                if inspect.isawaitable(res):
                    await res
        except BaseException:
            pass

    async def _commit_if_autocommit(self, conn: Any) -> None:
        """Commit writes outside an active transaction.

        Like ``AiosqliteAdapter``, standalone writes (depth 0) commit immediately
        to ensure persistence matches asyncpg-style autocommit when autocommit is
        disabled on the connection (default in asyncmy), preventing silent rollback
        and pool connection drop on release (#58).
        """
        if self._depth == 0 and self._tx_state.active_depth == 0:
            get_autocommit = getattr(conn, "get_autocommit", None)
            if callable(get_autocommit):
                res = get_autocommit()
                if inspect.isawaitable(res):
                    res = await res
                if not res:
                    await conn.commit()
            elif hasattr(conn, "autocommit") and not conn.autocommit:
                await conn.commit()

    @with_translated_errors
    async def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        encoded = self._dialect.encode_params(params)
        conn = await self._acquire()
        try:
            async with conn.cursor() as cursor:
                result = int(await cursor.execute(sql, encoded))
            await self._commit_if_autocommit(conn)
            return result
        finally:
            await self._release(conn)

    @with_translated_errors
    async def execute_many(self, sql: str, params_seq: Sequence[Sequence[Any]]) -> None:
        encoded_seq = self._dialect.encode_params_seq(params_seq)
        conn = await self._acquire()
        try:
            async with conn.cursor() as cursor:
                await cursor.executemany(sql, encoded_seq)
            await self._commit_if_autocommit(conn)
        finally:
            await self._release(conn)

    async def _apply_tx_options(
        self,
        conn: Any,
        *,
        isolation: IsolationLevel | None,
        readonly: bool,
        deferrable: bool,
    ) -> None:
        for statement in begin_sql(
            self._dialect.name,
            isolation=isolation,
            readonly=readonly,
            deferrable=deferrable,
        ):
            async with conn.cursor() as cursor:
                await cursor.execute(statement)

    @asynccontextmanager
    async def transaction(
        self,
        *,
        isolation: IsolationLevel | None = None,
        readonly: bool = False,
        deferrable: bool = False,
    ) -> AsyncGenerator[AsyncmyAdapter]:
        # Match fetch_*/execute*: the outer pool adapter must not be reused
        # while this task already holds a checkout (use the yielded adapter).
        ensure_not_checked_out(self._checkout_owners)
        if self._conn is not None and self._tx_state.active_depth != self._depth:
            msg = (
                "transaction already active on this adapter; "
                "use the adapter yielded by transaction() for nesting"
            )
            raise RuntimeError(msg)

        if self._pool is not None and self._conn is None:
            # Per-task checkout: peers may still acquire from the same pool.
            with checkout_scope(self._checkout_owners):
                active: Any = await self._pool.acquire()
                tx_state = _TxState(1)
                tx_failed = False
                try:
                    try:
                        # MySQL/TiDB apply SET TRANSACTION to the *next* transaction.
                        await self._apply_tx_options(
                            active,
                            isolation=isolation,
                            readonly=readonly,
                            deferrable=deferrable,
                        )
                        await active.begin()
                    except BaseException as exc:
                        tx_failed = True
                        try:
                            await active.rollback()
                        except BaseException:
                            tx_state.active_depth = -1
                            await self._close_broken_connection(active)
                        if isinstance(exc, Exception):
                            reraise_translated(exc)
                        raise exc

                    pinned = AsyncmyAdapter(
                        active,
                        dialect=self._dialect,
                        _depth=1,
                        _tx_state=tx_state,
                    )
                    try:
                        yield pinned
                    except BaseException as exc:
                        tx_failed = True
                        try:
                            await active.rollback()
                        except BaseException:
                            tx_state.active_depth = -1
                            await self._close_broken_connection(active)
                        else:
                            tx_state.active_depth = 0
                        if isinstance(exc, Exception):
                            reraise_translated(exc)
                        raise exc
                    else:
                        try:
                            await active.commit()
                        except BaseException as exc:
                            tx_failed = True
                            try:
                                await active.rollback()
                            except BaseException:
                                tx_state.active_depth = -1
                                await self._close_broken_connection(active)
                            else:
                                tx_state.active_depth = 0
                            if isinstance(exc, Exception):
                                reraise_translated(exc)
                            raise
                        else:
                            tx_state.active_depth = 0
                finally:
                    try:
                        await self._pool.release(active)
                    except BaseException:
                        if not tx_failed:
                            raise
            return

        assert self._conn is not None
        if self._depth == 0:
            with checkout_scope(self._checkout_owners):
                try:
                    await self._apply_tx_options(
                        self._conn,
                        isolation=isolation,
                        readonly=readonly,
                        deferrable=deferrable,
                    )
                    await self._conn.begin()
                except BaseException as exc:
                    try:
                        await self._conn.rollback()
                    except BaseException:
                        self._tx_state.active_depth = -1
                        await self._close_broken_connection(self._conn)
                    if isinstance(exc, Exception):
                        reraise_translated(exc)
                    raise exc

                self._tx_state.active_depth = 1
                pinned = AsyncmyAdapter(
                    self._conn,
                    dialect=self._dialect,
                    _depth=1,
                    _tx_state=self._tx_state,
                )
                try:
                    yield pinned
                except BaseException as exc:
                    try:
                        await self._conn.rollback()
                    except BaseException:
                        self._tx_state.active_depth = -1
                        await self._close_broken_connection(self._conn)
                    else:
                        self._tx_state.active_depth = 0
                    if isinstance(exc, Exception):
                        reraise_translated(exc)
                    raise exc
                else:
                    try:
                        await self._conn.commit()
                    except BaseException as exc:
                        try:
                            await self._conn.rollback()
                        except BaseException:
                            self._tx_state.active_depth = -1
                            await self._close_broken_connection(self._conn)
                        else:
                            self._tx_state.active_depth = 0
                        if isinstance(exc, Exception):
                            reraise_translated(exc)
                        raise exc
                    else:
                        self._tx_state.active_depth = 0
            return

        savepoint = f"atla_sp_{self._depth}"
        with checkout_scope(self._checkout_owners):
            try:
                async with self._conn.cursor() as cursor:
                    await cursor.execute(f"SAVEPOINT {savepoint}")
            except Exception as exc:
                reraise_translated(exc)
            self._tx_state.active_depth = self._depth + 1
            pinned = AsyncmyAdapter(
                self._conn,
                dialect=self._dialect,
                _depth=self._depth + 1,
                _savepoint=savepoint,
                _tx_state=self._tx_state,
            )
            try:
                yield pinned
            except BaseException as exc:
                try:
                    async with self._conn.cursor() as cursor:
                        await cursor.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                        await cursor.execute(f"RELEASE SAVEPOINT {savepoint}")
                    self._tx_state.active_depth = self._depth
                except BaseException:
                    self._tx_state.active_depth = -1
                    await self._close_broken_connection(self._conn)
                if isinstance(exc, Exception):
                    reraise_translated(exc)
                raise exc
            else:
                try:
                    async with self._conn.cursor() as cursor:
                        await cursor.execute(f"RELEASE SAVEPOINT {savepoint}")
                    self._tx_state.active_depth = self._depth
                except Exception as exc:
                    self._tx_state.active_depth = -1
                    await self._close_broken_connection(self._conn)
                    reraise_translated(exc)
