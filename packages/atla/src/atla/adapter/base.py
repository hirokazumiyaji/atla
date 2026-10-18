"""DBAdapter protocol shared by concrete driver adapters."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any, Literal, Protocol, runtime_checkable

from atla.dialect import Dialect

IsolationLevel = Literal[
    "read_uncommitted",
    "read_committed",
    "repeatable_read",
    "serializable",
]


class DBAdapter(Protocol):
    """Async database execution surface used by generated clients."""

    @property
    def dialect(self) -> Dialect: ...

    async def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Sequence[Any]]:
        """Fetch multiple rows as a sequence of tuples/records."""
        ...

    async def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Any] | None:
        """Fetch a single row, or None if no row matches."""
        ...

    async def fetch_val(self, sql: str, params: Sequence[Any] = ()) -> Any:
        """Fetch the first column of the first row."""
        ...

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        """Execute DML and return affected row count."""
        ...

    async def execute_many(self, sql: str, params_seq: Sequence[Sequence[Any]]) -> None:
        """Execute a batch DML statement."""
        ...

    def transaction(
        self,
        *,
        isolation: IsolationLevel | None = None,
        readonly: bool = False,
        deferrable: bool = False,
    ) -> AbstractAsyncContextManager[DBAdapter]:
        """Begin a transaction and yield a connection-pinned adapter."""
        ...


@runtime_checkable
class CopyAdapter(Protocol):
    """Optional adapter capability: bulk binary ``COPY`` insert.

    Implemented by :class:`~atla.adapter.asyncpg.AsyncpgAdapter` (PostgreSQL /
    CockroachDB) and forwarded by the routing / observability wrappers. Query
    builders detect it with ``isinstance(adapter, CopyAdapter)`` before choosing
    the ``COPY`` bulk path, so adapters without native support keep working.
    """

    async def copy_records_to_table(
        self,
        table: str,
        *,
        columns: Sequence[str],
        records: Sequence[Sequence[Any]],
    ) -> int:
        """Copy *records* into *table* and return the number of rows copied.

        ``table`` is the raw (unquoted) name, optionally schema-qualified as
        ``schema.table``. Column values are dialect-encoded before reaching the
        driver.
        """
        ...
