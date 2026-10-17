"""DBAdapter protocol shared by concrete driver adapters."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any, Protocol

from atla.dialect import Dialect


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

    def transaction(self) -> AbstractAsyncContextManager[DBAdapter]:
        """Begin a transaction and yield a connection-pinned adapter."""
        ...
