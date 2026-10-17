"""Emit AtlaClient facade."""

from __future__ import annotations

from atlagen.ir import SchemaIR


def emit_client(schema: SchemaIR) -> str:
    lines = [
        "from __future__ import annotations",
        "",
        "from collections.abc import AsyncGenerator, Awaitable, Callable",
        "from contextlib import asynccontextmanager",
        "from typing import TypeVar",
        "",
        "from atla.adapter.base import DBAdapter, IsolationLevel",
        "from atla.transaction import run_in_transaction as _run_in_transaction",
    ]
    if schema.tables:
        repo_imports = ", ".join(f"{table.model_name}Repository" for table in schema.tables)
        lines.append(f"from .query import {repo_imports}")
    lines.extend(
        [
            "",
            'T = TypeVar("T")',
            "",
            "class AtlaClient:",
            '    __slots__ = ("_adapter",)',
            "",
            "    def __init__(self, adapter: DBAdapter) -> None:",
            "        self._adapter = adapter",
            "",
        ]
    )
    for table in schema.tables:
        lines.extend(
            [
                "    @property",
                f"    def {table.accessor_name}(self) -> {table.model_name}Repository:",
                f"        return {table.model_name}Repository(self._adapter)",
                "",
            ]
        )
    lines.extend(
        [
            "    @asynccontextmanager",
            "    async def transaction(",
            "        self,",
            "        *,",
            "        isolation: IsolationLevel | None = None,",
            "        readonly: bool = False,",
            "        deferrable: bool = False,",
            "    ) -> AsyncGenerator[AtlaClient, None]:",
            "        async with self._adapter.transaction(",
            "            isolation=isolation,",
            "            readonly=readonly,",
            "            deferrable=deferrable,",
            "        ) as tx:",
            "            yield AtlaClient(tx)",
            "",
            "    async def run_in_transaction(",
            "        self,",
            "        fn: Callable[[AtlaClient], Awaitable[T]],",
            "        *,",
            "        max_retries: int = 3,",
            "        base_backoff: float = 0.01,",
            "        isolation: IsolationLevel | None = None,",
            "        readonly: bool = False,",
            "        deferrable: bool = False,",
            "    ) -> T:",
            "        async def _wrapped(adapter: DBAdapter) -> T:",
            "            return await fn(AtlaClient(adapter))",
            "",
            "        return await _run_in_transaction(",
            "            self._adapter,",
            "            _wrapped,",
            "            max_retries=max_retries,",
            "            base_backoff=base_backoff,",
            "            isolation=isolation,",
            "            readonly=readonly,",
            "            deferrable=deferrable,",
            "        )",
            "",
        ]
    )
    return "\n".join(lines)
