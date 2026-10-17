"""Emit AtlaClient facade."""

from __future__ import annotations

from atlagen.ir import SchemaIR


def emit_client(schema: SchemaIR) -> str:
    lines = [
        "from __future__ import annotations",
        "",
        "from collections.abc import AsyncGenerator",
        "from contextlib import asynccontextmanager",
        "",
        "from atla.adapter.base import DBAdapter",
    ]
    if schema.tables:
        repo_imports = ", ".join(f"{table.model_name}Repository" for table in schema.tables)
        lines.append(f"from .query import {repo_imports}")
    lines.extend(
        [
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
            "    async def transaction(self) -> AsyncGenerator[AtlaClient, None]:",
            "        async with self._adapter.transaction() as tx:",
            "            yield AtlaClient(tx)",
            "",
        ]
    )
    return "\n".join(lines)
