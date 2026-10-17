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
    slot_parts = ['"_adapter"']
    slot_parts.extend(repr(table.accessor_name) for table in schema.tables)
    slots = ", ".join(slot_parts)
    lines.extend(
        [
            "",
            "class AtlaClient:",
            # Trailing comma keeps a single-item slots definition a tuple.
            f"    __slots__ = ({slots},)",
            "",
            "    def __init__(self, adapter: DBAdapter) -> None:",
            "        self._adapter = adapter",
        ]
    )
    for table in schema.tables:
        lines.append(f"        self.{table.accessor_name} = {table.model_name}Repository(adapter)")
    lines.append("")
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
