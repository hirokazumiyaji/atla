"""Emit StrEnum modules from SchemaIR."""

from __future__ import annotations

from atlagen.ir import SchemaIR


def emit_enums(schema: SchemaIR) -> str:
    lines = [
        "from __future__ import annotations",
        "",
        "from enum import StrEnum",
        "",
    ]
    if not schema.enums:
        lines.append("# No enums defined in schema.")
        lines.append("")
        return "\n".join(lines)

    for enum in schema.enums:
        lines.append(f"class {enum.class_name}(StrEnum):")
        for value in enum.values:
            member = value.upper() if value.isidentifier() else _safe_member(value)
            lines.append(f"    {member} = {value!r}")
        lines.append("")
    return "\n".join(lines)


def _safe_member(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in value)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"V_{cleaned}"
    return cleaned.upper()
