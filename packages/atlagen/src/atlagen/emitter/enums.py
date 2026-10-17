"""Emit StrEnum modules from SchemaIR."""

from __future__ import annotations

from atlagen.ir import SchemaIR


def emit_enums(schema: SchemaIR) -> str:
    lines = [
        "from __future__ import annotations",
        "",
    ]
    if not schema.enums:
        lines.append("# No enums defined in schema.")
        lines.append("")
        return "\n".join(lines)

    lines.extend(["from enum import StrEnum", ""])
    for enum in schema.enums:
        lines.append(f"class {enum.class_name}(StrEnum):")
        used: set[str] = set()
        for value in enum.values:
            member = _unique_member(value, used)
            used.add(member)
            lines.append(f"    {member} = {value!r}")
        lines.append("")
    return "\n".join(lines)


def _unique_member(value: str, used: set[str]) -> str:
    base = _safe_member(value)
    if base not in used:
        return base
    index = 2
    while f"{base}_{index}" in used:
        index += 1
    return f"{base}_{index}"


def _safe_member(value: str) -> str:
    if value.isidentifier() and value.isupper():
        return value
    if value.isidentifier():
        return value.upper()
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in value)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"V_{cleaned}"
    return cleaned.upper()
