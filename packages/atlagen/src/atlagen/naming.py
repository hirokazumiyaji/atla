"""Name conversion helpers for generated Python identifiers."""

from __future__ import annotations

import keyword
import re

_SPLIT_RE = re.compile(r"[_\s-]+")


def singularize(name: str) -> str:
    """Best-effort English singularization for table names."""
    if not name:
        return name
    lower = name.lower()
    if lower.endswith("ies") and len(lower) > 3:
        return name[:-3] + ("Y" if name[-3].isupper() else "y")
    if lower.endswith(("sses", "xes", "zes", "ches", "shes")):
        return name[:-2]
    if lower.endswith("s") and not lower.endswith("ss"):
        return name[:-1]
    return name


def to_pascal_case(name: str) -> str:
    """Convert snake/kebab/space-separated name to PascalCase."""
    parts = [part for part in _SPLIT_RE.split(name) if part]
    if not parts:
        return name
    return "".join(part[:1].upper() + part[1:] for part in parts)


def table_model_name(table_name: str) -> str:
    """``users`` -> ``User``."""
    return to_pascal_case(singularize(table_name))


def table_accessor_name(table_name: str) -> str:
    """``users`` -> ``user``."""
    singular = singularize(table_name)
    if not singular:
        return singular
    return singular[:1].lower() + singular[1:]


# Builtins / soft-keyword-adjacent names that should not become attributes as-is.
# Note: ``id`` is intentionally allowed (common primary-key column).
_RESERVED_ATTRS = frozenset(
    {
        "type",
        "list",
        "dict",
        "set",
        "tuple",
        "str",
        "int",
        "float",
        "bool",
        "bytes",
        "object",
        "property",
        "super",
        "self",
        "cls",
    }
)


def safe_attr_name(column_name: str) -> str:
    """Return a Python-safe attribute name (append ``_`` for keywords)."""
    if (
        keyword.iskeyword(column_name)
        or keyword.issoftkeyword(column_name)
        or column_name in _RESERVED_ATTRS
    ):
        return f"{column_name}_"
    return column_name


def enum_class_name(enum_name: str) -> str:
    """``user_role`` -> ``UserRole``."""
    return to_pascal_case(enum_name)


def mysql_inline_enum_class_name(model_name: str, column_name: str) -> str:
    """``User`` + ``status`` -> ``UserStatusEnum``."""
    return f"{model_name}{to_pascal_case(column_name)}Enum"


def mysql_inline_enum_name(table_name: str, column_name: str) -> str:
    """Synthetic DB enum name for MySQL inline enums."""
    return f"{table_name}_{column_name}"
