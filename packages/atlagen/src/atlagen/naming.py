"""Name conversion helpers for generated Python identifiers."""

from __future__ import annotations

import keyword
import re

_SPLIT_RE = re.compile(r"[_\s-]+")

# Words that look plural but must not be singularized by the simple suffix rules.
_SINGULAR_EXCEPTIONS = frozenset(
    {
        "status",
        "statuses",
        "series",
        "news",
        "data",
        "media",
        "species",
        "analysis",
        "basis",
        "crisis",
        "meta",
        "info",
        "sms",
        "ops",
    }
)


def unqualified_table_name(table_name: str) -> str:
    """Strip schema qualification: ``other.users`` -> ``users``."""
    return table_name.rsplit(".", 1)[-1]


def singularize(name: str) -> str:
    """Best-effort English singularization for table names.

    This is intentionally simple (suffix heuristics + a small exception list).
    Irregular plurals and domain-specific names may need renaming at the DB
    layer; known traps include ``status``/``series``/``news``/``data``.
    """
    if not name:
        return name
    lower = name.lower()
    if lower in _SINGULAR_EXCEPTIONS:
        # Keep mass/invariant nouns; map statuses -> status.
        if lower == "statuses":
            return name[:-2]
        return name
    if lower.endswith("ies") and len(lower) > 3:
        return name[:-3] + ("Y" if name[-3].isupper() else "y")
    if lower.endswith(("sses", "xes", "zes", "ches", "shes")):
        return name[:-2]
    if lower.endswith("ses") and len(lower) > 4:
        # statuses already handled; avoid ``analyses`` -> ``analyse`` style cuts.
        return name[:-1]
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
    """``users`` / ``other.users`` -> ``User``."""
    return to_pascal_case(singularize(unqualified_table_name(table_name)))


def table_accessor_name(table_name: str) -> str:
    """``users`` / ``other.users`` -> ``user`` (escaped if reserved on AtlaClient)."""
    singular = singularize(unqualified_table_name(table_name))
    if not singular:
        return singular
    name = singular[:1].lower() + singular[1:]
    return safe_attr_name(name)


# Builtins / soft-keyword-adjacent names that should not become attributes as-is.
# Note: ``id`` is intentionally allowed (common primary-key column).
# Also reserve names emitted on generated Record / Table / Client classes.
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
        # Generated Table / Record members
        "as_",
        "_alias",
        "_from_row",
        "_COLUMNS",
        "_ROW_MAPPER",
        "_PK_INDEXES",
        "__table_name__",
        "sql_alias",
        "join_target",
        "select_list",
        "column_count",
        "get_row_mapper",
        "map_row",
        "map_row_optional",
        # Generated AtlaClient members (must not collide with table accessors / slots)
        "_adapter",
        "transaction",
        "run_in_transaction",
    }
)


def safe_attr_name(column_name: str) -> str:
    """Return a Python-safe attribute name (append ``_`` for keywords / reserved)."""
    name = column_name
    if not name.isidentifier():
        cleaned = "".join(ch if ch.isalnum() else "_" for ch in name)
        if not cleaned or cleaned[0].isdigit():
            cleaned = f"col_{cleaned}"
        name = cleaned
    while keyword.iskeyword(name) or keyword.issoftkeyword(name) or name in _RESERVED_ATTRS:
        name = f"{name}_"
    return name


def unique_attr_names(column_names: list[str]) -> list[str]:
    """Assign unique ``safe_attr_name`` values for a table's columns."""
    used: set[str] = set()
    result: list[str] = []
    for column_name in column_names:
        base = safe_attr_name(column_name)
        candidate = base
        index = 2
        while candidate in used or candidate in _RESERVED_ATTRS:
            candidate = f"{base}_{index}"
            index += 1
        used.add(candidate)
        result.append(candidate)
    return result


def enum_class_name(enum_name: str) -> str:
    """``user_role`` -> ``UserRole``."""
    return to_pascal_case(unqualified_table_name(enum_name))


def mysql_inline_enum_class_name(model_name: str, column_name: str) -> str:
    """``User`` + ``status`` -> ``UserStatusEnum``."""
    return f"{model_name}{to_pascal_case(column_name)}Enum"


def mysql_inline_enum_name(table_name: str, column_name: str) -> str:
    """Synthetic DB enum name for MySQL inline enums."""
    return f"{unqualified_table_name(table_name)}_{column_name}"
