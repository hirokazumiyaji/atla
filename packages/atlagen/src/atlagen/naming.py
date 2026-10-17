"""Name conversion helpers for generated Python identifiers."""

from __future__ import annotations

import keyword
import re

_SPLIT_RE = re.compile(r"[_\s-]+")
_CAMEL_1 = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_2 = re.compile(r"([a-z0-9])([A-Z])")

# Words that look plural but must not be singularized by the simple suffix rules.
# Includes singulars of irregular plurals that themselves end in ``s`` (so the
# trailing-``s`` heuristic does not turn ``thesis`` into ``thesi``, ``bus`` into
# ``bu``, etc.).
_UNCHANGING = frozenset(
    {
        "series",
        "news",
        "data",
        "media",
        "species",
        "meta",
        "info",
        "sms",
        "ops",
        "status",
        "analysis",
        "basis",
        "crisis",
        "thesis",
        "diagnosis",
        "oasis",
        "bus",
        "gas",
        "alias",
        "atlas",
        "virus",
        "bonus",
        "campus",
        "class",
    }
)

_IRREGULAR_SINGULAR: dict[str, str] = {
    "people": "person",
    "children": "child",
    "men": "man",
    "women": "woman",
    "teeth": "tooth",
    "feet": "foot",
    "mice": "mouse",
    "geese": "goose",
    "criteria": "criterion",
    "indices": "index",
    "matrices": "matrix",
    "vertices": "vertex",
    "leaves": "leaf",
    "lives": "life",
    "knives": "knife",
    "wolves": "wolf",
    "shelves": "shelf",
    "halves": "half",
    "loaves": "loaf",
    "quizzes": "quiz",
    "statuses": "status",
    # -ies that should become -ie (not -y)
    "movies": "movie",
    "cookies": "cookie",
    "zombies": "zombie",
    "smoothies": "smoothie",
    "calories": "calorie",
    "ties": "tie",
    "pies": "pie",
    "lies": "lie",
    "dies": "die",
    # -ses / -uses / -ases that need careful handling
    "buses": "bus",
    "gases": "gas",
    "aliases": "alias",
    "campuses": "campus",
    "viruses": "virus",
    "bonuses": "bonus",
    "atlases": "atlas",
    "classes": "class",
    "analyses": "analysis",
    "crises": "crisis",
    "theses": "thesis",
    "hypotheses": "hypothesis",
    "diagnoses": "diagnosis",
    "oases": "oasis",
}

# AtlaClient members that must not be overwritten by table accessors only.
# Intentionally *not* folded into column reserved attrs — a column named
# ``transaction`` should stay ``transaction`` on Record/Table.
_CLIENT_RESERVED = frozenset(
    {
        "transaction",
        "_adapter",
        "adapter",
    }
)

# Builtins / soft-keyword-adjacent names that should not become attributes as-is.
# Note: ``id`` is intentionally allowed (common primary-key column).
# Also reserve names emitted on generated Record / Table classes.
_RESERVED_COLUMN_ATTRS = frozenset(
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
        "__table_name__",
        "sql_alias",
        "join_target",
        "select_list",
        "column_count",
        "get_row_mapper",
        "map_row",
        "map_row_optional",
    }
)

# Back-compat alias used by callers that mean "column/table-member attrs".
_RESERVED_ATTRS = _RESERVED_COLUMN_ATTRS


def unqualified_table_name(table_name: str) -> str:
    """Strip schema qualification: ``other.users`` -> ``users``."""
    return table_name.rsplit(".", 1)[-1]


def to_snake_case(name: str) -> str:
    """Convert CamelCase / kebab / spaced names to snake_case."""
    text = name.replace("-", "_").replace(" ", "_")
    text = _CAMEL_1.sub(r"\1_\2", text)
    text = _CAMEL_2.sub(r"\1_\2", text)
    parts = [part for part in _SPLIT_RE.split(text) if part]
    return "_".join(part.lower() for part in parts)


def singularize(name: str) -> str:
    """Best-effort English singularization for table / token names."""
    if not name:
        return name
    lower = name.lower()
    if lower in _IRREGULAR_SINGULAR:
        singular = _IRREGULAR_SINGULAR[lower]
        return _match_case(name, singular)
    if lower in _UNCHANGING:
        return name
    if lower.endswith("ies") and len(lower) > 3:
        # babies -> baby; movies handled via irregular map
        return _match_case(name, lower[:-3] + "y")
    if lower.endswith(("sses", "xes", "zes", "ches", "shes")):
        return name[:-2]
    # -uses/-ases/… and -ves plurals (buses, leaves, …) live in
    # _IRREGULAR_SINGULAR so already-singular names like ``houses`` / ``archives``
    # are not mangled by a broad heuristic.
    if lower.endswith("s") and not lower.endswith("ss"):
        return name[:-1]
    return name


def singularize_identifier(name: str) -> str:
    """Singularize the last snake_case segment (``user_accounts`` -> ``user_account``)."""
    snake = to_snake_case(name)
    if "_" not in snake:
        return singularize(snake)
    head, _, tail = snake.rpartition("_")
    return f"{head}_{singularize(tail)}"


def _match_case(original: str, singular: str) -> str:
    if original.isupper():
        return singular.upper()
    if original[:1].isupper():
        return singular[:1].upper() + singular[1:]
    return singular


def to_pascal_case(name: str) -> str:
    """Convert snake/kebab/space-separated name to PascalCase."""
    snake = to_snake_case(name)
    parts = [part for part in snake.split("_") if part]
    if not parts:
        return name
    return "".join(part[:1].upper() + part[1:] for part in parts)


def table_model_name(table_name: str) -> str:
    """``users`` / ``other.users`` / ``UserAccounts`` -> ``User`` / ``UserAccount``."""
    return to_pascal_case(singularize_identifier(unqualified_table_name(table_name)))


def table_accessor_name(table_name: str) -> str:
    """``users`` / ``transactions`` / ``classes`` -> ``user`` / ``transaction_`` / ``class_``."""
    singular = singularize_identifier(unqualified_table_name(table_name))
    if not singular:
        return singular
    return safe_accessor_name(singular)


def _escape_ident(name: str, *, reserved: frozenset[str]) -> str:
    """Normalize *name* to a safe identifier, appending ``_`` while reserved."""
    text = to_snake_case(name) if not name.isidentifier() else name
    if not text.isidentifier():
        cleaned = "".join(ch if ch.isalnum() else "_" for ch in text)
        if not cleaned or cleaned[0].isdigit():
            cleaned = f"col_{cleaned}"
        text = cleaned
    # Prefer snake_case for CamelCase identifiers used as attrs.
    if any(ch.isupper() for ch in text[1:]):
        text = to_snake_case(text)
    while keyword.iskeyword(text) or keyword.issoftkeyword(text) or text in reserved:
        text = f"{text}_"
    return text


def safe_attr_name(column_name: str) -> str:
    """Return a Python-safe column/Table attribute name (keywords / Record members)."""
    return _escape_ident(column_name, reserved=_RESERVED_COLUMN_ATTRS)


def safe_accessor_name(name: str) -> str:
    """Return a Python-safe AtlaClient accessor name (includes client members)."""
    return _escape_ident(name, reserved=_RESERVED_COLUMN_ATTRS | _CLIENT_RESERVED)


def unique_attr_names(column_names: list[str]) -> list[str]:
    """Assign unique ``safe_attr_name`` values for a table's columns."""
    used: set[str] = set()
    result: list[str] = []
    for column_name in column_names:
        base = safe_attr_name(column_name)
        candidate = base
        index = 2
        while candidate in used or candidate in _RESERVED_COLUMN_ATTRS:
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
