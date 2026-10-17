"""Fast SQL operation / collection extraction with LRU caching."""

from __future__ import annotations

import re
from functools import lru_cache

_LEADING_WS = re.compile(r"^\s+")
_IDENT = re.compile(r'(?:"([^"]+)"|`([^`]+)`|\[([^\]]+)\]|([A-Za-z_][\w$]*))')
_FROM = re.compile(
    r"\bFROM\s+(?P<table>(?:\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][\w$]*)"
    r"(?:\s*\.\s*(?:\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][\w$]*))?)",
    re.IGNORECASE,
)
_INTO = re.compile(
    r"\bINTO\s+(?P<table>(?:\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][\w$]*)"
    r"(?:\s*\.\s*(?:\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][\w$]*))?)",
    re.IGNORECASE,
)
_UPDATE = re.compile(
    r"\bUPDATE\s+(?P<table>(?:\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][\w$]*)"
    r"(?:\s*\.\s*(?:\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][\w$]*))?)",
    re.IGNORECASE,
)
_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def _strip_ident(raw: str) -> str:
    raw = raw.strip()
    if "." in raw:
        raw = raw.rsplit(".", 1)[-1].strip()
    match = _IDENT.fullmatch(raw)
    if match is None:
        return raw.strip("\"'`[]")
    return next(g for g in match.groups() if g is not None)


@lru_cache(maxsize=2048)
def extract_sql_meta(sql: str) -> tuple[str, str | None]:
    """Return ``(operation_name, collection_name)`` for a SQL statement."""
    cleaned = _COMMENT.sub(" ", sql)
    cleaned = _LEADING_WS.sub("", cleaned)
    upper = cleaned[:16].upper()
    if upper.startswith("WITH"):
        # CTEs embed their own FROM clauses; avoid attributing the wrong table.
        return "SELECT", None
    if upper.startswith("SELECT"):
        match = _FROM.search(cleaned)
        return "SELECT", _strip_ident(match.group("table")) if match else None
    if upper.startswith("INSERT"):
        match = _INTO.search(cleaned)
        return "INSERT", _strip_ident(match.group("table")) if match else None
    if upper.startswith("UPDATE"):
        match = _UPDATE.search(cleaned)
        return "UPDATE", _strip_ident(match.group("table")) if match else None
    if upper.startswith("DELETE"):
        match = _FROM.search(cleaned)
        return "DELETE", _strip_ident(match.group("table")) if match else None
    if upper.startswith("BEGIN"):
        return "BEGIN", None
    if upper.startswith("COMMIT"):
        return "COMMIT", None
    if upper.startswith("ROLLBACK"):
        return "ROLLBACK", None
    if upper.startswith("SAVEPOINT") or upper.startswith("RELEASE"):
        return "SAVEPOINT", None
    token = cleaned.split(None, 1)[0].upper() if cleaned else "UNKNOWN"
    return token, None
