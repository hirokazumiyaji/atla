"""Fast SQL operation / collection extraction with LRU caching."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atla.dialect import Dialect

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
_COPY = re.compile(
    r"\bCOPY\s+(?P<table>(?:\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][\w$]*)"
    r"(?:\s*\.\s*(?:\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][\w$]*))?)",
    re.IGNORECASE,
)
_DOLLAR_TAG = re.compile(r"\$(?:[A-Za-z_]\w*)?\$")


def _parse_single_quote(sql: str, start: int, mysql_mode: bool) -> int:
    """Find the closing quote index of the single-quoted string literal starting at `start`."""
    n = len(sql)
    # PostgreSQL E'...' escape strings always use backslash escaping
    is_pg_escape = (
        start > 0
        and sql[start - 1] in ("E", "e")
        and (start == 1 or not (sql[start - 2].isalnum() or sql[start - 2] in ("_", "$")))
    )
    if is_pg_escape or mysql_mode:
        i = start + 1
        while i < n:
            if sql[i] == "\\":
                i += 2
                continue
            if sql[i] == "'":
                if i + 1 < n and sql[i + 1] == "'":
                    i += 2
                    continue
                return i
            i += 1
        return -1

    # Standard SQL (PostgreSQL standard strings, SQLite): only '' escapes
    i = start + 1
    while i < n:
        if sql[i] == "'":
            if i + 1 < n and sql[i + 1] == "'":
                i += 2
                continue
            return i
        i += 1
    return -1


def _parse_double_quote(sql: str, start: int, mysql_mode: bool) -> int:
    """Find closing quote index of double-quoted literal or identifier starting at `start`."""
    n = len(sql)
    i = start + 1
    while i < n:
        if mysql_mode and sql[i] == "\\":
            i += 2
            continue
        if sql[i] == '"':
            if i + 1 < n and sql[i + 1] == '"':
                i += 2
                continue
            return i
        i += 1
    return -1


def _parse_bracket_ident(sql: str, start: int) -> int:
    """If [...] is a bracket-quoted identifier (SQLite/T-SQL), return closing index, else -1."""
    n = len(sql)
    i = start + 1
    while i < n:
        c = sql[i]
        if c in ("'", '"', "`", "["):
            # Nested quotes or brackets indicate array constructors or expressions
            # (e.g. ARRAY['x]--', 'y']), not a bracket-quoted identifier.
            return -1
        if c == "]":
            if i + 1 < n and sql[i + 1] == "]":
                i += 2
                continue
            return i
        i += 1
    return -1


def _strip_comments(sql: str, mysql_mode: bool) -> str:
    """Strip SQL block (/* */) and line (--) comments while preserving literals."""
    out: list[str] = []
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]
        if ch == "/" and i + 1 < n and sql[i + 1] == "*":
            end = sql.find("*/", i + 2)
            if end == -1:
                out.append(" ")
                break
            out.append(" ")
            i = end + 2
            continue
        if ch == "-" and i + 1 < n and sql[i + 1] == "-":
            # In MySQL, -- starts a line comment only when followed by whitespace,
            # a control character, or end-of-input (otherwise e.g. 1--1 is subtraction).
            if i + 2 == n or sql[i + 2] in (" ", "\t", "\r", "\n"):
                end = i + 2
                while end < n and sql[end] not in ("\r", "\n"):
                    end += 1
                out.append(" ")
                i = end
                continue
            elif not mysql_mode:
                # In standard SQLite / PostgreSQL, -- starts a comment to the end of the line.
                end = i + 2
                while end < n and sql[end] not in ("\r", "\n"):
                    end += 1
                out.append(" ")
                i = end
                continue
        if ch == "'":
            end = _parse_single_quote(sql, i, mysql_mode)
            if end != -1:
                out.append(sql[i : end + 1])
                i = end + 1
                continue
            out.append(ch)
            i += 1
            continue
        if ch == '"':
            end = _parse_double_quote(sql, i, mysql_mode)
            if end != -1:
                out.append(sql[i : end + 1])
                i = end + 1
                continue
            out.append(ch)
            i += 1
            continue
        if ch == "`":
            end = sql.find("`", i + 1)
            while end != -1 and end + 1 < n and sql[end + 1] == "`":
                end = sql.find("`", end + 2)
            if end != -1:
                out.append(sql[i : end + 1])
                i = end + 1
                continue
            out.append(ch)
            i += 1
            continue
        if ch == "[":
            end = _parse_bracket_ident(sql, i)
            if end != -1:
                out.append(sql[i : end + 1])
                i = end + 1
                continue
            out.append(ch)
            i += 1
            continue
        if ch == "$":
            if i == 0 or (not sql[i - 1].isalnum() and sql[i - 1] not in ("_", "$")):
                m = _DOLLAR_TAG.match(sql, i)
                if m:
                    tag = m.group(0)
                    end = sql.find(tag, i + len(tag))
                    if end != -1:
                        end_pos = end + len(tag)
                        out.append(sql[i:end_pos])
                        i = end_pos
                        continue
        out.append(ch)
        i += 1
    return "".join(out)


def _mask_parens(sql: str, mysql_mode: bool) -> str:
    """Replace content within balanced parentheses with spaces to avoid false-positive clauses."""
    out: list[str] = []
    depth = 0
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]
        if ch == "'":
            end = _parse_single_quote(sql, i, mysql_mode)
            if end != -1:
                length = end + 1 - i
                out.append(" " * length if depth > 0 else sql[i : end + 1])
                i = end + 1
                continue
        if ch == '"':
            end = _parse_double_quote(sql, i, mysql_mode)
            if end != -1:
                length = end + 1 - i
                out.append(" " * length if depth > 0 else sql[i : end + 1])
                i = end + 1
                continue
        if ch == "`":
            end = sql.find("`", i + 1)
            while end != -1 and end + 1 < n and sql[end + 1] == "`":
                end = sql.find("`", end + 2)
            if end != -1:
                length = end + 1 - i
                out.append(" " * length if depth > 0 else sql[i : end + 1])
                i = end + 1
                continue
        if ch == "[":
            end = _parse_bracket_ident(sql, i)
            if end != -1:
                length = end + 1 - i
                out.append(" " * length if depth > 0 else sql[i : end + 1])
                i = end + 1
                continue
            out.append(" " if depth > 0 else ch)
            i += 1
            continue
        if ch == "$":
            if i == 0 or (not sql[i - 1].isalnum() and sql[i - 1] not in ("_", "$")):
                m = _DOLLAR_TAG.match(sql, i)
                if m:
                    tag = m.group(0)
                    end = sql.find(tag, i + len(tag))
                    if end != -1:
                        end_pos = end + len(tag)
                        length = end_pos - i
                        out.append(" " * length if depth > 0 else sql[i:end_pos])
                        i = end_pos
                        continue
        if ch == "(":
            depth += 1
            out.append("(")
        elif ch == ")":
            if depth > 0:
                depth -= 1
            out.append(")")
        else:
            out.append(" " if depth > 0 else ch)
        i += 1
    return "".join(out)


def _strip_ident(raw: str) -> str:
    raw = raw.strip()
    if "." in raw:
        raw = raw.rsplit(".", 1)[-1].strip()
    match = _IDENT.fullmatch(raw)
    if match is None:
        return raw.strip("\"'`[]")
    return next(g for g in match.groups() if g is not None)


def _extract_with_mode(sql: str, mysql_mode: bool) -> tuple[str, str | None]:
    cleaned = _strip_comments(sql, mysql_mode)
    cleaned = _LEADING_WS.sub("", cleaned)
    upper = cleaned[:16].upper()
    if upper.startswith("WITH"):
        # CTEs embed their own FROM clauses; avoid attributing the wrong table.
        return "SELECT", None
    if upper.startswith("SELECT"):
        masked = _mask_parens(cleaned, mysql_mode)
        match = _FROM.search(masked)
        return "SELECT", _strip_ident(match.group("table")) if match else None
    if upper.startswith("INSERT"):
        masked = _mask_parens(cleaned, mysql_mode)
        match = _INTO.search(masked)
        return "INSERT", _strip_ident(match.group("table")) if match else None
    if upper.startswith("UPDATE"):
        masked = _mask_parens(cleaned, mysql_mode)
        match = _UPDATE.search(masked)
        return "UPDATE", _strip_ident(match.group("table")) if match else None
    if upper.startswith("DELETE"):
        masked = _mask_parens(cleaned, mysql_mode)
        match = _FROM.search(masked)
        return "DELETE", _strip_ident(match.group("table")) if match else None
    if upper.startswith("COPY"):
        match = _COPY.search(cleaned)
        return "COPY", _strip_ident(match.group("table")) if match else None
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


@lru_cache(maxsize=2048)
def _extract_sql_meta_cached(sql: str, dialect_name: str | None) -> tuple[str, str | None]:
    if dialect_name is not None:
        is_mysql = dialect_name.lower() in ("mysql", "mariadb", "tidb")
        return _extract_with_mode(sql, mysql_mode=is_mysql)

    # Dialect not explicitly provided:
    # If query contains MySQL backtick identifier quotes, prefer MySQL mode.
    if "`" in sql:
        return _extract_with_mode(sql, mysql_mode=True)
    # Try standard SQL (PostgreSQL, SQLite) first
    op, coll = _extract_with_mode(sql, mysql_mode=False)
    if coll is not None:
        return op, coll
    # Try MySQL escaping rules
    op_mysql, coll_mysql = _extract_with_mode(sql, mysql_mode=True)
    if coll_mysql is not None:
        return op_mysql, coll_mysql
    return op, coll


def extract_sql_meta(
    sql: str,
    dialect: Dialect | str | None = None,
) -> tuple[str, str | None]:
    """Return ``(operation_name, collection_name)`` for a SQL statement."""
    dialect_name: str | None = None
    if isinstance(dialect, str):
        dialect_name = dialect
    elif dialect is not None:
        raw_name = getattr(dialect, "name", None)
        if isinstance(raw_name, str):
            dialect_name = raw_name
    return _extract_sql_meta_cached(sql, dialect_name)
