"""Single-pass SQL compiler for Expression Tree AST nodes."""

from __future__ import annotations

import re
from typing import Any

from atla.dialect import Dialect
from atla.expressions import (
    BetweenOp,
    BinaryOp,
    Column,
    Expr,
    FuncCall,
    InOp,
    Literal,
    LogicalOp,
    OrderByClause,
    RawSQL,
    UnaryOp,
)


class SQLCompiler:
    """Compile expression AST nodes into SQL text and bind parameters."""

    __slots__ = ("buf", "dialect", "params")

    def __init__(self, dialect: Dialect, initial_sql: str = "") -> None:
        self.dialect = dialect
        self.buf: list[str] = [initial_sql] if initial_sql else []
        self.params: list[Any] = []

    def add_param(self, value: Any) -> str:
        self.params.append(value)
        return self.dialect.placeholder(len(self.params))

    def write(self, fragment: str) -> None:
        self.buf.append(fragment)

    def visit(self, node: Expr[Any]) -> None:
        if isinstance(node, Literal):
            self.write(self.add_param(node.value))
            return
        if isinstance(node, Column):
            self._visit_column(node)
            return
        if isinstance(node, BinaryOp):
            self._visit_binary(node)
            return
        if isinstance(node, UnaryOp):
            self._visit_unary(node)
            return
        if isinstance(node, LogicalOp):
            self._visit_logical(node)
            return
        if isinstance(node, InOp):
            self._visit_in(node)  # pyright: ignore[reportUnknownArgumentType]
            return
        if isinstance(node, BetweenOp):
            self._visit_between(node)  # pyright: ignore[reportUnknownArgumentType]
            return
        if isinstance(node, FuncCall):
            self._visit_func(node)
            return
        if isinstance(node, RawSQL):
            self._visit_raw(node)
            return
        msg = f"unsupported expression node: {type(node).__name__}"
        raise TypeError(msg)

    def visit_order_by(self, clause: OrderByClause) -> None:
        if clause.nulls is not None and self.dialect.name in {"mysql", "tidb"}:
            msg = (
                f"{self.dialect.name} does not support NULLS {clause.nulls} in ORDER BY; "
                "omit nulls= or use a dialect that supports NULLS FIRST/LAST"
            )
            raise ValueError(msg)
        self.visit(clause.expr)
        self.write(f" {clause.direction}")
        if clause.nulls is not None:
            self.write(f" NULLS {clause.nulls}")

    def build(self) -> tuple[str, list[Any]]:
        return "".join(self.buf), self.params

    def compile_fragment(self, node: Expr[Any]) -> str:
        """Visit *node* and return only the SQL appended for that visit."""
        start = len(self.buf)
        self.visit(node)
        fragment = "".join(self.buf[start:])
        del self.buf[start:]
        return fragment

    def _visit_column(self, node: Column[Any]) -> None:
        qualifier = node.alias if node.alias is not None else node.table
        self.write(self.dialect.quote(qualifier))
        self.write(".")
        self.write(self.dialect.quote(node.name))

    def _visit_binary(self, node: BinaryOp[Any]) -> None:
        if node.op == "ILIKE" and not self.dialect.supports_ilike:
            self.write("LOWER(")
            self.visit(node.left)
            self.write(") LIKE LOWER(")
            self.visit(node.right)
            self.write(")")
            return
        self.visit(node.left)
        self.write(f" {node.op} ")
        self.visit(node.right)

    def _visit_unary(self, node: UnaryOp[Any]) -> None:
        if node.postfix:
            self.visit(node.operand)
            self.write(f" {node.op}")
            return
        self.write(node.op)
        needs_parens = isinstance(node.operand, (LogicalOp, BinaryOp, BetweenOp, InOp))
        if needs_parens:
            self.write("(")
            self.visit(node.operand)
            self.write(")")
        else:
            self.visit(node.operand)

    def _visit_logical(self, node: LogicalOp) -> None:
        self.write("(")
        for index, clause in enumerate(node.clauses):
            if index:
                self.write(f" {node.op} ")
            self.visit(clause)
        self.write(")")

    def _visit_in(self, node: InOp[Any]) -> None:
        if not node.values:
            # Avoid invalid `IN ()` SQL; keep bind-parameter free for the fallback.
            self.write("1 = 1" if node.negated else "1 = 0")
            return
        self.visit(node.expr)
        self.write(" NOT IN (" if node.negated else " IN (")
        for index, value in enumerate(node.values):
            if index:
                self.write(", ")
            self.write(self.add_param(value))
        self.write(")")

    def _visit_between(self, node: BetweenOp[Any]) -> None:
        self.visit(node.expr)
        self.write(" BETWEEN ")
        self.visit(node.low)
        self.write(" AND ")
        self.visit(node.high)

    def _visit_func(self, node: FuncCall[Any]) -> None:
        self.write(node.name)
        self.write("(")
        if not node.args:
            self.write("*" if node.name.upper() == "COUNT" else "")
        else:
            if node.distinct:
                self.write("DISTINCT ")
            for index, arg in enumerate(node.args):
                if index:
                    self.write(", ")
                self.visit(arg)
        self.write(")")

    def _visit_raw(self, node: RawSQL[Any]) -> None:
        if not node.params:
            self.write(node.sql)
            return

        style = self.dialect.positional_style
        expected = len(node.params)
        placeholders = _collect_raw_placeholders(node.sql, self.dialect)
        if style == "dollar":
            found = set(placeholders)
            if found != set(range(1, expected + 1)):
                msg = (
                    "RawSQL dollar placeholders must be a contiguous $1..$N sequence "
                    f"matching {expected} parameter(s); found {sorted(found)}"
                )
                raise ValueError(msg)
        elif len(placeholders) != expected:
            token = "?" if style == "qmark" else "%s"
            msg = (
                f"RawSQL has {len(placeholders)} '{token}' placeholder(s) "
                f"but {expected} parameter(s)"
            )
            raise ValueError(msg)

        # Re-bind fragment parameters at the current compiler offset so composed
        # expressions do not reuse stale placeholder indexes (especially `$n`).
        rebound = [self.add_param(value) for value in node.params]
        if style == "dollar":
            self.write(_rebind_dollar_placeholders(node.sql, rebound, self.dialect))
            return

        self.write(node.sql)


def _raw_scan_options(dialect: Dialect) -> tuple[bool, bool]:
    """Return ``(mysql_backslash_strings, nested_block_comments)`` for *dialect*."""
    name = dialect.name
    mysql_backslash = name in {"mysql", "tidb"}
    nested_comments = name in {"postgres", "cockroach"}
    return mysql_backslash, nested_comments


def _iter_raw_code_spans(
    sql: str,
    *,
    mysql_backslash: bool,
    nested_block_comments: bool,
) -> list[tuple[bool, str]]:
    """Split SQL into ``(is_code, text)`` spans, skipping literals/comments."""
    spans: list[tuple[bool, str]] = []
    i = 0
    n = len(sql)
    code_start = 0

    def flush_code(end: int) -> None:
        nonlocal code_start
        if end > code_start:
            spans.append((True, sql[code_start:end]))
        code_start = end

    while i < n:
        ch = sql[i]
        # Line comment
        if ch == "-" and i + 1 < n and sql[i + 1] == "-":
            flush_code(i)
            end = sql.find("\n", i)
            if end < 0:
                spans.append((False, sql[i:]))
                return spans
            spans.append((False, sql[i : end + 1]))
            i = end + 1
            code_start = i
            continue
        # Block comment (Postgres allows nesting)
        if ch == "/" and i + 1 < n and sql[i + 1] == "*":
            flush_code(i)
            if nested_block_comments:
                depth = 1
                j = i + 2
                while j < n and depth:
                    if sql[j] == "/" and j + 1 < n and sql[j + 1] == "*":
                        depth += 1
                        j += 2
                        continue
                    if sql[j] == "*" and j + 1 < n and sql[j + 1] == "/":
                        depth -= 1
                        j += 2
                        continue
                    j += 1
                spans.append((False, sql[i:j]))
                i = j
                code_start = i
                continue
            end = sql.find("*/", i + 2)
            end = n if end < 0 else end + 2
            spans.append((False, sql[i:end]))
            i = end
            code_start = i
            continue
        # Dollar-quoted string (Postgres)
        if ch == "$":
            tag_end = i + 1
            while tag_end < n and (sql[tag_end].isalnum() or sql[tag_end] == "_"):
                tag_end += 1
            if tag_end < n and sql[tag_end] == "$":
                tag = sql[i : tag_end + 1]
                close = sql.find(tag, tag_end + 1)
                if close >= 0:
                    flush_code(i)
                    end = close + len(tag)
                    spans.append((False, sql[i:end]))
                    i = end
                    code_start = i
                    continue
        # Single-quoted literal
        if ch == "'":
            flush_code(i)
            j = i + 1
            while j < n:
                if mysql_backslash and sql[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if sql[j] == "'":
                    if j + 1 < n and sql[j + 1] == "'":
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            spans.append((False, sql[i:j]))
            i = j
            code_start = i
            continue
        # Double-quoted identifier/literal (MySQL default: string with backslash escapes)
        if ch == '"':
            flush_code(i)
            j = i + 1
            while j < n:
                if mysql_backslash and sql[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if sql[j] == '"':
                    if j + 1 < n and sql[j + 1] == '"':
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            spans.append((False, sql[i:j]))
            i = j
            code_start = i
            continue
        # MySQL backtick identifier
        if ch == "`":
            flush_code(i)
            j = i + 1
            while j < n:
                if sql[j] == "`":
                    if j + 1 < n and sql[j + 1] == "`":
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            spans.append((False, sql[i:j]))
            i = j
            code_start = i
            continue
        i += 1

    flush_code(n)
    return spans


def _collect_raw_placeholders(sql: str, dialect: Dialect) -> list[int]:
    """Collect placeholder indexes found outside literals/comments."""
    mysql_backslash, nested_comments = _raw_scan_options(dialect)
    placeholders: list[int] = []
    for is_code, span in _iter_raw_code_spans(
        sql,
        mysql_backslash=mysql_backslash,
        nested_block_comments=nested_comments,
    ):
        if is_code:
            _scan_code_span(span, dialect.positional_style, placeholders)
    return placeholders


def _rebind_dollar_placeholders(sql: str, rebound: list[str], dialect: Dialect) -> str:
    """Rewrite `$n` placeholders outside literals/comments using rebound tokens."""
    mysql_backslash, nested_comments = _raw_scan_options(dialect)
    out: list[str] = []
    for is_code, span in _iter_raw_code_spans(
        sql,
        mysql_backslash=mysql_backslash,
        nested_block_comments=nested_comments,
    ):
        if is_code:
            out.append(
                re.sub(
                    r"\$(\d+)",
                    lambda m: rebound[int(m.group(1)) - 1],
                    span,
                )
            )
        else:
            out.append(span)
    return "".join(out)


def _scan_code_span(span: str, style: str, placeholders: list[int]) -> None:
    if not span:
        return
    if style == "dollar":
        for match in re.finditer(r"\$(\d+)", span):
            placeholders.append(int(match.group(1)))
        return
    if style == "qmark":
        placeholders.extend([0] * span.count("?"))
        return
    if style == "format":
        index = 0
        while index < len(span):
            if span.startswith("%%", index):
                index += 2
                continue
            if span.startswith("%s", index):
                end = index + 2
                # Avoid treating identifiers like `%score` as a `%s` bind placeholder.
                if end < len(span) and (span[end].isalnum() or span[end] == "_"):
                    index += 1
                    continue
                placeholders.append(0)
                index = end
                continue
            index += 1
        return
    msg = f"unsupported positional_style: {style!r}"
    raise ValueError(msg)


def compile_expr(dialect: Dialect, expr: Expr[Any]) -> tuple[str, list[Any]]:
    """Compile a single expression node to SQL and bind parameters."""
    compiler = SQLCompiler(dialect)
    compiler.visit(expr)
    return compiler.build()


def compile_order_by(dialect: Dialect, clause: OrderByClause) -> tuple[str, list[Any]]:
    """Compile an ORDER BY clause fragment."""
    compiler = SQLCompiler(dialect)
    compiler.visit_order_by(clause)
    return compiler.build()
