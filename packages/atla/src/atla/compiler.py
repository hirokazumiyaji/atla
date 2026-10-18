"""Single-pass SQL compiler for Expression Tree AST nodes."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from atla.dialect import Dialect
from atla.expressions import (
    AliasedExpr,
    BetweenOp,
    BinaryOp,
    CaseExpr,
    CastExpr,
    Column,
    ExcludedRef,
    ExistsOp,
    Expr,
    FuncCall,
    InOp,
    LikeOp,
    Literal,
    LogicalOp,
    OrderByClause,
    Param,
    RawSQL,
    SubqueryExpr,
    UnaryOp,
)


class _UnboundSlot:
    """Sentinel replacing a ``Param`` slot in a compiled template's parameter list."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<unbound param>"


UNBOUND_SLOT = _UnboundSlot()


class SQLCompiler:
    """Compile expression AST nodes into SQL text and bind parameters."""

    __slots__ = ("buf", "dialect", "params", "slot_occurrences", "slots_enabled")

    def __init__(
        self, dialect: Dialect, initial_sql: str = "", *, slots_enabled: bool = False
    ) -> None:
        self.dialect = dialect
        self.buf: list[str] = [initial_sql] if initial_sql else []
        self.params: list[Any] = []
        self.slots_enabled = slots_enabled
        # ``(name, params[index])`` per ``Param`` occurrence, in placeholder order.
        self.slot_occurrences: list[tuple[str, int]] = []

    def add_param(self, value: Any) -> str:
        self.params.append(self.dialect.encode_param(value))
        return self.dialect.placeholder(len(self.params))

    def _add_slot(self, name: str) -> str:
        self.slot_occurrences.append((name, len(self.params)))
        self.params.append(UNBOUND_SLOT)
        return self.dialect.placeholder(len(self.params))

    def compile_value(self, value: Any) -> str:
        """Compile an expression fragment or bind a literal parameter."""
        if isinstance(value, Expr):
            return self.compile_fragment(value)  # pyright: ignore[reportUnknownArgumentType]
        return self.add_param(value)

    def write(self, fragment: str) -> None:
        self.buf.append(fragment)

    def visit(self, node: Expr[Any]) -> None:
        handler = _VISIT_BY_TYPE.get(type(node))
        if handler is not None:
            handler(self, node)
            return
        # Subclass fallback (rare); keep isinstance chain for extensibility.
        if isinstance(node, Literal):
            self.write(self.add_param(node.value))
            return
        if isinstance(node, Param):
            self._visit_param(node)
            return
        if isinstance(node, Column):
            self._visit_column(node)
            return
        if isinstance(node, BinaryOp):
            self._visit_binary(node)
            return
        if isinstance(node, LikeOp):
            self._visit_like(node)
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
        if isinstance(node, AliasedExpr):
            self._visit_aliased(node)  # pyright: ignore[reportUnknownArgumentType]
            return
        if isinstance(node, CastExpr):
            self._visit_cast(node)  # pyright: ignore[reportUnknownArgumentType]
            return
        if isinstance(node, ExcludedRef):
            self._visit_excluded(node)  # pyright: ignore[reportUnknownArgumentType]
            return
        if isinstance(node, CaseExpr):
            self._visit_case(node)  # pyright: ignore[reportUnknownArgumentType]
            return
        if isinstance(node, SubqueryExpr):
            self._visit_subquery(node)
            return
        if isinstance(node, ExistsOp):
            self._visit_exists(node)
            return
        msg = f"unsupported expression node: {type(node).__name__}"
        raise TypeError(msg)

    def _visit_literal(self, node: Literal[Any]) -> None:
        self.write(self.add_param(node.value))

    def _visit_param(self, node: Param[Any]) -> None:
        if not self.slots_enabled:
            msg = (
                f"Param({node.name!r}) requires a precompiled query; call prepare() "
                "and bind the value when executing"
            )
            raise ValueError(msg)
        self.write(self._add_slot(node.name))

    def visit_order_by(self, clause: OrderByClause) -> None:
        # direction/nulls are validated once in ``OrderByClause.__init__``.
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

    def _visit_binary_child(self, node: Expr[Any]) -> None:
        if isinstance(node, (BinaryOp, LogicalOp)):
            self.write("(")
            self.visit(node)
            self.write(")")
            return
        self.visit(node)

    def _visit_binary(self, node: BinaryOp[Any]) -> None:
        if node.op == "ILIKE" and not self.dialect.supports_ilike:
            self.write("LOWER(")
            self.visit(node.left)
            self.write(") LIKE LOWER(")
            self.visit(node.right)
            self.write(")")
            return
        self._visit_binary_child(node.left)
        self.write(f" {node.op} ")
        self._visit_binary_child(node.right)

    def _visit_like(self, node: LikeOp) -> None:
        if node.op == "ILIKE" and not self.dialect.supports_ilike:
            self.write("LOWER(")
            self.visit(node.left)
            self.write(") LIKE LOWER(")
            self.visit(node.right)
            self.write(")")
        else:
            self.visit(node.left)
            self.write(f" {node.op} ")
            self.visit(node.right)
        self.write(f" ESCAPE {self.dialect.quote_string_literal(node.escape)}")

    def _visit_unary(self, node: UnaryOp[Any]) -> None:
        if node.postfix:
            self.visit(node.operand)
            self.write(f" {node.op}")
            return
        self.write(node.op)
        needs_parens = isinstance(node.operand, (LogicalOp, BinaryOp, LikeOp, BetweenOp, InOp))
        if needs_parens:
            self.write("(")
            self.visit(node.operand)
            self.write(")")
        else:
            self.visit(node.operand)

    def _visit_aliased(self, node: AliasedExpr[Any]) -> None:
        if isinstance(node.expr, (BinaryOp, LogicalOp)):
            self.write("(")
            self.visit(node.expr)
            self.write(")")
        else:
            self.visit(node.expr)
        self.write(" AS ")
        self.write(self.dialect.quote(node.alias))

    def _visit_cast(self, node: CastExpr[Any]) -> None:
        self.write("CAST(")
        self.visit(node.expr)
        self.write(" AS ")
        self.write(node.sql_type)
        self.write(")")

    def _visit_case(self, node: CaseExpr[Any]) -> None:
        self.write("CASE")
        for arm in node.whens:
            self.write(" WHEN ")
            self.visit(arm.when)
            self.write(" THEN ")
            self.visit(arm.result)
        if node.else_ is not None:
            self.write(" ELSE ")
            self.visit(node.else_)
        self.write(" END")

    def _visit_logical(self, node: LogicalOp) -> None:
        self.write("(")
        for index, clause in enumerate(node.clauses):
            if index:
                self.write(f" {node.op} ")
            self.visit(clause)
        self.write(")")

    def _visit_in(self, node: InOp[Any]) -> None:
        if node.subquery is not None:
            self.visit(node.expr)
            self.write(" NOT IN " if node.negated else " IN ")
            self._visit_subquery(node.subquery)
            return
        if not node.values:
            # Avoid invalid `IN ()` SQL; keep bind-parameter free for the fallback.
            self.write("1 = 1" if node.negated else "1 = 0")
            return
        self.visit(node.expr)
        self.write(" NOT IN (" if node.negated else " IN (")
        for index, value in enumerate(node.values):
            if index:
                self.write(", ")
            if isinstance(value, Param):
                # Fixed-arity prepared IN lists; a variable-length list needs a
                # different query shape per arity.
                self._visit_param(value)
            else:
                self.write(self.add_param(value))
        self.write(")")

    def _visit_subquery(self, node: SubqueryExpr[Any]) -> None:
        self.write("(")
        if not node.params:
            placeholders = _collect_raw_placeholders(node.sql, self.dialect)
            if placeholders:
                msg = (
                    "SubqueryExpr has placeholders but no parameters; "
                    f"found {len(placeholders)} placeholder(s)"
                )
                raise ValueError(msg)
            self.write(node.sql)
            self.write(")")
            return
        style = self.dialect.positional_style
        expected = len(node.params)
        placeholders = _collect_raw_placeholders(node.sql, self.dialect)
        if style == "dollar":
            found = set(placeholders)
            if found != set(range(1, expected + 1)):
                msg = (
                    "SubqueryExpr dollar placeholders must be a contiguous $1..$N sequence "
                    f"matching {expected} parameter(s); found {sorted(found)}"
                )
                raise ValueError(msg)
        elif len(placeholders) != expected:
            token = "?" if style == "qmark" else "%s"
            msg = (
                f"SubqueryExpr has {len(placeholders)} '{token}' placeholder(s) "
                f"but {expected} parameter(s)"
            )
            raise ValueError(msg)
        # Re-bind nested query parameters at the current compiler offset.
        rebound = [self.add_param(value) for value in node.params]
        if style == "dollar":
            self.write(_rebind_dollar_placeholders(node.sql, rebound, self.dialect))
        else:
            # ``%s`` / ``?`` placeholders are position-independent.
            self.write(node.sql)
        self.write(")")

    def _visit_exists(self, node: ExistsOp) -> None:
        self.write("NOT EXISTS " if node.negated else "EXISTS ")
        self._visit_subquery(node.subquery)

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

    def _visit_excluded(self, node: ExcludedRef[Any]) -> None:
        quoted = self.dialect.quote(node.column.name)
        if self.dialect.upsert_style == "duplicate":
            self.write(f"VALUES({quoted})")
            return
        self.write(f"EXCLUDED.{quoted}")

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


# Dispatch table lives at module level so ``visit`` avoids per-node isinstance chains;
# it intentionally binds the compiler's own protected ``_visit_*`` methods.
_VISIT_BY_TYPE: dict[type[Any], Callable[[SQLCompiler, Any], None]] = {
    Literal: SQLCompiler._visit_literal,  # pyright: ignore[reportPrivateUsage]
    Param: SQLCompiler._visit_param,  # pyright: ignore[reportPrivateUsage]
    Column: SQLCompiler._visit_column,  # pyright: ignore[reportPrivateUsage]
    BinaryOp: SQLCompiler._visit_binary,  # pyright: ignore[reportPrivateUsage]
    UnaryOp: SQLCompiler._visit_unary,  # pyright: ignore[reportPrivateUsage]
    LogicalOp: SQLCompiler._visit_logical,  # pyright: ignore[reportPrivateUsage]
    InOp: SQLCompiler._visit_in,  # pyright: ignore[reportPrivateUsage]
    BetweenOp: SQLCompiler._visit_between,  # pyright: ignore[reportPrivateUsage]
    FuncCall: SQLCompiler._visit_func,  # pyright: ignore[reportPrivateUsage]
    RawSQL: SQLCompiler._visit_raw,  # pyright: ignore[reportPrivateUsage]
}


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
