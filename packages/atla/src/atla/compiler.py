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
        if style == "dollar":
            found = {int(match) for match in re.findall(r"\$(\d+)", node.sql)}
            if found != set(range(1, expected + 1)):
                msg = (
                    "RawSQL dollar placeholders must be a contiguous $1..$N sequence "
                    f"matching {expected} parameter(s); found {sorted(found)}"
                )
                raise ValueError(msg)
        elif style == "qmark":
            count = node.sql.count("?")
            if count != expected:
                msg = f"RawSQL has {count} '?' placeholder(s) but {expected} parameter(s)"
                raise ValueError(msg)
        elif style == "format":
            count = node.sql.replace("%%", "").count("%s")
            if count != expected:
                msg = f"RawSQL has {count} '%s' placeholder(s) but {expected} parameter(s)"
                raise ValueError(msg)
        else:
            msg = f"unsupported positional_style: {style!r}"
            raise ValueError(msg)

        # Re-bind fragment parameters at the current compiler offset so composed
        # expressions do not reuse stale placeholder indexes (especially `$n`).
        rebound = [self.add_param(value) for value in node.params]
        if style == "dollar":

            def _replace(match: re.Match[str]) -> str:
                index = int(match.group(1)) - 1
                return rebound[index]

            self.write(re.sub(r"\$(\d+)", _replace, node.sql))
            return

        # `%s` / `?` placeholders are order-based; SQL text stays unchanged.
        self.write(node.sql)


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
