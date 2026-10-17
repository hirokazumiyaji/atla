"""Expression Tree AST for type-safe SQL construction."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, NoReturn, overload
from typing import cast as typing_cast

_SQL_TYPE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_ ]*(\([\d,\s]+\))?(\[\])?$")


def _coerce_literal[T](value: T | Expr[T]) -> Expr[T]:
    if isinstance(value, Expr):
        return typing_cast(Expr[T], value)
    return Literal(value)


class Expr[T_co]:
    """Base class for all SQL expression AST nodes."""

    __slots__ = ()

    # ``__eq__`` builds SQL predicates, not value equality. Identity hashing is
    # intentional (``object.__hash__``): do not derive a value hash from SQL form.
    # ``set`` / ``dict`` membership is therefore by object identity only.
    __hash__ = object.__hash__  # type: ignore[assignment, unused-ignore]  # pyright: ignore[reportAssignmentType]

    def __bool__(self) -> NoReturn:
        raise TypeError(
            "SQL expression cannot be evaluated as a Python boolean. "
            "Use 'and_(a, b)' / 'a & b' or 'or_(a, b)' / 'a | b' instead of 'and' / 'or', "
            "and '.is_null()' / '.is_not_null()' instead of 'is None'."
        )

    def __and__(self: Expr[bool], other: Expr[bool]) -> LogicalOp:
        return typing_cast(LogicalOp, and_(self, other))

    def __or__(self: Expr[bool], other: Expr[bool]) -> LogicalOp:
        return typing_cast(LogicalOp, or_(self, other))

    def __invert__(self: Expr[bool]) -> UnaryOp[bool]:
        return UnaryOp("NOT ", self, postfix=False)

    # --- Comparison ---

    @overload  # type: ignore[override]
    def __eq__[T](self: Expr[T], other: None) -> UnaryOp[bool]: ...

    @overload
    def __eq__[T](self: Expr[T], other: T | Expr[T]) -> BinaryOp[bool]: ...

    def __eq__[T](  # pyright: ignore[reportIncompatibleMethodOverride]
        self: Expr[T], other: T | Expr[T] | None
    ) -> BinaryOp[bool] | UnaryOp[bool]:
        if other is None:
            return UnaryOp("IS NULL", self, postfix=True)
        return BinaryOp("=", self, _coerce_literal(other))

    @overload  # type: ignore[override]
    def __ne__[T](self: Expr[T], other: None) -> UnaryOp[bool]: ...

    @overload
    def __ne__[T](self: Expr[T], other: T | Expr[T]) -> BinaryOp[bool]: ...

    def __ne__[T](  # pyright: ignore[reportIncompatibleMethodOverride]
        self: Expr[T], other: T | Expr[T] | None
    ) -> BinaryOp[bool] | UnaryOp[bool]:
        if other is None:
            return UnaryOp("IS NOT NULL", self, postfix=True)
        return BinaryOp("<>", self, _coerce_literal(other))

    def __lt__[T](self: Expr[T], other: T | Expr[T]) -> BinaryOp[bool]:
        return BinaryOp("<", self, _coerce_literal(other))

    def __le__[T](self: Expr[T], other: T | Expr[T]) -> BinaryOp[bool]:
        return BinaryOp("<=", self, _coerce_literal(other))

    def __gt__[T](self: Expr[T], other: T | Expr[T]) -> BinaryOp[bool]:
        return BinaryOp(">", self, _coerce_literal(other))

    def __ge__[T](self: Expr[T], other: T | Expr[T]) -> BinaryOp[bool]:
        return BinaryOp(">=", self, _coerce_literal(other))

    @overload
    def eq[T](self: Expr[T], other: None) -> UnaryOp[bool]: ...

    @overload
    def eq[T](self: Expr[T], other: T | Expr[T]) -> BinaryOp[bool]: ...

    def eq[T](self: Expr[T], other: T | Expr[T] | None) -> BinaryOp[bool] | UnaryOp[bool]:
        # Match ``__eq__``: ``None`` becomes ``IS NULL``, not ``= NULL``.
        return self == other

    @overload
    def ne[T](self: Expr[T], other: None) -> UnaryOp[bool]: ...

    @overload
    def ne[T](self: Expr[T], other: T | Expr[T]) -> BinaryOp[bool]: ...

    def ne[T](self: Expr[T], other: T | Expr[T] | None) -> BinaryOp[bool] | UnaryOp[bool]:
        return self != other

    def lt[T](self: Expr[T], other: T | Expr[T]) -> BinaryOp[bool]:
        return BinaryOp("<", self, _coerce_literal(other))

    def le[T](self: Expr[T], other: T | Expr[T]) -> BinaryOp[bool]:
        return BinaryOp("<=", self, _coerce_literal(other))

    def gt[T](self: Expr[T], other: T | Expr[T]) -> BinaryOp[bool]:
        return BinaryOp(">", self, _coerce_literal(other))

    def ge[T](self: Expr[T], other: T | Expr[T]) -> BinaryOp[bool]:
        return BinaryOp(">=", self, _coerce_literal(other))

    def is_null(self) -> UnaryOp[bool]:
        return UnaryOp("IS NULL", self, postfix=True)

    def is_not_null(self) -> UnaryOp[bool]:
        return UnaryOp("IS NOT NULL", self, postfix=True)

    def in_(self, values: Sequence[Any]) -> InOp[Any]:
        return InOp(self, tuple(values), negated=False)

    def not_in(self, values: Sequence[Any]) -> InOp[Any]:
        return InOp(self, tuple(values), negated=True)

    def between(self, low: Any, high: Any) -> BetweenOp[Any]:
        return BetweenOp(self, _coerce_literal(low), _coerce_literal(high))

    def like(self, pattern: str) -> BinaryOp[bool]:
        return BinaryOp("LIKE", self, Literal(pattern))

    def ilike(self, pattern: str) -> BinaryOp[bool]:
        return BinaryOp("ILIKE", self, Literal(pattern))

    def contains(self, substring: str) -> BinaryOp[bool]:
        return self.like(f"%{substring}%")

    def starts_with(self, prefix: str) -> BinaryOp[bool]:
        return self.like(f"{prefix}%")

    def ends_with(self, suffix: str) -> BinaryOp[bool]:
        return self.like(f"%{suffix}")

    # --- Arithmetic ---

    def __add__[T](self: Expr[T], other: T | Expr[T]) -> BinaryOp[T]:
        return BinaryOp("+", self, _coerce_literal(other))

    def __radd__[T](self: Expr[T], other: T | Expr[T]) -> BinaryOp[T]:
        return BinaryOp("+", _coerce_literal(other), self)

    def __sub__[T](self: Expr[T], other: T | Expr[T]) -> BinaryOp[T]:
        return BinaryOp("-", self, _coerce_literal(other))

    def __rsub__[T](self: Expr[T], other: T | Expr[T]) -> BinaryOp[T]:
        return BinaryOp("-", _coerce_literal(other), self)

    def __mul__[T](self: Expr[T], other: T | Expr[T]) -> BinaryOp[T]:
        return BinaryOp("*", self, _coerce_literal(other))

    def __rmul__[T](self: Expr[T], other: T | Expr[T]) -> BinaryOp[T]:
        return BinaryOp("*", _coerce_literal(other), self)

    def __truediv__[T](self: Expr[T], other: T | Expr[T]) -> BinaryOp[T]:
        return BinaryOp("/", self, _coerce_literal(other))

    def __rtruediv__[T](self: Expr[T], other: T | Expr[T]) -> BinaryOp[T]:
        return BinaryOp("/", _coerce_literal(other), self)

    def __mod__[T](self: Expr[T], other: T | Expr[T]) -> BinaryOp[T]:
        return BinaryOp("%", self, _coerce_literal(other))

    def __rmod__[T](self: Expr[T], other: T | Expr[T]) -> BinaryOp[T]:
        return BinaryOp("%", _coerce_literal(other), self)

    def __neg__[T](self: Expr[T]) -> UnaryOp[T]:
        return UnaryOp("-", self, postfix=False)

    # --- Ordering / alias ---

    def asc(self, *, nulls: str | None = None) -> OrderByClause:
        return OrderByClause(self, direction="ASC", nulls=nulls)

    def desc(self, *, nulls: str | None = None) -> OrderByClause:
        return OrderByClause(self, direction="DESC", nulls=nulls)

    def as_(self, alias: str) -> AliasedExpr[T_co]:
        return AliasedExpr(self, alias)


class Literal[T](Expr[T]):
    """Bind-parameter value node."""

    __slots__ = ("value",)

    def __init__(self, value: T) -> None:
        self.value = value


class BinaryOp[T](Expr[T]):
    """Binary operation node (comparison, arithmetic, LIKE, ...)."""

    __slots__ = ("left", "op", "right")

    def __init__(self, op: str, left: Expr[Any], right: Expr[Any]) -> None:
        self.op = op
        self.left = left
        self.right = right


class UnaryOp[T](Expr[T]):
    """Unary operation node (NOT, IS NULL, IS NOT NULL, ...)."""

    __slots__ = ("op", "operand", "postfix")

    def __init__(self, op: str, operand: Expr[Any], *, postfix: bool = False) -> None:
        self.op = op
        self.operand = operand
        self.postfix = postfix


class LogicalOp(Expr[bool]):
    """Flattened AND / OR of boolean clauses."""

    __slots__ = ("clauses", "op")

    def __init__(self, op: str, clauses: tuple[Expr[bool], ...]) -> None:
        self.op = op
        self.clauses = clauses


class InOp[T](Expr[bool]):
    """IN / NOT IN predicate. Empty values compile to a safe tautology/contradiction later."""

    __slots__ = ("expr", "negated", "values")

    def __init__(self, expr: Expr[T], values: tuple[T, ...], *, negated: bool) -> None:
        self.expr = expr
        self.values = values
        self.negated = negated


class BetweenOp[T](Expr[bool]):
    """BETWEEN low AND high predicate."""

    __slots__ = ("expr", "high", "low")

    def __init__(self, expr: Expr[T], low: Expr[T], high: Expr[T]) -> None:
        self.expr = expr
        self.low = low
        self.high = high


class FuncCall[T](Expr[T]):
    """SQL function call (COUNT, SUM, AVG, ...)."""

    __slots__ = ("args", "distinct", "name")

    def __init__(
        self,
        name: str,
        args: tuple[Expr[Any], ...],
        *,
        distinct: bool = False,
    ) -> None:
        self.name = name
        self.args = args
        self.distinct = distinct


class RawSQL[T](Expr[T]):
    """Type-annotated raw SQL escape hatch."""

    __slots__ = ("params", "sql")

    def __init__(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        self.sql = sql
        self.params = params


class AliasedExpr[T](Expr[T]):
    """Expression projected under an SQL alias (``expr AS name``)."""

    __slots__ = ("alias", "expr")

    def __init__(self, expr: Expr[T], alias: str) -> None:
        self.expr = expr
        self.alias = alias


class CastExpr[T](Expr[T]):
    """SQL ``CAST(expr AS type)``."""

    __slots__ = ("expr", "sql_type")

    def __init__(self, expr: Expr[Any], sql_type: str) -> None:
        self.expr = expr
        self.sql_type = sql_type


class WhenThen:
    """One ``WHEN … THEN …`` arm of a ``CASE`` expression."""

    __slots__ = ("result", "when")

    def __init__(self, when: Expr[bool], result: Expr[Any]) -> None:
        self.when = when
        self.result = result


class CaseExpr[T](Expr[T]):
    """SQL ``CASE WHEN … THEN … ELSE … END``."""

    __slots__ = ("else_", "whens")

    def __init__(
        self,
        whens: tuple[WhenThen, ...],
        else_: Expr[T] | None = None,
    ) -> None:
        self.whens = whens
        self.else_ = else_


class OrderByClause:
    """ORDER BY clause fragment."""

    __slots__ = ("direction", "expr", "nulls")

    def __init__(
        self,
        expr: Expr[Any],
        *,
        direction: str,
        nulls: str | None = None,
    ) -> None:
        self.expr = expr
        self.direction = direction
        self.nulls = nulls


class Column[T](Expr[T]):
    """Typed column descriptor used in generated table models.

    ``__eq__`` / ``__ne__`` return SQL predicates, not value equality.
    ``__hash__ = object.__hash__`` is intentional so hashing stays identity-based;
    ``set`` / ``dict`` keys compare by descriptor identity, never by SQL text.
    """

    __slots__ = ("alias", "name", "table")

    def __init__(self, table: str, name: str, alias: str | None = None) -> None:
        self.table = table
        self.name = name
        self.alias = alias


def _flatten_logical(op: str, clauses: tuple[Expr[bool], ...]) -> tuple[Expr[bool], ...]:
    flat: list[Expr[bool]] = []
    for clause in clauses:
        if isinstance(clause, LogicalOp) and clause.op == op:
            flat.extend(clause.clauses)
        else:
            flat.append(clause)
    return tuple(flat)


def and_(*clauses: Expr[bool]) -> Expr[bool]:
    """Combine predicates with AND, flattening nested AND nodes."""
    if not clauses:
        msg = "and_() requires at least one clause"
        raise ValueError(msg)
    flat = _flatten_logical("AND", clauses)
    if len(flat) == 1:
        return flat[0]
    return LogicalOp("AND", flat)


def or_(*clauses: Expr[bool]) -> Expr[bool]:
    """Combine predicates with OR, flattening nested OR nodes."""
    if not clauses:
        msg = "or_() requires at least one clause"
        raise ValueError(msg)
    flat = _flatten_logical("OR", clauses)
    if len(flat) == 1:
        return flat[0]
    return LogicalOp("OR", flat)


def not_(clause: Expr[bool]) -> Expr[bool]:
    """Negate a boolean expression."""
    return UnaryOp("NOT ", clause, postfix=False)


def count(expr: Expr[Any] | None = None, *, distinct: bool = False) -> FuncCall[int]:
    """SQL COUNT aggregate."""
    args: tuple[Expr[Any], ...] = (expr,) if expr is not None else ()
    return FuncCall("COUNT", args, distinct=distinct)


def sum_[T](expr: Expr[T]) -> FuncCall[T | None]:
    """SQL SUM aggregate."""
    return FuncCall("SUM", (expr,))


def avg(expr: Expr[Any]) -> FuncCall[float | None]:
    """SQL AVG aggregate."""
    return FuncCall("AVG", (expr,))


def min_[T](expr: Expr[T]) -> FuncCall[T | None]:
    """SQL MIN aggregate."""
    return FuncCall("MIN", (expr,))


def max_[T](expr: Expr[T]) -> FuncCall[T | None]:
    """SQL MAX aggregate."""
    return FuncCall("MAX", (expr,))


def coalesce[T](*exprs: Expr[T] | T) -> FuncCall[T]:
    """SQL ``COALESCE``."""
    if not exprs:
        msg = "coalesce() requires at least one argument"
        raise ValueError(msg)
    return FuncCall("COALESCE", tuple(_coerce_literal(expr) for expr in exprs))


def nullif[T](left: Expr[T] | T, right: Expr[T] | T) -> FuncCall[T | None]:
    """SQL ``NULLIF``."""
    return FuncCall("NULLIF", (_coerce_literal(left), _coerce_literal(right)))


def cast[T](
    expr: Expr[Any] | Any,
    sql_type: str,
    python_type: type[T] | None = None,
) -> CastExpr[T]:
    """SQL ``CAST(expr AS sql_type)``.

    Pass ``python_type`` to pin the result's static type, e.g.
    ``cast(col, "text", str)``.

    ``sql_type`` is concatenated into SQL as a trusted developer-supplied type
    name (not user input). Only simple type tokens are accepted.
    """
    if not _SQL_TYPE_RE.fullmatch(sql_type):
        msg = f"invalid SQL type: {sql_type!r}"
        raise ValueError(msg)
    _ = python_type
    return CastExpr(_coerce_literal(expr), sql_type)


def when(condition: Expr[bool], result: Expr[Any] | Any) -> WhenThen:
    """Build a ``WHEN … THEN …`` arm for :func:`case`."""
    return WhenThen(condition, _coerce_literal(result))


def case[T](*whens: WhenThen, else_: Expr[T] | T | None = None) -> CaseExpr[T]:
    """SQL searched ``CASE`` expression."""
    if not whens:
        msg = "case() requires at least one when(...)"
        raise ValueError(msg)
    else_expr = None if else_ is None else _coerce_literal(else_)
    return CaseExpr(whens, else_expr)


def lower(expr: Expr[str] | str) -> FuncCall[str]:
    """SQL ``LOWER``."""
    return FuncCall("LOWER", (_coerce_literal(expr),))


def upper(expr: Expr[str] | str) -> FuncCall[str]:
    """SQL ``UPPER``."""
    return FuncCall("UPPER", (_coerce_literal(expr),))


def concat(*exprs: Expr[Any] | Any) -> FuncCall[str]:
    """SQL ``CONCAT``.

    Requires SQLite 3.44+ (earlier versions have no ``CONCAT``; use ``||``).
    Postgres / MySQL / TiDB / CockroachDB support ``CONCAT`` natively.
    """
    if not exprs:
        msg = "concat() requires at least one argument"
        raise ValueError(msg)
    return FuncCall("CONCAT", tuple(_coerce_literal(expr) for expr in exprs))


def now() -> FuncCall[Any]:
    """SQL ``NOW()`` / current timestamp function.

    Not available on SQLite — use ``raw_sql("CURRENT_TIMESTAMP")`` or
    ``raw_sql("datetime('now')")`` instead.
    """
    return FuncCall("NOW", ())


def raw_sql(sql: str, *params: Any) -> RawSQL[Any]:
    """Escape hatch for dialect-specific or unsupported SQL fragments."""
    return RawSQL(sql, params)
