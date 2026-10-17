"""Expression Tree AST for type-safe SQL construction."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, NoReturn, cast


def _coerce_literal[T](value: T | Expr[T]) -> Expr[T]:
    if isinstance(value, Expr):
        return cast(Expr[T], value)
    return Literal(value)


class Expr[T_co]:
    """Base class for all SQL expression AST nodes."""

    __slots__ = ()

    def __bool__(self) -> NoReturn:
        raise TypeError(
            "SQL expression cannot be evaluated as a Python boolean. "
            "Use 'and_(a, b)' / 'a & b' or 'or_(a, b)' / 'a | b' instead of 'and' / 'or', "
            "and '.is_null()' / '.is_not_null()' instead of 'is None'."
        )

    def __and__(self: Expr[bool], other: Expr[bool]) -> LogicalOp:
        return cast(LogicalOp, and_(self, other))

    def __or__(self: Expr[bool], other: Expr[bool]) -> LogicalOp:
        return cast(LogicalOp, or_(self, other))

    def __invert__(self: Expr[bool]) -> UnaryOp[bool]:
        return UnaryOp("NOT ", self, postfix=False)


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

    ``__eq__`` / ``__ne__`` return SQL predicates, so equality is not
    identity-based. Hashing uses object identity (``__hash__ = object.__hash__``)
    so columns remain usable as ``set`` / ``dict`` keys by descriptor identity.

    Optional ``decoder`` applies the same row codec used by ``Record._from_row``
    when the column is projected via ``select()``.
    """

    __slots__ = ("alias", "decoder", "name", "table")

    def __init__(
        self,
        table: str,
        name: str,
        alias: str | None = None,
        *,
        decoder: Callable[[Any], Any] | None = None,
    ) -> None:
        self.table = table
        self.name = name
        self.alias = alias
        self.decoder = decoder

    # Identity hash: ``__eq__`` builds SQL, so value-equality hashing is undefined.
    __hash__ = object.__hash__

    # --- Comparison operators ---

    def __eq__(  # type: ignore[override]
        self, other: T | Expr[T] | None
    ) -> BinaryOp[bool] | UnaryOp[bool]:
        if other is None:
            return UnaryOp("IS NULL", self, postfix=True)
        return BinaryOp("=", self, _coerce_literal(other))

    def __ne__(  # type: ignore[override]
        self, other: T | Expr[T] | None
    ) -> BinaryOp[bool] | UnaryOp[bool]:
        if other is None:
            return UnaryOp("IS NOT NULL", self, postfix=True)
        return BinaryOp("<>", self, _coerce_literal(other))

    def __lt__(self, other: T | Expr[T]) -> BinaryOp[bool]:
        return BinaryOp("<", self, _coerce_literal(other))

    def __le__(self, other: T | Expr[T]) -> BinaryOp[bool]:
        return BinaryOp("<=", self, _coerce_literal(other))

    def __gt__(self, other: T | Expr[T]) -> BinaryOp[bool]:
        return BinaryOp(">", self, _coerce_literal(other))

    def __ge__(self, other: T | Expr[T]) -> BinaryOp[bool]:
        return BinaryOp(">=", self, _coerce_literal(other))

    # --- Explicit comparison methods (linter-friendly) ---

    def eq(self, other: T | Expr[T]) -> Expr[bool]:
        return self == other

    def ne(self, other: T | Expr[T]) -> Expr[bool]:
        return self != other

    def lt(self, other: T | Expr[T]) -> BinaryOp[bool]:
        return self < other

    def le(self, other: T | Expr[T]) -> BinaryOp[bool]:
        return self <= other

    def gt(self, other: T | Expr[T]) -> BinaryOp[bool]:
        return self > other

    def ge(self, other: T | Expr[T]) -> BinaryOp[bool]:
        return self >= other

    def is_null(self) -> UnaryOp[bool]:
        return UnaryOp("IS NULL", self, postfix=True)

    def is_not_null(self) -> UnaryOp[bool]:
        return UnaryOp("IS NOT NULL", self, postfix=True)

    # --- Collection, range & pattern matching ---

    def in_(self, values: Sequence[T]) -> InOp[T]:
        return InOp(self, tuple(values), negated=False)

    def not_in(self, values: Sequence[T]) -> InOp[T]:
        return InOp(self, tuple(values), negated=True)

    def between(self, low: T | Expr[T], high: T | Expr[T]) -> BetweenOp[T]:
        return BetweenOp(self, _coerce_literal(low), _coerce_literal(high))

    def like(self, pattern: str) -> BinaryOp[bool]:
        """SQL ``LIKE``; callers must escape ``%`` / ``_`` (and dialect ESCAPE) themselves."""
        return BinaryOp("LIKE", self, Literal(pattern))

    def ilike(self, pattern: str) -> BinaryOp[bool]:
        """SQL ``ILIKE`` / lowercased ``LIKE``; same escape rules as :meth:`like`."""
        return BinaryOp("ILIKE", self, Literal(pattern))

    def contains(self, substring: str) -> BinaryOp[bool]:
        """Substring match via ``LIKE '%…%'``.

        ``%`` and ``_`` in *substring* keep SQL LIKE wildcard meaning; escape
        them (or use :meth:`like` with an ``ESCAPE`` clause) before calling.
        """
        return self.like(f"%{substring}%")

    def starts_with(self, prefix: str) -> BinaryOp[bool]:
        """Prefix match via ``LIKE '…%'``; see :meth:`contains` for wildcard escaping."""
        return self.like(f"{prefix}%")

    def ends_with(self, suffix: str) -> BinaryOp[bool]:
        """Suffix match via ``LIKE '%…'``; see :meth:`contains` for wildcard escaping."""
        return self.like(f"%{suffix}")

    # --- Arithmetic (e.g. UPDATE User.age + 1) ---

    def __add__(self, other: T | Expr[T]) -> BinaryOp[T]:
        return BinaryOp("+", self, _coerce_literal(other))

    def __sub__(self, other: T | Expr[T]) -> BinaryOp[T]:
        return BinaryOp("-", self, _coerce_literal(other))

    def __mul__(self, other: T | Expr[T]) -> BinaryOp[T]:
        return BinaryOp("*", self, _coerce_literal(other))

    # --- Ordering ---

    def asc(self, *, nulls: str | None = None) -> OrderByClause:
        return OrderByClause(self, direction="ASC", nulls=nulls)

    def desc(self, *, nulls: str | None = None) -> OrderByClause:
        return OrderByClause(self, direction="DESC", nulls=nulls)


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


def raw_sql(sql: str, *params: Any) -> RawSQL[Any]:
    """Escape hatch for dialect-specific or unsupported SQL fragments."""
    return RawSQL(sql, params)
