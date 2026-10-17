"""Expression Tree AST for type-safe SQL construction."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any, NoReturn, cast
from typing import Literal as TypingLiteral

OrderDirection = TypingLiteral["ASC", "DESC"]
NullsPosition = TypingLiteral["FIRST", "LAST"]

_FUNC_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")

# Exact tokens emitted by the expression API / compiler (no free-form SQL).
_BINARY_OPS = frozenset(
    {
        "=",
        "<>",
        "<",
        "<=",
        ">",
        ">=",
        "+",
        "-",
        "*",
        "LIKE",
        "ILIKE",
    }
)
_UNARY_OPS = frozenset({"NOT ", "IS NULL", "IS NOT NULL"})
_LOGICAL_OPS = frozenset({"AND", "OR"})


def _coerce_literal[T](value: T | Expr[T]) -> Expr[T]:
    if isinstance(value, Expr):
        return cast(Expr[T], value)
    return Literal(value)


def _validate_binary_op(op: str) -> str:
    if op not in _BINARY_OPS:
        msg = f"invalid SQL binary operator: {op!r}"
        raise ValueError(msg)
    return op


def _validate_unary_op(op: str) -> str:
    if op not in _UNARY_OPS:
        msg = f"invalid SQL unary operator: {op!r}"
        raise ValueError(msg)
    return op


def _validate_logical_op(op: str) -> str:
    if op not in _LOGICAL_OPS:
        msg = f"invalid SQL logical operator: {op!r}"
        raise ValueError(msg)
    return op


def _validate_func_name(name: str) -> str:
    if not _FUNC_NAME_RE.fullmatch(name):
        msg = f"invalid SQL function name: {name!r}"
        raise ValueError(msg)
    return name


def _normalize_order_direction(direction: str) -> OrderDirection:
    normalized = direction.upper()
    if normalized not in {"ASC", "DESC"}:
        msg = f"ORDER BY direction must be ASC or DESC, got {direction!r}"
        raise ValueError(msg)
    return cast(OrderDirection, normalized)


def _normalize_nulls(nulls: str | None) -> NullsPosition | None:
    if nulls is None:
        return None
    normalized = nulls.upper()
    if normalized not in {"FIRST", "LAST"}:
        msg = f"ORDER BY nulls must be FIRST or LAST, got {nulls!r}"
        raise ValueError(msg)
    return cast(NullsPosition, normalized)


# Use ``!`` rather than ``\`` so ESCAPE literals stay portable on MySQL/TiDB
# (where ``\`` is a string-literal escape by default).
_LIKE_ESCAPE = "!"


def escape_like(value: str) -> str:
    """Escape ``!``, ``%``, and ``_`` for LIKE patterns using ``ESCAPE '!'``."""
    return (
        value.replace(_LIKE_ESCAPE, _LIKE_ESCAPE * 2)
        .replace("%", f"{_LIKE_ESCAPE}%")
        .replace("_", f"{_LIKE_ESCAPE}_")
    )


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
        self.op = _validate_binary_op(op)
        self.left = left
        self.right = right


class LikeOp(Expr[bool]):
    """LIKE / ILIKE with a mandatory ESCAPE clause and a pre-escaped pattern."""

    __slots__ = ("escape", "left", "op", "right")

    def __init__(
        self,
        op: TypingLiteral["LIKE", "ILIKE"],
        left: Expr[Any],
        pattern: str,
        *,
        escape: str = _LIKE_ESCAPE,
    ) -> None:
        if op not in {"LIKE", "ILIKE"}:
            msg = f"invalid LIKE operator: {op!r}"
            raise ValueError(msg)
        if len(escape) != 1 or "\x00" in escape:
            msg = f"LIKE ESCAPE must be a single non-NUL character, got {escape!r}"
            raise ValueError(msg)
        self.op = op
        self.left = left
        self.right = Literal(pattern)
        self.escape = escape


class UnaryOp[T](Expr[T]):
    """Unary operation node (NOT, IS NULL, IS NOT NULL, ...)."""

    __slots__ = ("op", "operand", "postfix")

    def __init__(self, op: str, operand: Expr[Any], *, postfix: bool = False) -> None:
        self.op = _validate_unary_op(op)
        self.operand = operand
        self.postfix = postfix


class LogicalOp(Expr[bool]):
    """Flattened AND / OR of boolean clauses."""

    __slots__ = ("clauses", "op")

    def __init__(self, op: str, clauses: tuple[Expr[bool], ...]) -> None:
        self.op = _validate_logical_op(op)
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
        self.name = _validate_func_name(name)
        self.args = args
        self.distinct = distinct


class RawSQL[T](Expr[T]):
    """Type-annotated raw SQL escape hatch."""

    __slots__ = ("params", "sql")

    def __init__(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        self.sql = sql
        self.params = params


class ExcludedRef[T](Expr[T]):
    """Upsert conflict-row reference (``EXCLUDED.col`` / ``VALUES(col)``)."""

    __slots__ = ("column",)

    def __init__(self, column: Column[T]) -> None:
        self.column = column


class OrderByClause:
    """ORDER BY clause fragment."""

    __slots__ = ("direction", "expr", "nulls")

    def __init__(
        self,
        expr: Expr[Any],
        *,
        direction: OrderDirection,
        nulls: str | None = None,
    ) -> None:
        # Validated at construction (runtime guard for untyped callers); the
        # compiler trusts these slots and does not re-check them.
        self.expr = expr
        self.direction = _normalize_order_direction(direction)
        self.nulls = _normalize_nulls(nulls)


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
        """Raw SQL ``LIKE`` — ``%`` / ``_`` keep wildcard meaning.

        For literal substring / prefix / suffix matching (with automatic
        escaping and ``ESCAPE '!'``), use :meth:`contains`, :meth:`starts_with`,
        or :meth:`ends_with` instead.
        """
        return BinaryOp("LIKE", self, Literal(pattern))

    def ilike(self, pattern: str) -> BinaryOp[bool]:
        """Raw SQL ``ILIKE`` / lowercased ``LIKE`` — wildcards stay active.

        Prefer :meth:`icontains`, :meth:`istarts_with`, or :meth:`iends_with`
        when the input should be matched literally.
        """
        return BinaryOp("ILIKE", self, Literal(pattern))

    def contains(self, substring: str) -> LikeOp:
        """Substring match via ``LIKE`` with escaped wildcards and ``ESCAPE '!'``."""
        return LikeOp("LIKE", self, f"%{escape_like(substring)}%")

    def starts_with(self, prefix: str) -> LikeOp:
        """Prefix match via ``LIKE`` with escaped wildcards and ``ESCAPE '!'``."""
        return LikeOp("LIKE", self, f"{escape_like(prefix)}%")

    def ends_with(self, suffix: str) -> LikeOp:
        """Suffix match via ``LIKE`` with escaped wildcards and ``ESCAPE '!'``."""
        return LikeOp("LIKE", self, f"%{escape_like(suffix)}")

    def icontains(self, substring: str) -> LikeOp:
        """Case-insensitive substring match (``ILIKE`` / lowercased ``LIKE``)."""
        return LikeOp("ILIKE", self, f"%{escape_like(substring)}%")

    def istarts_with(self, prefix: str) -> LikeOp:
        """Case-insensitive prefix match."""
        return LikeOp("ILIKE", self, f"{escape_like(prefix)}%")

    def iends_with(self, suffix: str) -> LikeOp:
        """Case-insensitive suffix match."""
        return LikeOp("ILIKE", self, f"%{escape_like(suffix)}")

    # --- Arithmetic (e.g. UPDATE User.age + 1) ---

    def __add__(self, other: T | Expr[T]) -> BinaryOp[T]:
        return BinaryOp("+", self, _coerce_literal(other))

    def __sub__(self, other: T | Expr[T]) -> BinaryOp[T]:
        return BinaryOp("-", self, _coerce_literal(other))

    def __mul__(self, other: T | Expr[T]) -> BinaryOp[T]:
        return BinaryOp("*", self, _coerce_literal(other))

    # --- Ordering ---

    def asc(self, *, nulls: NullsPosition | None = None) -> OrderByClause:
        return OrderByClause(self, direction="ASC", nulls=nulls)

    def desc(self, *, nulls: NullsPosition | None = None) -> OrderByClause:
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


def excluded[T](column: Column[T]) -> ExcludedRef[T]:
    """Reference the proposed INSERT row in an upsert ``DO UPDATE`` assignment.

    Compiles to ``EXCLUDED."col"`` (Postgres / CockroachDB / SQLite) or
    ``VALUES(`col`)`` (MySQL / TiDB).
    """
    return ExcludedRef(column)
