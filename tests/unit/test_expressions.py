"""Unit tests for the Expression Tree AST."""

from __future__ import annotations

import pytest

from atla import (
    BetweenOp,
    BinaryOp,
    Column,
    FuncCall,
    InOp,
    Literal,
    LogicalOp,
    OrderByClause,
    RawSQL,
    UnaryOp,
    and_,
    avg,
    count,
    max_,
    min_,
    not_,
    or_,
    raw_sql,
    sum_,
)


def test_expr_nodes_use_slots() -> None:
    age = Column[int]("users", "age")
    nodes: list[object] = [
        age,
        Literal(1),
        BinaryOp[bool]("=", age, Literal(1)),
        UnaryOp[bool]("IS NULL", age, postfix=True),
        LogicalOp("AND", (age.eq(1), age.eq(2))),
        InOp(age, (1, 2), negated=False),
        BetweenOp(age, Literal(1), Literal(10)),
        FuncCall[int]("COUNT", ()),
        RawSQL[bool]("1 = 1"),
        OrderByClause(age, direction="ASC"),
    ]
    for node in nodes:
        assert not hasattr(node, "__dict__")
        assert hasattr(type(node), "__slots__")


def test_bool_guard_raises_type_error() -> None:
    age = Column[int]("users", "age")
    with pytest.raises(TypeError, match="SQL expression cannot be evaluated"):
        bool(age >= 20)


def test_comparison_and_null_predicates() -> None:
    age = Column[int]("users", "age")
    deleted_at = Column[object]("users", "deleted_at")

    eq = age == 20
    assert isinstance(eq, BinaryOp)
    assert eq.op == "="
    assert isinstance(eq.right, Literal)
    assert eq.right.value == 20

    is_null = deleted_at == None  # noqa: E711
    assert isinstance(is_null, UnaryOp)
    assert is_null.op == "IS NULL"
    assert is_null.postfix is True

    is_not_null = deleted_at != None  # noqa: E711
    assert isinstance(is_not_null, UnaryOp)
    assert is_not_null.op == "IS NOT NULL"

    assert isinstance(age.is_null(), UnaryOp)
    assert isinstance(age.eq(1), BinaryOp)
    assert isinstance(age.ge(18), BinaryOp)


def test_collection_range_and_string_methods() -> None:
    age = Column[int]("users", "age")
    name = Column[str]("users", "name")

    in_expr = age.in_([1, 2, 3])
    assert isinstance(in_expr, InOp)
    assert in_expr.values == (1, 2, 3)
    assert in_expr.negated is False

    empty_not_in = age.not_in([])
    assert empty_not_in.values == ()
    assert empty_not_in.negated is True

    between = age.between(1, 10)
    assert isinstance(between, BetweenOp)
    assert isinstance(between.low, Literal)
    assert between.low.value == 1

    like = name.like("a%")
    assert isinstance(like, BinaryOp)
    assert like.op == "LIKE"

    contains = name.contains("al")
    assert isinstance(contains.right, Literal)
    assert contains.right.value == "%al%"
    starts = name.starts_with("A")
    assert isinstance(starts.right, Literal)
    assert starts.right.value == "A%"
    ends = name.ends_with("e")
    assert isinstance(ends.right, Literal)
    assert ends.right.value == "%e"
    assert name.ilike("%x%").op == "ILIKE"


def test_arithmetic_and_ordering() -> None:
    age = Column[int]("users", "age")
    created = Column[object]("users", "created_at")

    plus = age + 1
    assert isinstance(plus, BinaryOp)
    assert plus.op == "+"
    assert isinstance((age - 1), BinaryOp)
    assert isinstance((age * 2), BinaryOp)

    asc = created.asc(nulls="FIRST")
    assert isinstance(asc, OrderByClause)
    assert asc.direction == "ASC"
    assert asc.nulls == "FIRST"
    assert created.desc().direction == "DESC"


def test_logical_helpers_flatten() -> None:
    age = Column[int]("users", "age")
    active = Column[bool]("users", "is_active")

    left = and_(age >= 20, active.eq(True))
    right = and_(age < 65, active.eq(True))
    combined = and_(left, right)
    assert isinstance(combined, LogicalOp)
    assert combined.op == "AND"
    assert len(combined.clauses) == 4

    bit_and = (age >= 20) & active.eq(True)
    assert isinstance(bit_and, LogicalOp)
    assert bit_and.op == "AND"

    bit_or = (age < 18) | (age > 65)
    assert isinstance(bit_or, LogicalOp)
    assert bit_or.op == "OR"
    assert isinstance(or_(age < 18, age > 65), LogicalOp)

    negated = ~(active.eq(True))
    assert isinstance(negated, UnaryOp)
    assert negated.op == "NOT "
    assert isinstance(not_(active.eq(True)), UnaryOp)

    single = and_(age >= 20)
    assert isinstance(single, BinaryOp)


def test_aggregates_and_raw_sql() -> None:
    age = Column[int]("users", "age")

    assert isinstance(count(), FuncCall)
    assert count().name == "COUNT"
    assert count().args == ()
    assert count(age, distinct=True).distinct is True
    assert sum_(age).name == "SUM"
    assert avg(age).name == "AVG"
    assert min_(age).name == "MIN"
    assert max_(age).name == "MAX"

    raw = raw_sql("age > $1", 10)
    assert isinstance(raw, RawSQL)
    assert raw.sql == "age > $1"
    assert raw.params == (10,)
