"""Unit tests for dialects and the single-pass SQLCompiler."""

from __future__ import annotations

import pytest

from atla import (
    CockroachDialect,
    Column,
    Dialect,
    MySQLDialect,
    PostgresDialect,
    SQLCompiler,
    SQLiteDialect,
    TiDBDialect,
    and_,
    compile_expr,
    compile_order_by,
    count,
    or_,
    raw_sql,
)

DIALECTS: list[Dialect] = [
    PostgresDialect(),
    CockroachDialect(),
    MySQLDialect(),
    TiDBDialect(),
    SQLiteDialect(),
]


def _ph(dialect: Dialect, index: int) -> str:
    return dialect.placeholder(index)


@pytest.mark.parametrize("dialect", DIALECTS, ids=lambda d: d.name)
def test_compound_predicate_and_params(dialect: Dialect) -> None:
    age = Column[int]("users", "age")
    active = Column[bool]("users", "is_active")
    expr = and_(age >= 20, active.eq(True), or_(age < 65, age.is_null()))
    sql, params = compile_expr(dialect, expr)

    q = dialect.quote
    age_sql = f"{q('users')}.{q('age')}"
    active_sql = f"{q('users')}.{q('is_active')}"
    p1, p2, p3 = _ph(dialect, 1), _ph(dialect, 2), _ph(dialect, 3)
    expected = (
        f"({age_sql} >= {p1} AND {active_sql} = {p2} AND ({age_sql} < {p3} OR {age_sql} IS NULL))"
    )
    assert sql == expected
    assert params == [20, True, 65]


@pytest.mark.parametrize("dialect", DIALECTS, ids=lambda d: d.name)
def test_in_between_null_arithmetic_order_by(dialect: Dialect) -> None:
    age = Column[int]("users", "age")
    name = Column[str]("users", "name")
    created = Column[object]("users", "created_at")
    q = dialect.quote
    age_sql = f"{q('users')}.{q('age')}"
    name_sql = f"{q('users')}.{q('name')}"

    in_sql, in_params = compile_expr(dialect, age.in_([1, 2]))
    assert in_sql == f"{age_sql} IN ({_ph(dialect, 1)}, {_ph(dialect, 2)})"
    assert in_params == [1, 2]

    empty_in_sql, empty_in_params = compile_expr(dialect, age.in_([]))
    assert empty_in_sql == "1 = 0"
    assert empty_in_params == []

    empty_not_in_sql, empty_not_in_params = compile_expr(dialect, age.not_in([]))
    assert empty_not_in_sql == "1 = 1"
    assert empty_not_in_params == []

    between_sql, between_params = compile_expr(dialect, age.between(18, 30))
    assert between_sql == (f"{age_sql} BETWEEN {_ph(dialect, 1)} AND {_ph(dialect, 2)}")
    assert between_params == [18, 30]

    null_sql, null_params = compile_expr(dialect, age.is_null())
    assert null_sql == f"{age_sql} IS NULL"
    assert null_params == []

    arith_sql, arith_params = compile_expr(dialect, age + 1)
    assert arith_sql == f"{age_sql} + {_ph(dialect, 1)}"
    assert arith_params == [1]

    if dialect.name in {"mysql", "tidb"}:
        with pytest.raises(ValueError, match="does not support NULLS"):
            compile_order_by(dialect, created.desc(nulls="LAST"))
        order_sql, order_params = compile_order_by(dialect, created.desc())
        assert order_sql == f"{q('users')}.{q('created_at')} DESC"
        assert order_params == []
    else:
        order_sql, order_params = compile_order_by(dialect, created.desc(nulls="LAST"))
        assert order_sql == f"{q('users')}.{q('created_at')} DESC NULLS LAST"
        assert order_params == []

    like_sql, like_params = compile_expr(dialect, name.ilike("%alice%"))
    if dialect.supports_ilike:
        assert like_sql == f"{name_sql} ILIKE {_ph(dialect, 1)}"
    else:
        assert like_sql == f"LOWER({name_sql}) LIKE LOWER({_ph(dialect, 1)})"
    assert like_params == ["%alice%"]


@pytest.mark.parametrize("dialect", DIALECTS, ids=lambda d: d.name)
def test_count_and_alias_column(dialect: Dialect) -> None:
    age = Column[int]("users", "age", alias="u")
    sql, params = compile_expr(dialect, count(age, distinct=True))
    q = dialect.quote
    assert sql == f"COUNT(DISTINCT {q('u')}.{q('age')})"
    assert params == []


def test_placeholder_styles() -> None:
    assert PostgresDialect().placeholder(1) == "$1"
    assert CockroachDialect().placeholder(2) == "$2"
    assert MySQLDialect().placeholder(1) == "%s"
    assert TiDBDialect().placeholder(3) == "%s"
    assert SQLiteDialect().placeholder(1) == "?"


def test_raw_sql_rebinding_on_postgres() -> None:
    age = Column[int]("users", "age")
    expr = and_(age == 1, raw_sql("x = $1", 2))
    sql, params = compile_expr(PostgresDialect(), expr)
    assert sql == '("users"."age" = $1 AND x = $2)'
    assert params == [1, 2]


def test_raw_sql_rebinding_on_sqlite_keeps_qmark() -> None:
    age = Column[int]("users", "age")
    expr = and_(age == 1, raw_sql("x = ?", 2))
    sql, params = compile_expr(SQLiteDialect(), expr)
    assert sql == '("users"."age" = ? AND x = ?)'
    assert params == [1, 2]


def test_raw_sql_placeholder_count_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="placeholder"):
        compile_expr(SQLiteDialect(), raw_sql("x = ? AND y = ?", 1))
    with pytest.raises(ValueError, match="placeholder"):
        compile_expr(MySQLDialect(), raw_sql("x = %s AND y = %s", 1))


def test_raw_sql_reuses_repeated_dollar_placeholders() -> None:
    sql, params = compile_expr(
        PostgresDialect(),
        and_(
            Column[int]("users", "age") == 1,
            raw_sql("x = $1 OR y = $1", 9),
        ),
    )
    assert sql == '("users"."age" = $1 AND x = $2 OR y = $2)'
    assert params == [1, 9]


def test_raw_sql_format_ignores_modulo_identifiers() -> None:
    sql, params = compile_expr(
        MySQLDialect(),
        raw_sql("score %score = %s", 3),
    )
    assert sql == "score %score = %s"
    assert params == [3]


def test_raw_sql_ignores_postgres_dollar_quoted_strings() -> None:
    sql, params = compile_expr(
        PostgresDialect(),
        and_(
            Column[int]("users", "age") == 1,
            raw_sql("body = $$keep $1$$ AND x = $1", 9),
        ),
    )
    assert sql == '("users"."age" = $1 AND body = $$keep $1$$ AND x = $2)'
    assert params == [1, 9]


def test_raw_sql_ignores_placeholders_inside_literals() -> None:
    sql, params = compile_expr(
        SQLiteDialect(),
        raw_sql("note = '?' AND id = ?", 7),
    )
    assert sql == "note = '?' AND id = ?"
    assert params == [7]

    pg_sql, pg_params = compile_expr(
        PostgresDialect(),
        and_(
            Column[int]("users", "age") == 1,
            raw_sql("note = '$1' AND x = $1", 2),
        ),
    )
    assert pg_sql == '("users"."age" = $1 AND note = \'$1\' AND x = $2)'
    assert pg_params == [1, 2]


def test_raw_sql_mysql_backslash_string_not_placeholder() -> None:
    sql, params = compile_expr(
        MySQLDialect(),
        raw_sql("note = 'it\\'s %s' AND id = %s", 7),
    )
    assert sql == "note = 'it\\'s %s' AND id = %s"
    assert params == [7]

    dbl_sql, dbl_params = compile_expr(
        MySQLDialect(),
        raw_sql('SELECT "a\\"b" , %s', 1),
    )
    assert dbl_sql == 'SELECT "a\\"b" , %s'
    assert dbl_params == [1]


def test_raw_sql_postgres_nested_block_comment() -> None:
    sql, params = compile_expr(
        PostgresDialect(),
        raw_sql("/* outer /* inner $1 */ still $1 */ x = $1", 9),
    )
    assert sql == "/* outer /* inner $1 */ still $1 */ x = $1"
    assert params == [9]


def test_mysql_nulls_rejection_does_not_mutate_compiler() -> None:
    created = Column[object]("users", "created_at")
    compiler = SQLCompiler(MySQLDialect())
    with pytest.raises(ValueError, match="does not support NULLS"):
        compiler.visit_order_by(created.desc(nulls="LAST"))
    assert compiler.build() == ("", [])
