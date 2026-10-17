# `atla` Runtime Core Specification

This document specifies the internal module design of the zero-dependency runtime package `atla`, covering the Expression Tree AST, the single-pass SQL compiler, database dialects, the `DBAdapter` protocol, transaction management, and safety exceptions.

---

## 1. Design Principles

1. **Zero External Dependencies (`dependencies = []`)**: Relies exclusively on the Python 3.12+ standard library.
2. **Mandatory `__slots__` on Every Class**: Eliminates per-instance `__dict__` allocation to maximize attribute access speed and minimize memory footprint.
3. **Immutability**: All expression nodes and query builders are immutable and safe to reuse or branch across queries without side effects.
4. **End-to-End Type Parameter Propagation**: Expression nodes carry their evaluated Python result type `T` via `Expr[T]`, preserving strict type inference under Pyright and Mypy.

---

## 2. Expression Tree Specification (`atla/expressions.py`)

### 2.1 Base Class `Expr[T]`

All SQL expression AST nodes inherit from `Expr[T_co]`, where `T_co` represents the Python type produced when the expression is evaluated in SQL (e.g., `bool` for predicates, `int` / `float` for numeric expressions).

```python
from __future__ import annotations
from typing import Any, Generic, NoReturn, Sequence, TypeVar

T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")

class Expr(Generic[T_co]):
    __slots__ = ()

    def __bool__(self) -> NoReturn:
        raise TypeError(
            "SQL expression cannot be evaluated as a Python boolean. "
            "Use 'and_(a, b)' / 'a & b' or 'or_(a, b)' / 'a | b' instead of 'and' / 'or', "
            "and '.is_null()' / '.is_not_null()' instead of 'is None'."
        )

    # Logical operators for combining Expr[bool] predicates
    def __and__(self: Expr[bool], other: Expr[bool]) -> LogicalOp:
        return LogicalOp("AND", (self, other))

    def __or__(self: Expr[bool], other: Expr[bool]) -> LogicalOp:
        return LogicalOp("OR", (self, other))

    def __invert__(self: Expr[bool]) -> UnaryOp[bool]:
        return UnaryOp("NOT ", self)
```

> [!IMPORTANT]
> **Preventing Silent Short-Circuit Bugs via `__bool__`**
> Python does not allow overloading the `and`, `or`, `not`, or `is` keywords. If a developer accidentally writes `.where(User.age >= 20 and User.is_active == True)`, `Expr.__bool__` immediately raises a clear `TypeError` instead of silently short-circuiting and dropping the first condition.

### 2.2 Column Descriptor `Column[T]`

Each table column is represented as a typed `Column[T]` instance:

```python
class Column(Expr[T]):
    __slots__ = ("table", "name", "alias")

    def __init__(self, table: str, name: str, alias: str | None = None) -> None:
        self.table = table
        self.name = name
        self.alias = alias

    # --- Comparison Operators (Operator Overloads) ---
    def __eq__(self, other: T | Expr[T]) -> BinaryOp[bool]:  # type: ignore[override]
        if other is None:
            return UnaryOp("IS NULL", self, postfix=True)  # type: ignore[return-value]
        return BinaryOp("=", self, _coerce_literal(other))

    def __ne__(self, other: T | Expr[T]) -> BinaryOp[bool]:  # type: ignore[override]
        if other is None:
            return UnaryOp("IS NOT NULL", self, postfix=True)  # type: ignore[return-value]
        return BinaryOp("<>", self, _coerce_literal(other))

    def __lt__(self, other: T | Expr[T]) -> BinaryOp[bool]:
        return BinaryOp("<", self, _coerce_literal(other))

    def __le__(self, other: T | Expr[T]) -> BinaryOp[bool]:
        return BinaryOp("<=", self, _coerce_literal(other))

    def __gt__(self, other: T | Expr[T]) -> BinaryOp[bool]:
        return BinaryOp(">", self, _coerce_literal(other))

    def __ge__(self, other: T | Expr[T]) -> BinaryOp[bool]:
        return BinaryOp(">=", self, _coerce_literal(other))

    # --- Explicit Comparison Methods (Linter-Friendly & Type-Safe) ---
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

    # --- NULL Predicates ---
    def is_null(self) -> UnaryOp[bool]:
        return UnaryOp("IS NULL", self, postfix=True)

    def is_not_null(self) -> UnaryOp[bool]:
        return UnaryOp("IS NOT NULL", self, postfix=True)

    # --- Collection, Range & Pattern Matching ---
    def in_(self, values: Sequence[T]) -> InOp[T]:
        return InOp(self, tuple(values), negated=False)

    def not_in(self, values: Sequence[T]) -> InOp[T]:
        return InOp(self, tuple(values), negated=True)

    def between(self, low: T | Expr[T], high: T | Expr[T]) -> BetweenOp[T]:
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

    # --- Arithmetic Operators (Used in UPDATE clauses like `User.age + 1`) ---
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
```

### 2.3 AST Node Reference

| Class | Base Class | `__slots__` | Responsibility & SQL Compilation Behavior |
| :--- | :--- | :--- | :--- |
| `Literal[T]` | `Expr[T]` | `("value",)` | Holds a bind parameter value (`$1`, `%s`, `?`). |
| `Column[T]` | `Expr[T]` | `("table", "name", "alias")` | Qualified column identifier (e.g., `"users"."age"`). |
| `BinaryOp[T]` | `Expr[T]` | `("op", "left", "right")` | Binary operation (`"users"."age" >= $1`, `"users"."age" + $1`). |
| `UnaryOp[T]` | `Expr[T]` | `("op", "operand", "postfix")` | Unary operation (`NOT (...)`, `"users"."deleted_at" IS NULL`). |
| `LogicalOp` | `Expr[bool]` | `("op", "clauses")` | Logical conjunction/disjunction (`(A AND B AND C)`, `(A OR B)`). Nested identical operators are flattened. |
| `InOp[T]` | `Expr[bool]` | `("expr", "values", "negated")` | `IN ($1, $2)`. **When `values` is empty, safely compiles to `1 = 0` (`negated=False`) or `1 = 1` (`negated=True`) to prevent SQL syntax errors.** |
| `BetweenOp[T]` | `Expr[bool]` | `("expr", "low", "high")` | `"users"."age" BETWEEN $1 AND $2`. |
| `FuncCall[T]` | `Expr[T]` | `("name", "args", "distinct")` | SQL function calls (`COUNT(*)`, `MAX("users"."age")`, `COALESCE(...)`). |
| `RawSQL[T]` | `Expr[T]` | `("sql", "params")` | Type-annotated raw SQL escape hatch `raw_sql[bool]("...", params)`. |
| `OrderByClause`| - | `("expr", "direction", "nulls")` | `ORDER BY "users"."created_at" DESC NULLS LAST`. |

### 2.4 Logical & Aggregate Helper Functions (`atla/__init__.py`)

- **Logical Combinators**:
  - `and_(*clauses: Expr[bool]) -> Expr[bool]`
  - `or_(*clauses: Expr[bool]) -> Expr[bool]`
  - `not_(clause: Expr[bool]) -> Expr[bool]`
- **Aggregate Functions**:
  - `count(expr: Expr[Any] | None = None, *, distinct: bool = False) -> FuncCall[int]`
  - `sum_(expr: Expr[T]) -> FuncCall[T | None]`
  - `avg(expr: Expr[Any]) -> FuncCall[float | None]`
  - `min_(expr: Expr[T]) -> FuncCall[T | None]`
  - `max_(expr: Expr[T]) -> FuncCall[T | None]`
- **Raw SQL Escape Hatch**:
  - `raw_sql(sql: str, *params: Any) -> RawSQL[Any]`

---

## 3. Dialects & Single-Pass SQL Compiler (`atla/dialect.py`, `atla/compiler.py`)

### 3.1 Dialect Matrix

Lightweight `Dialect` classes encapsulate RDBMS-specific SQL syntax rules:

| Dialect Class | Target Engine | Identifier Quote | Placeholders | `RETURNING` Support | `ILIKE` Compilation | Upsert Syntax |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `PostgresDialect` | PostgreSQL | `"ident"` | `$1, $2, ...` | Yes | Native `ILIKE` | `ON CONFLICT (...) DO UPDATE` |
| `CockroachDialect`| CockroachDB | `"ident"` | `$1, $2, ...` | Yes | Native `ILIKE` | `ON CONFLICT (...) DO UPDATE` |
| `MySQLDialect` | MySQL | `` `ident` `` | `%s, %s, ...` | No (`lastrowid`) | `LOWER(a) LIKE LOWER(b)` | `ON DUPLICATE KEY UPDATE` |
| `TiDBDialect` | TiDB | `` `ident` `` | `%s, %s, ...` | No (`lastrowid`) | `LOWER(a) LIKE LOWER(b)` | `ON DUPLICATE KEY UPDATE` |
| `SQLiteDialect` | SQLite 3.35+ | `"ident"` | `?, ?, ...` | Yes | `LOWER(a) LIKE LOWER(b)` | `ON CONFLICT (...) DO UPDATE` |

```python
class Dialect:
    __slots__ = ()
    name: str
    quote_char: str
    supports_returning: bool
    supports_ilike: bool
    positional_style: str  # "dollar" ($1) | "format" (%s) | "qmark" (?)

    def quote(self, ident: str) -> str: ...
    def placeholder(self, index: int) -> str: ...
```

### 3.2 Single-Pass SQL Compiler (`SQLCompiler`)

To avoid intermediate string allocations from repeated `+` concatenations, `SQLCompiler` traverses the AST in a single pass, appending SQL fragments to a `list[str]` buffer and joining once via `"".join(buf)`.

```python
class SQLCompiler:
    __slots__ = ("dialect", "buf", "params")

    def __init__(self, dialect: Dialect, initial_sql: str = "") -> None:
        self.dialect = dialect
        self.buf: list[str] = [initial_sql] if initial_sql else []
        self.params: list[Any] = []

    def add_param(self, value: Any) -> str:
        self.params.append(value)
        return self.dialect.placeholder(len(self.params))

    def visit(self, node: Expr[Any]) -> None:
        # Fast structural pattern matching / isinstance dispatch over AST nodes
        ...

    def build(self) -> tuple[str, list[Any]]:
        return "".join(self.buf), self.params
```

---

## 4. Database Adapter & Transaction Specification (`atla/adapter/`)

### 4.1 `DBAdapter` Protocol

```python
from contextlib import AbstractAsyncContextManager
from typing import Any, Protocol, Sequence

class DBAdapter(Protocol):
    @property
    def dialect(self) -> Dialect: ...

    async def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Sequence[Any]]:
        """Fetch multiple rows as a sequence of tuples/records."""
        ...

    async def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Any] | None:
        """Fetch a single row as a tuple/record, or None if no row matches."""
        ...

    async def fetch_val(self, sql: str, params: Sequence[Any] = ()) -> Any:
        """Fetch the first column of the first row (used for COUNT and scalar queries)."""
        ...

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        """Execute a DML statement (INSERT/UPDATE/DELETE) and return affected row count."""
        ...

    async def execute_many(self, sql: str, params_seq: Sequence[Sequence[Any]]) -> None:
        """Execute a batch DML statement."""
        ...

    def transaction(self) -> AbstractAsyncContextManager[DBAdapter]:
        """Begin a transaction and yield a DBAdapter pinned to the active connection."""
        ...
```

### 4.2 Concrete Driver Adapters

#### 1. `AsyncpgAdapter` (PostgreSQL / CockroachDB)
- Wraps `asyncpg.Pool` or `asyncpg.Connection`.
- `fetch_all`: Calls `await conn.fetch(sql, *params)`. Because `asyncpg.Record` natively implements the sequence protocol and C-level `*record` unpacking, rows are unpacked into `RecordCls(*row)` with zero conversion overhead.
- `execute`: Parses the affected row count from the command status tag (e.g., `"UPDATE 3"` -> `3`).
- `transaction()`:
  - When wrapping a `Pool`, acquires a connection via `async with self._pool.acquire() as conn:` and opens `async with conn.transaction():`, yielding a connection-pinned `AsyncpgAdapter(conn)`.
  - When already inside a transaction on a pinned connection, opens a nested `conn.transaction()` (`SAVEPOINT`).

#### 2. `AsyncmyAdapter` (MySQL / TiDB)
- Wraps `asyncmy.Pool` or `asyncmy.Connection`.
- Uses the standard tuple cursor (`asyncmy.cursors.Cursor`) rather than `DictCursor` to avoid per-row dictionary allocation.
- `transaction()`: Manages `await conn.begin()` / `await conn.commit()` / `await conn.rollback()` (and `SAVEPOINT` for nested transactions).

#### 3. `AiosqliteAdapter` (SQLite)
- Wraps `aiosqlite.Connection`.
- Supports `PRAGMA foreign_keys = ON` and `SAVEPOINT`-based transaction isolation.

---

## 5. Exception Hierarchy (`atla/errors.py`)

```python
class AtlaError(Exception):
    """Base exception for all atla runtime errors."""

class NotFoundError(AtlaError):
    """Raised when `.one()` returns 0 rows."""

class MultipleResultsError(AtlaError):
    """Raised when `.one()` returns more than 1 row."""

class UnsafeMutationError(AtlaError):
    """Raised when an UPDATE or DELETE query without a WHERE clause is executed without `.allow_full_table()`."""
```

> [!TIP]
> **Guarding Against Accidental Full-Table Mutations**
> To prevent catastrophic production bugs such as `await db.user.delete().execute()` wiping an entire table when `.where(...)` is forgotten, `update()` and `delete()` raise `UnsafeMutationError` by default if no `WHERE` clause is present. Full-table mutations require an explicit `.allow_full_table()` call.
