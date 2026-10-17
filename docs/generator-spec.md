# `atlagen` Code Generator Specification

This document specifies `atlagen`, the Ahead-of-Time (AOT) CLI code generator that reads Atlas schema definitions and emits zero-overhead, strictly typed Python models and query builders.

---

## 1. CLI Interface & Configuration (`atlagen/cli.py`, `atlagen/config.py`)

### 1.1 Command-Line Syntax

`atlagen` supports three input modes:
1. **Standard Input Pipeline (Recommended)**: `atlas schema inspect ... --format '{{ json . }}' | atlagen generate ...`
2. **Saved Atlas JSON File**: `atlagen generate --schema ./schema.json ...`
3. **Direct Atlas CLI Invocation**: `atlagen generate --schema file://schema.hcl --dev-url docker://postgres/16/dev ...`

```bash
# 1. Read Atlas JSON from stdin (default when --schema is omitted or '-')
atlas schema inspect -u "postgres://localhost:5432/mydb" --format '{{ json . }}' \
  | atlagen generate --dialect postgres --out ./app/db/generated

# 2. Read from a saved Atlas JSON file
atlagen generate \
  --schema ./schema.inspect.json \
  --dialect mysql \
  --out ./app/db/generated

# 3. Invoke `atlas schema inspect` automatically under the hood
atlagen generate \
  --schema "file://schema.hcl" \
  --dev-url "docker://postgres/16/dev" \
  --dialect postgres \
  --out ./app/db/generated

# 4. CI check mode (verifies that generated files match the current schema without modifying disk)
atlagen check --schema ./schema.inspect.json --dialect postgres --out ./app/db/generated
```

### 1.2 CLI Options

| Option | Required | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dialect` | Yes* | - | Target DB dialect (`postgres`, `cockroach`, `mysql`, `tidb`, `sqlite`). |
| `--out`, `-o` | Yes* | - | Output directory path for generated Python files. |
| `--schema`, `-s` | No | `-` (stdin) | Path to an Atlas JSON file, `-` for stdin, or an Atlas URL (`file://...`, `postgres://...`). |
| `--dev-url` | No | `None` | Passed to `atlas schema inspect --dev-url` when inspecting HCL or SQL files directly. |
| `--schema-name` | No | Auto (first schema) | Target database schema name in multi-schema databases (e.g., `public`). |
| `--frozen / --mutable` | No | `--frozen` | Whether generated `{Table}Record` dataclasses specify `frozen=True`. |

*\*Can be omitted on the command line if configured in `pyproject.toml` under `[tool.atlagen]`.*

### 1.3 `pyproject.toml` Configuration Support

```toml
[tool.atlagen]
schema = "file://db/schema.hcl"
dev_url = "docker://postgres/16/dev"
dialect = "postgres"
out = "src/app/db/generated"
frozen = true
```

---

## 2. Internal Representation (`atlagen/ir.py`)

Because Atlas JSON output exhibits subtle structural variations across database engines, `parser.py` normalizes the raw JSON into a dialect-agnostic Internal Representation (`SchemaIR`) before code generation:

```python
from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class EnumIR:
    name: str                  # Database enum name (e.g., "user_role")
    class_name: str            # Generated Python StrEnum class name (e.g., "UserRole")
    values: tuple[str, ...]    # Enum values (e.g., ("admin", "member", "guest"))

@dataclass(slots=True, frozen=True)
class ColumnIR:
    name: str                  # Database column name (e.g., "created_at")
    attr_name: str             # Python attribute name (appends `_` if colliding with Python keywords)
    raw_type: str              # Raw SQL type string from Atlas JSON (e.g., "character varying(255)")
    python_type: str           # Read-time Python type annotation (e.g., "str", "int | None")
    base_python_type: str      # Non-nullable base Python type (e.g., "str", "int")
    import_symbol: tuple[str, str] | None  # Required import (e.g., ("datetime", "datetime"))
    nullable: bool             # Whether the column allows NULL
    has_default: bool          # True if column has DEFAULT, AUTO_INCREMENT, SERIAL, or IDENTITY
    is_primary_key: bool       # True if part of the table's primary key
    row_codec: str | None      # Inline decoding expression for `_from_row` (None if driver returns native type)

@dataclass(slots=True, frozen=True)
class ForeignKeyIR:
    symbol: str                # Foreign key constraint name
    columns: tuple[str, ...]   # Local column names
    ref_table: str             # Referenced table name
    ref_columns: tuple[str, ...]# Referenced column names

@dataclass(slots=True, frozen=True)
class TableIR:
    name: str                  # Database table name (e.g., "users")
    model_name: str            # Singular PascalCase prefix (e.g., "User")
    accessor_name: str         # Property name on AtlaClient (e.g., "user")
    columns: tuple[ColumnIR, ...]
    primary_key: tuple[str, ...]
    foreign_keys: tuple[ForeignKeyIR, ...]

@dataclass(slots=True, frozen=True)
class SchemaIR:
    name: str
    dialect: str
    enums: tuple[EnumIR, ...] = ()
    tables: tuple[TableIR, ...] = ()
```

---

## 3. SQL-to-Python Type Mapping Rules (`atlagen/mapper.py`)

Raw SQL type strings are normalized (stripping length specifiers like `varchar(255)`, precision specifiers like `numeric(10, 2)`, `unsigned` modifiers, and timezone clauses) and mapped according to the following rules:

| Engine Family | Normalized SQL Type | Python Type (`models.py`) | `Column[T]` Definition | Required Stdlib Import |
| :--- | :--- | :--- | :--- | :--- |
| **PostgreSQL / CockroachDB** | `smallint`, `integer`, `bigint`, `serial`, `bigserial`, `int2`, `int4`, `int8`, `rowid` | `int` | `Column[int]` | - |
| | `real`, `double precision`, `float4`, `float8` | `float` | `Column[float]` | - |
| | `numeric`, `decimal`, `money` | `Decimal` | `Column[Decimal]` | `from decimal import Decimal` |
| | `boolean`, `bool` | `bool` | `Column[bool]` | - |
| | `varchar`, `character varying`, `char`, `text`, `citext`, `inet`, `cidr` | `str` | `Column[str]` | - |
| | `bytea`, `bytes` | `bytes` | `Column[bytes]` | - |
| | `uuid` | `UUID` | `Column[UUID]` | `from uuid import UUID` |
| | `date` | `date` | `Column[date]` | `from datetime import date` |
| | `time`, `timetz` | `time` | `Column[time]` | `from datetime import time` |
| | `timestamp`, `timestamptz`, `timestamp with time zone` | `datetime` | `Column[datetime]` | `from datetime import datetime` |
| | `interval` | `timedelta` | `Column[timedelta]` | `from datetime import timedelta` |
| | `json`, `jsonb` | `Any` (`dict` / `list`) | `Column[Any]` | `from typing import Any` |
| | `<type>[]` (Array types) | `list[<PythonType>]` | `Column[list[<PythonType>]]` | - |
| | `<enum>` (User-defined Enum) | `<EnumClass>` | `Column[<EnumClass>]` | `from .enums import <EnumClass>` |
| **MySQL / TiDB** | `tinyint(1)`, `boolean`, `bool` | `bool` | `Column[bool]` | - |
| | `tinyint`, `smallint`, `mediumint`, `int`, `integer`, `bigint` (signed/unsigned) | `int` | `Column[int]` | - |
| | `float`, `double` | `float` | `Column[float]` | - |
| | `decimal`, `numeric` | `Decimal` | `Column[Decimal]` | `from decimal import Decimal` |
| | `char`, `varchar`, `tinytext`, `text`, `mediumtext`, `longtext` | `str` | `Column[str]` | - |
| | `binary`, `varbinary`, `tinyblob`, `blob`, `mediumblob`, `longblob` | `bytes` | `Column[bytes]` | - |
| | `date` | `date` | `Column[date]` | `from datetime import date` |
| | `time` | `time` | `Column[time]` | `from datetime import time` |
| | `datetime`, `timestamp` | `datetime` | `Column[datetime]` | `from datetime import datetime` |
| | `json` | `Any` (`dict` / `list`) | `Column[Any]` | `from typing import Any` |
| | `enum('a', 'b', ...)` | `<Table><Col>Enum` (`StrEnum`) | `Column[<Table><Col>Enum]` | `from .enums import ...` |
| **SQLite** | `integer`, `int`, `tinyint`, `bigint` | `int` | `Column[int]` | - |
| | `real`, `float`, `double` | `float` | `Column[float]` | - |
| | `numeric`, `decimal` | `Decimal` | `Column[Decimal]` | `from decimal import Decimal` |
| | `boolean`, `bool` | `bool` | `Column[bool]` | - |
| | `text`, `varchar`, `char`, `clob` | `str` | `Column[str]` | - |
| | `blob` | `bytes` | `Column[bytes]` | - |
| | `datetime`, `timestamp` | `datetime` | `Column[datetime]` | `from datetime import datetime` |
| | `date` | `date` | `Column[date]` | `from datetime import date` |
| | `uuid` | `UUID` | `Column[UUID]` | `from uuid import UUID` |
| | `json` | `Any` | `Column[Any]` | `from typing import Any` |

### Nullability & Default Rules
1. **Nullability (`null: true`)**:
   - When `ColumnIR.nullable == True`, the type annotation uses standard Python 3.10+ union syntax **`T | None`** (e.g., `Column[str | None]`, `bio: str | None`).
2. **Required vs. Optional Parameters in `insert()`**:
   - Columns where `nullable == False` AND `has_default == False` (no server `DEFAULT`, `AUTO_INCREMENT`, or `IDENTITY`) are emitted as **required keyword-only arguments (`col: T`)** on `insert()`.
   - Columns where `nullable == True` OR `has_default == True` are emitted as **optional keyword-only arguments (`col: T | UnsetType = UNSET`)**. Any argument left as `UNSET` is omitted from the `INSERT` column list so the database applies its native default value.

---

## 4. Emitted Code Specification (`emitter/`)

`atlagen` writes five files into the target output directory. Every generated file starts with `# Code generated by atlagen. DO NOT EDIT.`.

```text
generated/
├── __init__.py    # Re-exports AtlaClient, Record models, Table singletons, and Enums
├── enums.py       # StrEnum classes corresponding to database enums
├── models.py      # {Table}Record (slots dataclass) and {Table}Table (Column descriptors)
├── query.py       # Typed {Table}Query / Insert / Update / Delete builders
└── client.py      # AtlaClient (table repositories & transaction context manager)
```

### 4.1 Example `models.py` Output

```python
# Code generated by atlagen. DO NOT EDIT.
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence
from atla.expressions import Column

@dataclass(slots=True, frozen=True)
class UserRecord:
    id: int
    email: str
    name: str
    age: int
    is_active: bool
    deleted_at: datetime | None
    created_at: datetime

    @classmethod
    def _from_row(cls, row: Sequence[Any]) -> UserRecord:
        return cls(*row)


class UsersTable:
    __slots__ = ("_alias", "id", "email", "name", "age", "is_active", "deleted_at", "created_at")
    __table_name__ = "users"

    def __init__(self, alias: str | None = None) -> None:
        self._alias = alias
        t = alias or "users"
        self.id: Column[int] = Column(t, "id")
        self.email: Column[str] = Column(t, "email")
        self.name: Column[str] = Column(t, "name")
        self.age: Column[int] = Column(t, "age")
        self.is_active: Column[bool] = Column(t, "is_active")
        self.deleted_at: Column[datetime | None] = Column(t, "deleted_at")
        self.created_at: Column[datetime] = Column(t, "created_at")

    def as_(self, alias: str) -> UsersTable:
        """Return a new UsersTable instance bound to the given SQL table alias for self-joins."""
        return UsersTable(alias=alias)


User = UsersTable()
```

### 4.2 Example `query.py` Output (AOT Pre-computed SQL & Typed Builders)

```python
# Code generated by atlagen. DO NOT EDIT.
from __future__ import annotations

from datetime import datetime
from atla.adapter import DBAdapter
from atla.expressions import Expr
from atla.query import BaseSelectQuery, BaseInsertQuery, BaseUpdateQuery, BaseDeleteQuery
from atla.types import UNSET, UnsetType
from .models import UserRecord

class UserQuery(BaseSelectQuery[UserRecord]):
    __slots__ = ()
    # Static SELECT clause pre-computed at codegen time (zero runtime string-join overhead)
    _TABLE_NAME = '"users"'
    _SELECT_SQL = (
        'SELECT "users"."id", "users"."email", "users"."name", '
        '"users"."age", "users"."is_active", "users"."deleted_at", "users"."created_at" '
        'FROM "users"'
    )
    _COUNT_SQL = 'SELECT COUNT(*) FROM "users"'
    _ROW_MAPPER = UserRecord._from_row


class UserInsertBuilder(BaseInsertQuery[UserRecord]):
    __slots__ = ()
    _TABLE_NAME = '"users"'
    _RETURNING_COLS = (
        '"id", "email", "name", "age", "is_active", "deleted_at", "created_at"'
    )
    _ROW_MAPPER = UserRecord._from_row


class UserUpdateBuilder(BaseUpdateQuery[UserRecord]):
    __slots__ = ()
    _TABLE_NAME = '"users"'
    _RETURNING_COLS = (
        '"id", "email", "name", "age", "is_active", "deleted_at", "created_at"'
    )
    _ROW_MAPPER = UserRecord._from_row

    def set(
        self,
        *,
        email: str | Expr[str] | UnsetType = UNSET,
        name: str | Expr[str] | UnsetType = UNSET,
        age: int | Expr[int] | UnsetType = UNSET,
        is_active: bool | Expr[bool] | UnsetType = UNSET,
        deleted_at: datetime | None | Expr[datetime | None] | UnsetType = UNSET,
    ) -> UserUpdateBuilder:
        return self._set_values(
            email=email,
            name=name,
            age=age,
            is_active=is_active,
            deleted_at=deleted_at,
        )


class UserRepository:
    __slots__ = ("_adapter",)

    def __init__(self, adapter: DBAdapter) -> None:
        self._adapter = adapter

    def query(self) -> UserQuery:
        return UserQuery(self._adapter)

    def where(self, *predicates: Expr[bool]) -> UserQuery:
        return UserQuery(self._adapter).where(*predicates)

    def insert(
        self,
        *,
        email: str,
        name: str,
        age: int,
        id: int | UnsetType = UNSET,
        is_active: bool | UnsetType = UNSET,
        deleted_at: datetime | None | UnsetType = UNSET,
        created_at: datetime | UnsetType = UNSET,
    ) -> UserInsertBuilder:
        return UserInsertBuilder(self._adapter).values(
            id=id,
            email=email,
            name=name,
            age=age,
            is_active=is_active,
            deleted_at=deleted_at,
            created_at=created_at,
        )

    def update(self) -> UserUpdateBuilder:
        return UserUpdateBuilder(self._adapter)

    def delete(self) -> BaseDeleteQuery:
        return BaseDeleteQuery(self._adapter, table_name='"users"')
```

### 4.3 Example `client.py` Output

```python
# Code generated by atlagen. DO NOT EDIT.
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator
from atla.adapter import DBAdapter
from .query import UserRepository, PostRepository

class AtlaClient:
    __slots__ = ("_adapter", "user", "post")

    def __init__(self, adapter: DBAdapter) -> None:
        self._adapter = adapter
        self.user = UserRepository(adapter)
        self.post = PostRepository(adapter)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[AtlaClient]:
        """Begin a database transaction and yield an AtlaClient pinned to the transaction connection."""
        async with self._adapter.transaction() as tx_adapter:
            yield AtlaClient(tx_adapter)
```
