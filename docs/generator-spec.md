# `atlagen` コードジェネレータ詳細仕様 (Generator Specification)

本ドキュメントでは、Atlas のスキーマ定義から型安全かつゼロオーバーヘッドな Python コードを事前生成する CLI ツール `atlagen` の仕様を定義します。

---

## 1. CLI インターフェース & 設定仕様 (`atlagen/cli.py`, `atlagen/config.py`)

### 1.1 コマンドライン構文

`atlagen` は以下の3通りの入力方式をサポートします：
1. **標準入力パイプライン (推奨)**: `atlas schema inspect ... --format '{{ json . }}' | atlagen generate ...`
2. **保存済み Atlas JSON ファイル**: `atlagen generate --schema ./schema.json ...`
3. **Atlas CLI 自動呼び出し**: `atlagen generate --schema file://schema.hcl --dev-url docker://postgres/16/dev ...`

```bash
# 1. 標準入力から Atlas JSON を受け取る場合 (--schema 省略時は stdin を読み取る)
atlas schema inspect -u "postgres://localhost:5432/mydb" --format '{{ json . }}' \
  | atlagen generate --dialect postgres --out ./app/db/generated

# 2. Atlas JSON ファイルを直接指定する場合
atlagen generate \
  --schema ./schema.inspect.json \
  --dialect mysql \
  --out ./app/db/generated

# 3. schema.hcl と dev-url を指定して内部で atlas schema inspect を実行する場合
atlagen generate \
  --schema "file://schema.hcl" \
  --dev-url "docker://postgres/16/dev" \
  --dialect postgres \
  --out ./app/db/generated

# 4. CI 向け差分検証モード (生成コードが最新スキーマと一致しているか検査)
atlagen check --schema ./schema.inspect.json --dialect postgres --out ./app/db/generated
```

### 1.2 CLI オプション一覧

| オプション | 必須 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `--dialect` | Yes* | - | ターゲット DB 方言 (`postgres`, `cockroach`, `mysql`, `tidb`, `sqlite`) |
| `--out`, `-o` | Yes* | - | 生成コードの出力先ディレクトリパス |
| `--schema`, `-s` | No | `-` (stdin) | Atlas JSON ファイルパス、または Atlas URL (`file://...`, `postgres://...`) |
| `--dev-url` | No | `None` | `--schema` に HCL/SQL を指定した際、Atlas CLI に渡す `--dev-url` |
| `--schema-name` | No | 自動 (先頭スキーマ) | マルチスキーマ環境で対象とする DB スキーマ名（例: `public`） |
| `--frozen / --mutable` | No | `--frozen` | 生成する `{Table}Record` dataclass に `frozen=True` を付与するか |

*※ `pyproject.toml` の `[tool.atlagen]` セクションが定義されている場合は CLI 引数を省略可能。*

### 1.3 `pyproject.toml` 設定サポート

```toml
[tool.atlagen]
schema = "file://db/schema.hcl"
dev_url = "docker://postgres/16/dev"
dialect = "postgres"
out = "src/app/db/generated"
frozen = true
```

---

## 2. 内部中間表現 (Internal Representation: `atlagen/ir.py`)

Atlas の JSON 出力フォーマットは RDBMS ごとに細かな表現差があるため、`parser.py` で一度正規化された内部 AST (`SchemaIR`) に変換します。

```python
from dataclasses import dataclass, field

@dataclass(slots=True, frozen=True)
class EnumIR:
    name: str                  # DB 上の Enum 型名 (例: "user_role")
    class_name: str            # Python の Enum クラス名 (例: "UserRole")
    values: tuple[str, ...]    # 列挙値タプル (例: ("admin", "member", "guest"))

@dataclass(slots=True, frozen=True)
class ColumnIR:
    name: str                  # DB カラム名 (例: "created_at")
    attr_name: str             # Python 属性名 (予約語衝突時は末尾 `_` 付与)
    raw_type: str              # Atlas JSON 上の型文字列 (例: "character varying(255)")
    python_type: str           # 読み取り時の Python 型表現 (例: "str", "int | None")
    base_python_type: str      # Optional を除いた基底型 (例: "str", "int")
    import_symbol: tuple[str, str] | None  # 必要な import (例: ("datetime", "datetime"))
    nullable: bool             # NULL 許容か
    has_default: bool          # DEFAULT 値または AUTO_INCREMENT / IDENTITY があるか
    is_primary_key: bool       # 主キー構成カラムか
    row_codec: str | None      # 行デコード時の変換式テンプレート (不要なら None)

@dataclass(slots=True, frozen=True)
class ForeignKeyIR:
    symbol: str                # FK 制約名
    columns: tuple[str, ...]   # 自テーブルのカラム名タプル
    ref_table: str             # 参照先テーブル名
    ref_columns: tuple[str, ...]# 参照先カラム名タプル

@dataclass(slots=True, frozen=True)
class TableIR:
    name: str                  # DB テーブル名 (例: "users")
    model_name: str            # PascalCase 単数形プレフィックス (例: "User")
    accessor_name: str         # AtlaClient 上のプロパティ名 (例: "user")
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

## 3. RDBMS 別 型マッピング規則 (`atlagen/mapper.py`)

SQL 型文字列はサイズ指定 (`varchar(255)`), 精度指定 (`numeric(10, 2)`), 符号指定 (`bigint unsigned`), タイムゾーン表記 (`timestamp with time zone`) などを正規化した上で以下の規則で Python 型へマッピングします。

| DB 分類 | SQL 型 (正規化後) | Python 型 (`models.py`) | `Column[T]` 型定義 | 必要な標準ライブラリ Import |
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
| | `json`, `jsonb` | `dict[str, Any] \| list[Any]` | `Column[Any]` | `from typing import Any` |
| | `<type>[]` (配列型) | `list[<PythonType>]` | `Column[list[<PythonType>]]` | - |
| | `<enum>` (ユーザー定義 Enum) | `<EnumClass>` | `Column[<EnumClass>]` | `from .enums import <EnumClass>` |
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

### Nullability & Default 規則
1. **Nullability (`null: true`)**:
   - `ColumnIR.nullable == True` の場合、Python 3.10+ 標準の Union 記法 **`T | None`** として出力する（例: `Column[str | None]`, `bio: str | None`）。
2. **Insert 時の必須 / 省略可能判定**:
   - `nullable == False` かつ `has_default == False`（`auto_increment` / `identity` / `default` 式がいずれも無い）カラムのみ、`insert()` の **必須キーワード引数 (`T`)** とする。
   - `nullable == True` または `has_default == True` のカラムは、デフォルト値 `UNSET` を持つ **省略可能キーワード引数 (`T | UnsetType = UNSET`)** とする。これにより、指定されなかったカラムは `INSERT` 文のカラムリスト自体から除外され、DB 側の `DEFAULT` 値が正しく適用される。

---

## 4. 生成コード仕様 (`emitter/`)

`atlagen` は出力ディレクトリに以下の5ファイルを生成します。すべてのファイル先頭には `# Code generated by atlagen. DO NOT EDIT.` ヘッダが付与されます。

```text
generated/
├── __init__.py    # AtlaClient, 各 Record, 各 Table シングルトン, 各 Enum の再エクスポート
├── enums.py       # DB Enum に対応する StrEnum クラス群
├── models.py      # {Table}Record (slots dataclass) と {Table}Table (Column 定義)
├── query.py       # 型付き {Table}Query / Insert / Update / Delete ビルダ
└── client.py      # AtlaClient (テーブルアクセサ & トランザクション制御)
```

### 4.1 `models.py` の出力例

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
        """自己結合やサブクエリ用にテーブルエイリアスを付与した新しい UsersTable を返す。"""
        return UsersTable(alias=alias)


User = UsersTable()
```

### 4.2 `query.py` の出力例 (AOT 事前計算 & 完全型付き Builder)

```python
# Code generated by atlagen. DO NOT EDIT.
from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence
from atla.adapter import DBAdapter
from atla.expressions import Expr, OrderByClause
from atla.query import BaseSelectQuery, BaseInsertQuery, BaseUpdateQuery, BaseDeleteQuery
from atla.types import UNSET, UnsetType
from .models import UserRecord, UsersTable, User

class UserQuery(BaseSelectQuery[UserRecord]):
    __slots__ = ()
    # コード生成時に確定している SELECT 句を定数化（実行時の文字列結合コストゼロ）
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

### 4.3 `client.py` の出力例

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
        """トランザクションを開始し、同一コネクションに束縛された AtlaClient を返す。"""
        async with self._adapter.transaction() as tx_adapter:
            yield AtlaClient(tx_adapter)
```
