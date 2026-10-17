# `atla` ランタイムコア詳細仕様 (Runtime Specification)

本ドキュメントでは、外部依存ゼロのランタイムパッケージ `atla` のモジュール設計、式木 (Expression Tree) AST、単一パス SQL コンパイラ、方言 (Dialect)、および DB アダプタのプロトコル仕様を定義します。

---

## 1. 設計原則

1. **外部依存ゼロ (`dependencies = []`)**: Python 3.11+ 標準ライブラリのみに依存。
2. **全クラス `__slots__` 必須**: 動的辞書 (`__dict__`) のアロケーションを排除し、属性アクセス速度とメモリ効率を最大化。
3. **不変性 (Immutability)**: すべての式ノードとクエリビルダはイミュータブルであり、再利用しても副作用を起こさない。
4. **完全な型パラメータ伝播**: 式木は評価結果の型 `T` を保持する `Expr[T]` として定義され、Pyright / Mypy の strict モードで型推論が途切れない。

---

## 2. 式木 (Expression Tree) 仕様 (`atla/expressions.py`)

### 2.1 基底クラス `Expr[T]`

すべての SQL 式ノードは `Expr[T_co]` を継承します。`T_co` はその式が SQL 上で評価された際の Python 型（例: 条件式なら `bool`、数値演算なら `int` / `float`）を表します。

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

    # 論理演算子 (Expr[bool] 同士の結合)
    def __and__(self: Expr[bool], other: Expr[bool]) -> LogicalOp:
        return LogicalOp("AND", (self, other))

    def __or__(self: Expr[bool], other: Expr[bool]) -> LogicalOp:
        return LogicalOp("OR", (self, other))

    def __invert__(self: Expr[bool]) -> UnaryOp[bool]:
        return UnaryOp("NOT ", self)
```

> [!IMPORTANT]
> **`__bool__` による安全性保証**
> Python の `and` / `or` キーワードや `is None` はオーバーロードできません。もしユーザーが誤って `.where(User.age >= 20 and User.is_active == True)` と記述した場合、`Expr.__bool__` が即座に明快な `TypeError` を送出するため、片方の条件が暗黙に無視される重大なバグを未然に防ぎます。

### 2.2 カラム表現 `Column[T]`

テーブルの各カラムは `Column[T]` インスタンスとして定義されます。

```python
class Column(Expr[T]):
    __slots__ = ("table", "name", "alias")

    def __init__(self, table: str, name: str, alias: str | None = None) -> None:
        self.table = table
        self.name = name
        self.alias = alias

    # --- 比較演算子 (Operator Overloads) ---
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

    # --- 明示的比較メソッド (Linter フレンドリー & 型安全) ---
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

    # --- NULL 判定 ---
    def is_null(self) -> UnaryOp[bool]:
        return UnaryOp("IS NULL", self, postfix=True)

    def is_not_null(self) -> UnaryOp[bool]:
        return UnaryOp("IS NOT NULL", self, postfix=True)

    # --- コレクション・範囲・文字列マッチ ---
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

    # --- 算術演算 (UPDATE 句の `User.age + 1` 等で利用) ---
    def __add__(self, other: T | Expr[T]) -> BinaryOp[T]:
        return BinaryOp("+", self, _coerce_literal(other))

    def __sub__(self, other: T | Expr[T]) -> BinaryOp[T]:
        return BinaryOp("-", self, _coerce_literal(other))

    def __mul__(self, other: T | Expr[T]) -> BinaryOp[T]:
        return BinaryOp("*", self, _coerce_literal(other))

    # --- ソート指定 ---
    def asc(self, *, nulls: str | None = None) -> OrderByClause:
        return OrderByClause(self, direction="ASC", nulls=nulls)

    def desc(self, *, nulls: str | None = None) -> OrderByClause:
        return OrderByClause(self, direction="DESC", nulls=nulls)
```

### 2.3 AST ノード一覧

| クラス名 | 継承元 | `__slots__` | 役割・SQL 出力例 |
| :--- | :--- | :--- | :--- |
| `Literal[T]` | `Expr[T]` | `("value",)` | バインドパラメータ値 (`$1`, `%s`, `?`) を保持 |
| `Column[T]` | `Expr[T]` | `("table", "name", "alias")` | `"users"."age"` などのカラム識別子 |
| `BinaryOp[T]` | `Expr[T]` | `("op", "left", "right")` | 二項演算 (`"users"."age" >= $1`, `"users"."age" + $1`) |
| `UnaryOp[T]` | `Expr[T]` | `("op", "operand", "postfix")` | 単項演算 (`NOT (...)`, `"users"."deleted_at" IS NULL`) |
| `LogicalOp` | `Expr[bool]` | `("op", "clauses")` | 論理結合 (`(A AND B AND C)`, `(A OR B)`)。同一演算子のネストはフラット化 |
| `InOp[T]` | `Expr[bool]` | `("expr", "values", "negated")` | `IN ($1, $2)`。**`values` が空タプルの場合は SQL 構文エラーを防ぐため `1 = 0` (`negated=False`) または `1 = 1` (`negated=True`) に安全に展開** |
| `BetweenOp[T]` | `Expr[bool]` | `("expr", "low", "high")` | `"users"."age" BETWEEN $1 AND $2` |
| `FuncCall[T]` | `Expr[T]` | `("name", "args", "distinct")` | `COUNT(*)`, `MAX("users"."age")`, `COALESCE(...)` |
| `RawSQL[T]` | `Expr[T]` | `("sql", "params")` | エスケープハッチ `raw_sql[bool]("...", params)` |
| `OrderByClause`| - | `("expr", "direction", "nulls")` | `ORDER BY "users"."created_at" DESC NULLS LAST` |

### 2.4 論理・集約ヘルパー関数 (`atla/__init__.py`)

- **論理結合**:
  - `and_(*clauses: Expr[bool]) -> Expr[bool]`
  - `or_(*clauses: Expr[bool]) -> Expr[bool]`
  - `not_(clause: Expr[bool]) -> Expr[bool]`
- **集約関数**:
  - `count(expr: Expr[Any] | None = None, *, distinct: bool = False) -> FuncCall[int]`
  - `sum_(expr: Expr[T]) -> FuncCall[T | None]`
  - `avg(expr: Expr[Any]) -> FuncCall[float | None]`
  - `min_(expr: Expr[T]) -> FuncCall[T | None]`
  - `max_(expr: Expr[T]) -> FuncCall[T | None]`
- **Raw SQL エスケープハッチ**:
  - `raw_sql(sql: str, *params: Any) -> RawSQL[Any]`

---

## 3. Dialect & 単一パス SQL コンパイラ (`atla/dialect.py`, `atla/compiler.py`)

### 3.1 Dialect 仕様

各 RDBMS の構文差異を吸収するため、軽量な `Dialect` クラスを定義します。

| Dialect クラス | 対象 DB | 識別子クォート | プレースホルダ | `RETURNING` 対応 | `ILIKE` 変換 | Upsert 構文 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `PostgresDialect` | PostgreSQL | `"ident"` | `$1, $2, ...` | Yes | ネイティブ `ILIKE` | `ON CONFLICT (...) DO UPDATE` |
| `CockroachDialect`| CockroachDB | `"ident"` | `$1, $2, ...` | Yes | ネイティブ `ILIKE` | `ON CONFLICT (...) DO UPDATE` |
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

### 3.2 単一パス SQL コンパイラ (`SQLCompiler`)

式木 AST から SQL 文字列とバインドパラメータ配列を生成する際、文字列連結 (`+`) による中間文字列アロケーションを避け、`list[str]` バッファへの `append` と最後の `"".join(buf)` で単一パス (Single-pass) コンパイルを行います。

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
        # match / isinstance 分岐により高速に buf と params を構築
        ...

    def build(self) -> tuple[str, list[Any]]:
        return "".join(self.buf), self.params
```

---

## 4. DB アダプタ & トランザクション仕様 (`atla/adapter/`)

### 4.1 `DBAdapter` プロトコル定義

```python
from contextlib import AbstractAsyncContextManager
from typing import Any, Protocol, Sequence

class DBAdapter(Protocol):
    @property
    def dialect(self) -> Dialect: ...

    async def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Sequence[Any]]:
        """複数行をタプル/Record のシーケンスとして取得する。"""
        ...

    async def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Any] | None:
        """単一行をタプル/Record として取得する。存在しない場合は None。"""
        ...

    async def fetch_val(self, sql: str, params: Sequence[Any] = ()) -> Any:
        """単一行の第1カラム値を取得する（COUNT や RETURNING id 用）。"""
        ...

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        """DML (INSERT/UPDATE/DELETE) を実行し、影響行数 (または lastrowid) を返す。"""
        ...

    async def execute_many(self, sql: str, params_seq: Sequence[Sequence[Any]]) -> None:
        """バッチ DML を実行する。"""
        ...

    def transaction(self) -> AbstractAsyncContextManager[DBAdapter]:
        """トランザクションを開始し、同一コネクションに束縛された DBAdapter を yield する。"""
        ...
```

### 4.2 各ドライバアダプタの実装詳細

#### 1. `AsyncpgAdapter` (PostgreSQL / CockroachDB)
- `asyncpg.Pool` または `asyncpg.Connection` をラップ。
- `fetch_all`: `await conn.fetch(sql, *params)` を呼び出す。`asyncpg.Record` は C レベルでシーケンスプロトコルと `*record` アンパックをサポートするため、変換コストゼロで `RecordCls(*row)` に渡せる。
- `execute`: `await conn.execute(sql, *params)` のステータスタグ（例: `"UPDATE 3"`）から影響行数を高速パースして `int` を返す。
- `transaction()`:
  - Pool の場合は `async with self._pool.acquire() as conn:` + `async with conn.transaction():` を実行し、その `conn` をラップした `_AsyncpgConnAdapter(conn)` を yield する。
  - 既にトランザクション内の Connection の場合はネストされた `conn.transaction()` (SAVEPOINT) を開始する。

#### 2. `AsyncmyAdapter` (MySQL / TiDB)
- `asyncmy.Pool` または `asyncmy.Connection` をラップ。
- 辞書生成コストを避けるため、標準のタプルカーソル (`asyncmy.cursors.Cursor`) を使用。
- `transaction()`: `await conn.begin()` 〜 `await conn.commit()` / `await conn.rollback()`（ネスト時は `SAVEPOINT`）を制御。

#### 3. `AiosqliteAdapter` (SQLite)
- `aiosqlite.Connection` をラップ。
- `PRAGMA foreign_keys = ON` や `SAVEPOINT` を用いたトランザクション分離をサポート。

---

## 5. 例外階層 (`atla/errors.py`)

```python
class AtlaError(Exception):
    """atla の基底例外クラス"""

class NotFoundError(AtlaError):
    """`.one()` 実行時に結果が 0 件だった場合に送出"""

class MultipleResultsError(AtlaError):
    """`.one()` 実行時に結果が 2 件以上存在した場合に送出"""

class UnsafeMutationError(AtlaError):
    """WHERE 句なしの UPDATE / DELETE が `.allow_full_table()` なしで実行された場合に送出"""
```

> [!TIP]
> **`UnsafeMutationError` による事故防止**
> 本番環境での `db.user.delete().execute()` のような WHERE 句忘れによる全件削除・全件更新事故を防ぐため、デフォルトで `where()` が未指定の `update()` / `delete()` は `UnsafeMutationError` を送出します。意図的に全件対象とする場合は `.allow_full_table()` の明示呼び出しを要求します。
