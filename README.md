# atla & atlagen

**Zero-overhead, compile-time type-safe SQL toolkit & code generator for Python, powered by [Atlas](https://atlasgo.io/).**

`atla` と `atlagen` は、スキーマ管理・マイグレーションのデファクトスタンダードである **Atlas** を Single Source of Truth とし、Python において **C 拡張ドライバ (`asyncpg`, `asyncmy`, `aiosqlite` 等) 本来の実行速度** と **Pyright / Mypy による完全な静的型安全性** を両立するための次世代データベースツールキットです。

---

## コア設計思想 (Core Philosophy)

1. **Single Source of Truth (Atlas First)**
   - スキーマ定義 (`schema.hcl` / `.sql`) とマイグレーション管理はすべて **Atlas** に一任します。
   - Python コード側で二重にモデル定義やマイグレーション履歴を管理する必要はありません。
2. **Zero Runtime Overhead**
   - SQLAlchemy や Pydantic などの重量級ランタイム依存を一切排除した **純粋 Python・依存ゼロ (`atla`)** の超軽量ランタイムです。
   - 静的な `SELECT` 句の文字列結合や型変換ロジックはコード生成時 (`atlagen`) に事前解決（AOT コンパイル）され、実行時は DB ドライバが返す行タプルを `@dataclass(slots=True, frozen=True)` へ直接アンパック (`Record(*row)`) します。
3. **Compile-time Type Safety**
   - Drizzle ORM / Ent にインスパイアされたジェネリクス式木 (`Column[T]`, `Expr[T]`) とコード生成された Fluent QueryBuilder を採用。
   - カラム参照、比較演算、`INSERT` / `UPDATE` の必須・任意引数、`JOIN` や部分選択 (`SELECT`) の戻り値型まで、Pyright / Mypy の strict モードで完全に型推論・静的検証されます。
4. **Multi-RDBMS & NewSQL Ready**
   - **PostgreSQL**, **CockroachDB**, **MySQL**, **TiDB**, **SQLite** の方言 (Dialect)・型システム・UPSERT 構文・RETURNING 句の差異を吸収し、ネイティブサポートします。

---

## システムアーキテクチャ (Architecture Overview)

```mermaid
flowchart TD
    Schema["Atlas Schema (schema.hcl / schema.sql / migrations)"]
    Inspect["Atlas CLI (atlas schema inspect --format '{{ json . }}')"]
    JSON["Atlas Schema JSON"]

    subgraph atlagen ["atlagen (AOT Code Generator CLI)"]
        Parser["parser.py: JSON 正規化 & 内部 IR 構築"]
        Mapper["mapper.py: Dialect 別 SQL 型 -> Python 型変換"]
        Emitter["emitter.py: 静的 SQL 定数・モデル・クエリ生成"]
        Parser --> Mapper --> Emitter
    end

    subgraph generated ["generated/ (プロジェクト内生成コード)"]
        Enums["enums.py: StrEnum 定義"]
        Models["models.py: slots dataclass Record & Table 式木"]
        Query["query.py: 型付き Query / Mutation Builder"]
        Client["client.py: AtlaClient & Transaction 管理"]
    end

    subgraph atla ["atla (Runtime Core: 外部依存ゼロ)"]
        Expr["expressions.py: Expr[T], Column[T], 論理/比較/集約演算"]
        Compiler["compiler.py & dialect.py: 単一パス SQL コンパイラ"]
        Adapter["adapter.py: DBAdapter Protocol (asyncpg / asyncmy / aiosqlite)"]
    end

    Schema --> Inspect --> JSON --> Parser
    Emitter --> Enums & Models & Query & Client
    Query --> Expr & Compiler
    Client --> Adapter
```

---

## Developer Experience (利用イメージ)

### 1. コード生成 (`atlagen`)

```bash
# Atlas CLI でスキーマを検査し、パイプラインで型安全な Python コードを生成
atlas schema inspect \
  -u "file://schema.hcl" \
  --dev-url "docker://postgres/16/dev" \
  --format '{{ json . }}' \
  | atlagen generate --dialect postgres --out ./app/db/generated
```

### 2. アプリケーションでのクエリ・更新 (`atla`)

```python
import asyncpg
from atla.adapter import AsyncpgAdapter
from app.db.generated import AtlaClient, Post, User

async def main() -> None:
    pool = await asyncpg.create_pool("postgresql://user:pass@localhost:5432/app")
    db = AtlaClient(AsyncpgAdapter(pool))

    # 1. 完全な型安全性と自動補完による SELECT クエリ
    # User.age >= 20 や User.is_active.eq(True) は Expr[bool] として静的検証される
    users = await (
        db.user.query()
        .where(
            User.age >= 20,
            User.is_active.eq(True),
            User.deleted_at.is_null(),
        )
        .order_by(User.created_at.desc())
        .limit(10)
        .all()
    )
    for user in users:
        # user は UserRecord (@dataclass(slots=True, frozen=True)) と推論される
        print(user.id, user.name, user.age)

    # 2. 必須カラム・デフォルト値付きカラムを区別する型安全な INSERT
    # id や created_at など DB 側デフォルト値があるカラムは省略可能、必須カラムは省略すると型エラー
    new_user = await (
        db.user.insert(
            email="alice@example.com",
            name="Alice",
            age=28,
        )
        .returning()
        .one()
    )

    # 3. トランザクションと式木を用いたアトミックな UPDATE
    async with db.transaction() as tx:
        await (
            tx.user.update()
            .set(age=User.age + 1)
            .where(User.id == new_user.id)
            .execute()
        )

    # 4. 型安全な JOIN (戻り値は list[tuple[UserRecord, PostRecord]])
    rows = await (
        db.user.query()
        .join(Post, Post.user_id == User.id)
        .where(Post.published.eq(True))
        .all()
    )
    for user, post in rows:
        print(f"{user.name}: {post.title}")
```

---

## ドキュメント構成 (Documentation)

詳細な設計仕様および実装ロードマップは以下のドキュメントを参照してください。

- **[システムアーキテクチャ & 設計原則 (`docs/architecture.md`)](docs/architecture.md)**
  - 設計思想の深掘り、初期企画からの改善点、モノレポ構成、AOT 最適化戦略、Python 型システムとの統合設計
- **[`atla` ランタイムコア仕様 (`docs/runtime-spec.md`)](docs/runtime-spec.md)**
  - 式木 (`Expr[T]`, `Column[T]`)、単一パス SQL コンパイラ、Dialect 仕様、DB アダプタ & トランザクション仕様
- **[`atlagen` コードジェネレータ仕様 (`docs/generator-spec.md`)](docs/generator-spec.md)**
  - CLI 仕様、Atlas JSON パーサ & 内部 IR、RDBMS 別型マッピング規則、生成コード (`models.py`, `query.py`, `client.py`) の詳細設計
- **[実装ロードマップ & タスク計画 (`docs/roadmap.md`)](docs/roadmap.md)**
  - フェーズ分割 (Phase 0 〜 Phase 5)、マイルストーン、品質基準・ベンチマーク計画