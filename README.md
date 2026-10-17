# atla & atlagen

**Zero-overhead, compile-time type-safe SQL toolkit & code generator for Python, powered by [Atlas](https://atlasgo.io/).**

`atla` and `atlagen` form a next-generation database toolkit for Python that treats **Atlas** as the Single Source of Truth for schema management and migrations, delivering **native C-extension driver speed (`asyncpg`, `asyncmy`, `aiosqlite`, etc.)** alongside **complete compile-time type safety with Pyright and Mypy**.

---

## Core Philosophy

1. **Single Source of Truth (Atlas First)**
   - Schema definitions (`schema.hcl` / `.sql`) and database migrations are delegated entirely to **Atlas**.
   - Never maintain duplicate model definitions or migration histories in Python code.
2. **Zero Runtime Overhead**
   - `atla` is an ultra-lightweight, **pure-Python runtime with zero third-party dependencies**—no SQLAlchemy, no Pydantic, and no hidden reflection.
   - Static SQL fragments (such as `SELECT` column lists) and dialect-specific row decoders are pre-computed Ahead-of-Time (AOT) during code generation (`atlagen`). At runtime, row tuples returned by DB drivers are unpacked directly into `@dataclass(slots=True, frozen=True)` records (`Record(*row)`).
3. **Compile-time Type Safety**
   - Inspired by Drizzle ORM and Ent, `atla` uses generic expression trees (`Column[T]`, `Expr[T]`) paired with generated fluent query builders.
   - Column references, comparison operators, required vs. optional `INSERT` / `UPDATE` arguments, and return types for `JOIN` and partial `SELECT` queries are 100% statically inferred and verified under Pyright and Mypy strict mode.
4. **Multi-RDBMS & NewSQL Ready**
   - First-class native support for **PostgreSQL**, **CockroachDB**, **MySQL**, **TiDB**, and **SQLite**, abstracting away dialect differences in identifier quoting, parameter placeholders, `UPSERT` syntax, and `RETURNING` clauses.

---

## System Architecture

```mermaid
flowchart TD
    Schema["Atlas Schema (schema.hcl / schema.sql / migrations)"]
    Inspect["Atlas CLI (atlas schema inspect --format '{{ json . }}')"]
    JSON["Atlas Schema JSON"]

    subgraph atlagen ["atlagen (AOT Code Generator CLI)"]
        Parser["parser.py: JSON normalization & IR construction"]
        Mapper["mapper.py: Dialect-aware SQL type -> Python type mapping"]
        Emitter["emitter.py: Static SQL constants, models & query builders"]
        Parser --> Mapper --> Emitter
    end

    subgraph generated ["generated/ (Checked-in Application Code)"]
        Enums["enums.py: StrEnum definitions"]
        Models["models.py: slots dataclass Records & Table descriptors"]
        Query["query.py: Typed Query / Mutation Builders"]
        Client["client.py: AtlaClient & Transaction management"]
    end

    subgraph atla ["atla (Runtime Core: Zero External Dependencies)"]
        Expr["expressions.py: Expr[T], Column[T], logical/comparison/aggregate ops"]
        Compiler["compiler.py & dialect.py: Single-pass SQL compiler"]
        Adapter["adapter.py: DBAdapter Protocol (asyncpg / asyncmy / aiosqlite)"]
    end

    Schema --> Inspect --> JSON --> Parser
    Emitter --> Enums & Models & Query & Client
    Query --> Expr & Compiler
    Client --> Adapter
```

---

## Developer Experience

### 1. Generate Code (`atlagen`)

```bash
# Inspect your schema with Atlas CLI and pipe the JSON output into atlagen
atlas schema inspect \
  -u "file://schema.hcl" \
  --dev-url "docker://postgres/16/dev" \
  --format '{{ json . }}' \
  | atlagen generate --dialect postgres --out ./app/db/generated
```

### 2. Query & Mutate with Full Type Safety (`atla`)

```python
import asyncpg
from atla.adapter import AsyncpgAdapter
from app.db.generated import AtlaClient, Post, User

async def main() -> None:
    pool = await asyncpg.create_pool("postgresql://user:pass@localhost:5432/app")
    db = AtlaClient(AsyncpgAdapter(pool))

    # 1. Type-safe SELECT query with full IDE autocompletion
    # User.age >= 20 and User.is_active.eq(True) are statically verified as Expr[bool]
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
        # `user` is inferred as `UserRecord` (@dataclass(slots=True, frozen=True))
        print(user.id, user.name, user.age)

    # 2. Type-safe INSERT distinguishing required vs. database-defaulted columns
    # Columns with server defaults (e.g., id, created_at) are optional; omitting a required column is a static type error
    new_user = await (
        db.user.insert(
            email="alice@example.com",
            name="Alice",
            age=28,
        )
        .returning()
        .one()
    )

    # 3. Transactions & atomic expression updates
    async with db.transaction() as tx:
        await (
            tx.user.update()
            .set(age=User.age + 1)
            .where(User.id == new_user.id)
            .execute()
        )

    # 4. Type-safe JOINs (inferred return type: list[tuple[UserRecord, PostRecord]])
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

## Documentation

For detailed specifications and engineering plans, see:

- **[System Architecture & Design Principles (`docs/architecture.md`)](docs/architecture.md)**
  - Core design pillars, architectural refinements over the initial proposal, monorepo layout, AOT optimization strategy, and Python type system integration.
- **[`atla` Runtime Core Specification (`docs/runtime-spec.md`)](docs/runtime-spec.md)**
  - Expression Tree (`Expr[T]`, `Column[T]`), single-pass SQL compiler, dialect specifications, `DBAdapter` protocol, and transaction management.
- **[`atlagen` Code Generator Specification (`docs/generator-spec.md`)](docs/generator-spec.md)**
  - CLI specification, Atlas JSON parser & internal IR (`SchemaIR`), SQL-to-Python type mapping rules across dialects, and emitted code layout (`models.py`, `query.py`, `client.py`).
- **[Implementation Roadmap (`docs/roadmap.md`)](docs/roadmap.md)**
  - Phased milestones (Phase 0 to Phase 5), tracking issues, verification strategy, and performance benchmarks.