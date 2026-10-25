"""Unit tests for Issue #61: atlagen composite PK, enums, collisions, and config."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

from atla.dialect import SQLiteDialect
from atlagen.config import ConfigError, resolve_config
from atlagen.emitter import generate_files
from atlagen.emitter.enums import _safe_member, emit_enums  # pyright: ignore[reportPrivateUsage]
from atlagen.emitter.query import (
    _inbound_relations,  # pyright: ignore[reportPrivateUsage]
    _outbound_non_pk_fk_comments,  # pyright: ignore[reportPrivateUsage]
    _outbound_relations,  # pyright: ignore[reportPrivateUsage]
)
from atlagen.ir import ColumnIR, EnumIR, ForeignKeyIR, SchemaIR, TableIR
from atlagen.mapper import map_schema, resolve_sql_type
from atlagen.parser import AtlasParseError, _assert_no_symbol_collisions, _validate_override_ident  # pyright: ignore[reportPrivateUsage]


def _col(name: str, *, pk: bool = False, py: str = "int", raw: str = "integer") -> ColumnIR:
    return ColumnIR(
        name=name,
        attr_name=name,
        raw_type=raw,
        python_type=py,
        base_python_type=py,
        import_symbol=None,
        nullable=False,
        schema_nullable=False,
        has_default=False,
        is_primary_key=pk,
        row_codec=None,
    )


def test_composite_pk_does_not_generate_eager_loading() -> None:
    # Table A with composite PK (org_id, id)
    parent_table = TableIR(
        name="tenants",
        model_name="Tenant",
        accessor_name="tenant",
        columns=(
            _col("org_id", pk=True),
            _col("id", pk=True),
            _col("name", py="str", raw="text"),
        ),
        primary_key=("org_id", "id"),
        foreign_keys=(),
    )

    # Table B references parent_table.org_id
    child_table = TableIR(
        name="members",
        model_name="Member",
        accessor_name="member",
        columns=(
            _col("id", pk=True),
            _col("tenant_org_id"),
        ),
        primary_key=("id",),
        foreign_keys=(
            ForeignKeyIR(
                symbol="fk_members_tenant",
                columns=("tenant_org_id",),
                ref_table="tenants",
                ref_columns=("org_id",),
            ),
        ),
    )

    schema = SchemaIR(name="test", dialect="postgres", tables=(parent_table, child_table))

    # 1. Inbound relations for parent_table: must be empty because parent has composite PK
    inbound = _inbound_relations(parent_table, schema)
    assert inbound == []

    # 2. Outbound relations for child_table: must be empty because ref_col is not single PK
    outbound = _outbound_relations(child_table, schema)
    assert outbound == []

    # 3. Comments emitted for child_table: must note FK to non-PK column
    comments = _outbound_non_pk_fk_comments(child_table, schema)
    assert len(comments) == 1
    assert "FK to non-PK column" in comments[0]


def test_table_overrides_accessor_name_used_in_with_methods() -> None:
    user_table = TableIR(
        name="users",
        model_name="User",
        accessor_name="user",
        columns=(_col("id", pk=True),),
        primary_key=("id",),
        foreign_keys=(),
    )
    # Child table with customized accessor_name "articles" instead of default "post"
    post_table = TableIR(
        name="posts",
        model_name="Post",
        accessor_name="articles",
        columns=(_col("id", pk=True), _col("user_id")),
        primary_key=("id",),
        foreign_keys=(
            ForeignKeyIR(
                symbol="fk_posts_user",
                columns=("user_id",),
                ref_table="users",
                ref_columns=("id",),
            ),
        ),
    )
    schema = SchemaIR(name="test", dialect="postgres", tables=(user_table, post_table))
    files = generate_files(schema)
    query_code = files["query.py"]
    # Must emit with_articles based on accessor_name override
    assert "def with_articles(" in query_code


def test_safe_member_and_enum_emission_avoids_sunder_names() -> None:
    # Test values with symbols, sunder patterns, spaces, leading numbers
    test_values = ["(a)", "[draft]", "_internal_", "-x-", " a ", "123", "___"]

    members = [_safe_member(v) for v in test_values]
    for member in members:
        # Must not be sunder (_..._) or dunder (__...__)
        assert not (member.startswith("_") and member.endswith("_"))
        assert member.isidentifier()

    enum = EnumIR(
        name="status",
        class_name="StatusEnum",
        values=tuple(test_values),
    )
    schema = SchemaIR(name="test", dialect="postgres", tables=(), enums=(enum,))
    code = emit_enums(schema)

    # Executing the emitted enum module must succeed without ValueError
    namespace: dict[str, Any] = {}
    exec(code, namespace)
    status_cls = namespace["StatusEnum"]
    # Check members exist and have correct values
    assert status_cls.INTERNAL.value == "_internal_"
    assert status_cls.DRAFT.value == "[draft]"
    assert status_cls.A.value == "(a)"


def test_enum_name_resolution_case_insensitive() -> None:
    enum = EnumIR(
        name="UserRole",
        class_name="UserRole",
        values=("admin", "member"),
    )
    schema = SchemaIR(name="test", dialect="postgres", tables=(), enums=(enum,))
    mapped_schema = map_schema(schema)
    assert mapped_schema is not None

    # Resolve using raw types with different casing and schema prefixes
    mapping1 = resolve_sql_type("UserRole", dialect="postgres", enums={"userrole": enum})
    assert mapping1.base_python_type == "UserRole"

    mapping2 = resolve_sql_type("userrole", dialect="postgres", enums={"userrole": enum})
    assert mapping2.base_python_type == "UserRole"

    mapping3 = resolve_sql_type(
        "public.UserRole",
        dialect="postgres",
        enums={"public.userrole": enum, "userrole": enum},
    )
    assert mapping3.base_python_type == "UserRole"


def test_reserved_model_names_and_collisions() -> None:
    # Reserved model names must be rejected
    reserved_names = [
        "TypedDict",
        "Required",
        "NotRequired",
        "dataclass",
        "UpsertQuery",
        "NotFoundError",
        "OrderByClause",
        "PreloadOneSelectQuery",
        "IsolationLevel",
        "TypeAlias",
        "Callable",
        "Awaitable",
    ]
    for name in reserved_names:
        with pytest.raises(AtlasParseError, match="collides with a generated export suffix"):
            _validate_override_ident(name, kind="model_name", table_name="test")

    with pytest.raises(AtlasParseError, match="collides with a generated export suffix"):
        _validate_override_ident("UserInsertParam", kind="model_name", table_name="test")

    # Symbol collision check for UpsertBuilder, DeleteBuilder, InsertParam
    t = TableIR(
        name="users",
        model_name="User",
        accessor_name="user",
        columns=(_col("id", pk=True),),
        primary_key=("id",),
        foreign_keys=(),
    )
    enum_colliding = EnumIR(
        name="UserInsertParam",
        class_name="UserInsertParam",
        values=("A",),
    )
    with pytest.raises(AtlasParseError, match="collides between"):
        _assert_no_symbol_collisions([t], [enum_colliding])

    enum_colliding2 = EnumIR(
        name="UserUpsertBuilder",
        class_name="UserUpsertBuilder",
        values=("A",),
    )
    with pytest.raises(AtlasParseError, match="collides between"):
        _assert_no_symbol_collisions([t], [enum_colliding2])


def test_generated_init_reexports_insert_param() -> None:
    t = TableIR(
        name="users",
        model_name="User",
        accessor_name="user",
        columns=(_col("id", pk=True),),
        primary_key=("id",),
        foreign_keys=(),
    )
    schema = SchemaIR(name="test", dialect="postgres", tables=(t,))
    files = generate_files(schema)
    init_code = files["__init__.py"]
    assert "UserInsertParam" in init_code
    assert "'UserInsertParam'" in init_code


@pytest.mark.asyncio
async def test_composite_pk_get_many_validates_tuple_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    t = TableIR(
        name="memberships",
        model_name="Membership",
        accessor_name="membership",
        columns=(
            _col("tenant_id", pk=True),
            _col("user_id", pk=True),
            _col("role", py="str", raw="text"),
        ),
        primary_key=("tenant_id", "user_id"),
        foreign_keys=(),
    )
    schema = SchemaIR(name="test", dialect="sqlite", tables=(t,))
    files = generate_files(schema)
    pkg = tmp_path / "gen_pkg_test"
    pkg.mkdir()
    for rel, text in files.items():
        (pkg / rel).write_text(text)
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])

    query_mod = importlib.import_module("gen_pkg_test.query")
    repo_cls = query_mod.MembershipRepository

    class DummyAdapter:
        def __init__(self) -> None:
            self.dialect = SQLiteDialect()

        async def fetch_all(self, sql: str, params: Any = ()) -> list[Any]:
            return []

    repo = repo_cls(DummyAdapter())

    # Passing a 1-tuple when 2 elements are expected must raise ValueError
    with pytest.raises(ValueError, match="expected key tuple of length 2, got 1"):
        await repo.get_many([(1,)])

    # Passing a 3-tuple when 2 elements are expected must raise ValueError
    with pytest.raises(ValueError, match="expected key tuple of length 2, got 3"):
        await repo.get_many([(1, 2, 3)])

    # Passing valid 2-tuples must succeed
    results = await repo.get_many([(1, 2), (3, 4)])
    assert results == []


def test_config_rejects_unknown_keys() -> None:
    args = argparse.Namespace(
        command="generate",
        dialect="postgres",
        out=None,
        schema=None,
        dev_url=None,
        schema_name=None,
        frozen=None,
        mutable=False,
    )

    # Valid config
    valid = resolve_config(
        args,
        pyproject_settings={"out": "appdb", "dev_url": "postgres://..."},
    )
    assert valid.dialect == "postgres"

    # Unknown hyphenated keys or typos in [tool.atlagen]
    with pytest.raises(ConfigError, match="unrecognized configuration key\\(s\\): 'dev-url'"):
        resolve_config(
            args,
            pyproject_settings={"out": "appdb", "dev-url": "postgres://..."},
        )

    with pytest.raises(ConfigError, match="unrecognized configuration key\\(s\\): 'schema-name'"):
        resolve_config(
            args,
            pyproject_settings={"out": "appdb", "schema-name": "public"},
        )
