"""Tests for atlagen.emitter.models codegen details."""

from __future__ import annotations

from atlagen.emitter.models import emit_models
from atlagen.ir import ColumnIR, SchemaIR, TableIR


def _col(
    name: str,
    *,
    raw_type: str = "text",
    nullable: bool = False,
    schema_nullable: bool | None = None,
    is_primary_key: bool = False,
    row_codec: str | None = None,
) -> ColumnIR:
    return ColumnIR(
        name=name,
        attr_name=name,
        raw_type=raw_type,
        python_type="str | None" if nullable else "str",
        base_python_type="str",
        import_symbol=None,
        nullable=nullable,
        schema_nullable=schema_nullable if schema_nullable is not None else nullable,
        has_default=False,
        is_primary_key=is_primary_key,
        row_codec=row_codec,
    )


def test_single_column_row_codec_uses_tuple_unpack() -> None:
    schema = SchemaIR(
        name="public",
        dialect="sqlite",
        tables=(
            TableIR(
                name="flags",
                model_name="Flag",
                accessor_name="flag",
                columns=(_col("payload", row_codec="json.loads({value})"),),
                primary_key=("payload",),
                foreign_keys=(),
            ),
        ),
    )
    source = emit_models(schema)
    assert "(c0,) = row" in source
    assert "c0 = row" not in source


def test_schema_nullable_pk_omits_pk_indexes() -> None:
    schema = SchemaIR(
        name="public",
        dialect="sqlite",
        tables=(
            TableIR(
                name="items",
                model_name="Item",
                accessor_name="item",
                columns=(
                    _col(
                        "id",
                        raw_type="text",
                        nullable=False,
                        schema_nullable=True,
                        is_primary_key=True,
                    ),
                    _col("name"),
                ),
                primary_key=("id",),
                foreign_keys=(),
            ),
        ),
    )
    source = emit_models(schema)
    assert "_PK_INDEXES = ()" in source


def test_non_null_single_pk_emits_indexes() -> None:
    schema = SchemaIR(
        name="public",
        dialect="postgres",
        tables=(
            TableIR(
                name="users",
                model_name="User",
                accessor_name="user",
                columns=(
                    _col("id", raw_type="bigint", schema_nullable=False, is_primary_key=True),
                    _col("email"),
                ),
                primary_key=("id",),
                foreign_keys=(),
            ),
        ),
    )
    source = emit_models(schema)
    assert "_PK_INDEXES = (0,)" in source


def test_array_row_codec_nullability() -> None:
    schema = SchemaIR(
        name="public",
        dialect="postgres",
        tables=(
            TableIR(
                name="users",
                model_name="User",
                accessor_name="user",
                columns=(
                    _col("id", raw_type="bigint", is_primary_key=True),
                    _col(
                        "roles",
                        raw_type="role[]",
                        nullable=False,
                        row_codec="None if {value} is None else [(Role(item)) for item in {value}]",
                    ),
                    _col(
                        "opt_roles",
                        raw_type="role[]",
                        nullable=True,
                        row_codec="None if {value} is None else [(Role(item)) for item in {value}]",
                    ),
                ),
                primary_key=("id",),
                foreign_keys=(),
            ),
        ),
    )
    source = emit_models(schema)
    assert "([(Role(item)) for item in c1])" in source
    assert "None if c1 is None" not in source
    assert "None if c2 is None" in source

