"""Dialect-agnostic intermediate representation for Atlas schemas."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class EnumIR:
    """Normalized enum type (Postgres enum or MySQL inline enum)."""

    name: str
    class_name: str
    values: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class ColumnIR:
    """Normalized table column."""

    name: str
    attr_name: str
    raw_type: str
    python_type: str
    base_python_type: str
    import_symbol: tuple[str, str] | None
    nullable: bool
    has_default: bool
    is_primary_key: bool
    row_codec: str | None


@dataclass(slots=True, frozen=True)
class ForeignKeyIR:
    """Normalized foreign key constraint."""

    symbol: str
    columns: tuple[str, ...]
    ref_table: str
    ref_columns: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class IndexIR:
    """Normalized index definition."""

    name: str
    columns: tuple[str, ...]
    unique: bool = False


@dataclass(slots=True, frozen=True)
class TableIR:
    """Normalized table."""

    name: str
    model_name: str
    accessor_name: str
    columns: tuple[ColumnIR, ...]
    primary_key: tuple[str, ...]
    foreign_keys: tuple[ForeignKeyIR, ...]
    indexes: tuple[IndexIR, ...] = ()


@dataclass(slots=True, frozen=True)
class SchemaIR:
    """Normalized schema ready for code generation."""

    name: str
    dialect: str
    enums: tuple[EnumIR, ...] = ()
    tables: tuple[TableIR, ...] = ()
