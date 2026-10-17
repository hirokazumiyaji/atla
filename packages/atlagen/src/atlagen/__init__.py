"""atlagen: AOT code generator CLI for atla."""

from atlagen.ir import (
    ColumnIR,
    EnumIR,
    ForeignKeyIR,
    IndexIR,
    SchemaIR,
    TableIR,
)
from atlagen.mapper import TypeMapping, map_column, map_schema, resolve_sql_type
from atlagen.parser import AtlasParseError, parse_atlas_json

__version__ = "0.1.0"

__all__ = [
    "AtlasParseError",
    "ColumnIR",
    "EnumIR",
    "ForeignKeyIR",
    "IndexIR",
    "SchemaIR",
    "TableIR",
    "TypeMapping",
    "__version__",
    "map_column",
    "map_schema",
    "parse_atlas_json",
    "resolve_sql_type",
]
