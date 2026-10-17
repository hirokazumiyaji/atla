"""atlagen: AOT code generator CLI for atla."""

from atlagen.ir import (
    ColumnIR,
    EnumIR,
    ForeignKeyIR,
    IndexIR,
    SchemaIR,
    TableIR,
)
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
    "__version__",
    "parse_atlas_json",
]
