"""Table descriptors used by generated models for joins and preloads."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, cast

from atla.dialect import Dialect


class TableRef[RecordT]:
    """Base class for generated ``*Table`` descriptors (``User``, ``Post``, …)."""

    __slots__ = ("_alias",)

    __table_name__: ClassVar[str]
    _COLUMNS: ClassVar[tuple[str, ...]]
    _ROW_MAPPER: ClassVar[Any]
    # Primary-key column indexes for O(1) LEFT JOIN null checks.
    # Empty tuple falls back to scanning the whole row slice (custom descriptors).
    _PK_INDEXES: ClassVar[tuple[int, ...]] = ()

    def __init__(self, alias: str | None = None) -> None:
        self._alias = alias

    def sql_alias(self) -> str:
        """Return the SQL correlation name (alias or unqualified table name)."""
        if self._alias is not None:
            return self._alias
        return self.__table_name__.rsplit(".", 1)[-1]

    def join_target(self, dialect: Dialect) -> str:
        """Return ``table`` or ``table AS alias`` for FROM / JOIN clauses."""
        table_sql = ".".join(dialect.quote(part) for part in self.__table_name__.split("."))
        if self._alias is None:
            return table_sql
        return f"{table_sql} AS {dialect.quote(self._alias)}"

    def select_list(self, dialect: Dialect) -> str:
        """Return dialect-quoted ``alias.col, …`` projection for this table."""
        qual = dialect.quote(self.sql_alias())
        return ", ".join(f"{qual}.{dialect.quote(name)}" for name in self._COLUMNS)

    @property
    def column_count(self) -> int:
        return len(self._COLUMNS)

    def get_row_mapper(self) -> Any:
        return type(self)._ROW_MAPPER

    def map_row(self, row: Sequence[Any]) -> RecordT:
        return cast(RecordT, self.get_row_mapper()(row))

    def map_row_optional(self, row: Sequence[Any]) -> RecordT | None:
        pk_indexes = type(self)._PK_INDEXES
        if pk_indexes:
            # LEFT JOIN absent rows are NULL across every PK column.
            if all(row[index] is None for index in pk_indexes):
                return None
        elif all(value is None for value in row):
            return None
        return self.map_row(row)
