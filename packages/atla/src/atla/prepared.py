"""Precompiled SQL templates with named bind slots (``prepare()``)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import starmap
from typing import Any

from atla.adapter.base import DBAdapter

RowMapper = Callable[[Sequence[Any]], Any]
RowsMapper = Callable[[Sequence[Sequence[Any]]], list[Any]]


def rows_mapper_from_record[RowT](mapper: Callable[[Sequence[Any]], RowT]) -> RowsMapper:
    """Hydrate a row batch with a per-row mapper (generated ``_ROW_MAPPER``)."""

    def hydrate(rows: Sequence[Sequence[Any]]) -> list[RowT]:
        return [mapper(row) for row in rows]

    return hydrate


def rows_mapper_from_ctor[RowT](ctor: Callable[..., RowT]) -> RowsMapper:
    """Hydrate codec-free rows with ``starmap`` over the record constructor."""

    def hydrate(rows: Sequence[Sequence[Any]]) -> list[RowT]:
        return list(starmap(ctor, rows))

    return hydrate


class CompiledQuery[RowT]:
    """A SQL template compiled once and executed repeatedly with new parameters.

    Query builders expose ``prepare()``, which runs the normal compiler exactly
    once: SQL text, fixed bind values (for example ``LIMIT``), and the ordered
    :class:`~atla.expressions.Param` slots are all resolved up front. Every
    execution afterwards only assembles the parameter tuple — no AST nodes, no
    ``SQLCompiler`` instance, and no SQL string concatenation.

    Values are bound by name (``**kwargs``) or positionally (``*args``, following
    the slot's first appearance in the template). One name may fill several
    ``Param`` occurrences. Bind values are dialect-encoded at bind time, so a
    prepared template may be executed with ``str``, ``UUID``, ``Decimal``,
    ``datetime``, or JSON values without recompiling.

    ``first()`` issues the template unchanged (``fetch_one``); it does not add a
    ``LIMIT``. Bake ``.limit(1)`` into the query before ``prepare()`` when the
    lookup must fetch exactly one row. There is intentionally no ``one()`` here:
    the compiler cannot count rows without materializing them, so use
    ``.limit(2)`` plus ``first()`` when uniqueness matters.
    """

    __slots__ = (
        "_adapter",
        "_fixed",
        "_occurrences",
        "_row_mapper",
        "_rows_mapper",
        "_scalar_mapper",
        "_slot_names",
        "_sql",
    )

    def __init__(
        self,
        adapter: DBAdapter,
        *,
        sql: str,
        params: Sequence[Any],
        slot_occurrences: Sequence[tuple[str, int]],
        row_mapper: RowMapper | None = None,
        rows_mapper: RowsMapper | None = None,
        scalar_mapper: Callable[[Any], Any] | None = None,
    ) -> None:
        self._adapter = adapter
        self._sql = sql
        self._fixed = list(params)
        self._occurrences = tuple(slot_occurrences)
        # Distinct slot names in placeholder order; duplicates share one value.
        self._slot_names: tuple[str, ...] = tuple(
            dict.fromkeys(name for name, _ in self._occurrences)
        )
        self._row_mapper = row_mapper
        self._rows_mapper = rows_mapper
        # Set only for single-column projections; decodes the fetched scalar the
        # same way the row mapper decodes that column.
        self._scalar_mapper = scalar_mapper

    @property
    def sql(self) -> str:
        """The compiled SQL text, reused by every execution."""
        return self._sql

    @property
    def param_names(self) -> tuple[str, ...]:
        """Names of the ``Param`` slots to bind, in placeholder order."""
        return self._slot_names

    def _bound_params(self, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> list[Any]:
        names = self._slot_names
        if len(args) > len(names):
            msg = f"expected at most {len(names)} positional parameter(s), got {len(args)}"
            raise TypeError(msg)
        values: dict[str, Any] = dict(zip(names, args, strict=False))
        for name, value in kwargs.items():
            if name not in names:
                msg = f"unknown parameter {name!r}; expected one of {names!r}"
                raise TypeError(msg)
            if name in values:
                msg = f"parameter {name!r} given twice"
                raise TypeError(msg)
            values[name] = value
        if len(values) != len(names):
            missing = ", ".join(repr(name) for name in names if name not in values)
            msg = f"missing parameter(s): {missing}"
            raise TypeError(msg)
        bound = list(self._fixed)
        encode = self._adapter.dialect.encode_param
        for name, index in self._occurrences:
            bound[index] = encode(values[name])
        return bound

    def bind(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        """Resolve ``Param`` slots into a driver-ready positional parameter tuple.

        Useful when the caller executes the compiled SQL itself, and when
        measuring template reuse: binding never recompiles SQL.
        """
        return tuple(self._bound_params(args, kwargs))

    async def all(self, *args: Any, **kwargs: Any) -> list[RowT]:
        """Fetch every matching row and hydrate them into ``RowT`` values."""
        rows_mapper = self._rows_mapper
        if rows_mapper is None:
            msg = "this compiled query does not return rows; use execute()"
            raise ValueError(msg)
        rows = await self._adapter.fetch_all(self._sql, self._bound_params(args, kwargs))
        return rows_mapper(rows)

    async def first(self, *args: Any, **kwargs: Any) -> RowT | None:
        """Fetch the first row of the template (no implicit ``LIMIT``), or ``None``."""
        row_mapper = self._row_mapper
        if row_mapper is None:
            msg = "this compiled query does not return rows; use execute()"
            raise ValueError(msg)
        row = await self._adapter.fetch_one(self._sql, self._bound_params(args, kwargs))
        return None if row is None else row_mapper(row)

    async def fetch_val(self, *args: Any, **kwargs: Any) -> Any:
        """Fetch the first column of the first row (single-column projections only)."""
        scalar_mapper = self._scalar_mapper
        if scalar_mapper is None:
            msg = "fetch_val() requires a single-column projection; use all() or first()"
            raise ValueError(msg)
        value = await self._adapter.fetch_val(self._sql, self._bound_params(args, kwargs))
        return scalar_mapper(value)

    async def execute(self, *args: Any, **kwargs: Any) -> int:
        """Run a statement that returns no rows and report the affected row count."""
        if self._row_mapper is not None:
            msg = "this compiled query returns rows; use all() or first()"
            raise ValueError(msg)
        return await self._adapter.execute(self._sql, self._bound_params(args, kwargs))
