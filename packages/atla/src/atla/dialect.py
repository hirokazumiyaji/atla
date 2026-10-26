"""RDBMS dialect abstractions for identifier quoting and placeholders."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from typing import Any, ClassVar, Literal, cast
from uuid import UUID

from atla.types import Json

_REQUIRED_DIALECT_ATTRS = (
    "name",
    "quote_char",
    "supports_returning",
    "supports_ilike",
    "positional_style",
    "supports_for_update",
    "supports_for_share",
    "supports_skip_locked",
    "supports_lock_of",
    "supports_distinct_on",
)


class Dialect:
    """Base dialect describing engine-specific SQL surface syntax.

    Concrete subclasses must define the class attributes listed in
    ``_REQUIRED_DIALECT_ATTRS``. Instantiating :class:`Dialect` directly is
    rejected so missing configuration surfaces early.
    """

    __slots__ = ()

    name: ClassVar[str]
    quote_char: ClassVar[str]
    supports_returning: ClassVar[bool]
    supports_ilike: ClassVar[bool]
    positional_style: ClassVar[str]  # "dollar" | "format" | "qmark"
    supports_for_update: ClassVar[bool]
    supports_for_share: ClassVar[bool]
    supports_skip_locked: ClassVar[bool]
    supports_lock_of: ClassVar[bool]
    supports_distinct_on: ClassVar[bool]
    upsert_style: ClassVar[Literal["conflict", "duplicate"]] = "conflict"
    # Driver natively accepts these Python types as bind parameters.
    native_uuid: ClassVar[bool] = False
    native_decimal: ClassVar[bool] = False
    native_datetime: ClassVar[bool] = False
    # Driver maps Python ``list`` to native SQL arrays (Postgres/Cockroach).
    # When False, lists are JSON-encoded. ``dict`` is always JSON-encoded.
    native_array: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        for attr in _REQUIRED_DIALECT_ATTRS:
            if not any(
                attr in base.__dict__
                for base in cls.__mro__
                if base is not Dialect and base is not object
            ):
                msg = f"{cls.__name__} must define dialect attribute {attr!r}"
                raise TypeError(msg)

    def __init__(self) -> None:
        if type(self) is Dialect:
            msg = "Dialect is abstract; instantiate a concrete dialect subclass"
            raise TypeError(msg)

    def quote(self, ident: str) -> str:
        if not ident:
            msg = "identifier must not be empty"
            raise ValueError(msg)
        if "\x00" in ident:
            msg = "identifier must not contain NUL bytes"
            raise ValueError(msg)
        q = self.quote_char
        escaped = ident.replace(q, q + q)
        return f"{q}{escaped}{q}"

    def quote_string_literal(self, value: str) -> str:
        """Quote a SQL string literal (e.g. ``ESCAPE`` character).

        MySQL/TiDB treat ``\\`` as an escape inside string literals by default,
        so a literal backslash must be emitted as ``\\\\``. Other dialects only
        need ``'`` doubled.
        """
        if self.name in {"mysql", "tidb"}:
            escaped = value.replace("\\", "\\\\").replace("'", "''")
        else:
            escaped = value.replace("'", "''")
        return f"'{escaped}'"

    def placeholder(self, index: int) -> str:
        if self.positional_style == "dollar":
            return f"${index}"
        if self.positional_style == "format":
            return "%s"
        if self.positional_style == "qmark":
            return "?"
        msg = f"unsupported positional_style: {self.positional_style!r}"
        raise ValueError(msg)

    def encode_param(self, value: Any) -> Any:
        """Normalize a bind parameter for this dialect's driver.

        Hot path: ``None`` and plain primitives are returned unchanged (no
        copies). Non-native types (``UUID``, ``Decimal``, ``datetime``, JSON
        containers, ``Enum``) are converted to driver-safe forms. ``list`` is
        passed through unchanged when ``native_array`` is set (Postgres/CockroachDB
        arrays) and JSON-encoded otherwise.

        Encoding is idempotent and runs in two places on purpose: the compiler
        encodes values as they are bound so compiled ``(sql, params)`` pairs are
        driver-safe on their own (e.g. for inspection or ``RawSQL``-style reuse),
        and adapters encode again at the execution boundary so params supplied
        directly to ``fetch_*``/``execute*`` get the same treatment. A second pass
        is a no-op because already-encoded values (``str``, numbers, native types)
        are returned unchanged.
        """
        # Fast path — keep zero overhead for the common case.
        # Check Enum before ``str`` so StrEnum unwraps to ``.value``.
        if value is None or isinstance(value, (bool, int, float, bytes, bytearray)):
            return value
        if isinstance(value, Enum):
            inner = value.value
            if isinstance(inner, (bool, int, float, str, bytes, bytearray)) or inner is None:
                return inner
            return self._encode_param_slow(inner)
        if isinstance(value, str):
            return value
        return self._encode_param_slow(value)

    def _encode_param_slow(self, value: Any) -> Any:
        if isinstance(value, UUID):
            return value if self.native_uuid else str(value)
        if isinstance(value, Decimal):
            return value if self.native_decimal else str(value)
        if isinstance(value, datetime):
            return value if self.native_datetime else value.isoformat()
        if isinstance(value, date):
            return value if self.native_datetime else value.isoformat()
        if isinstance(value, time):
            return value if self.native_datetime else value.isoformat()
        if isinstance(value, Json):
            return json.dumps(value.value, separators=(",", ":"), default=str)
        if isinstance(value, list):
            if self.native_array:
                encoded = [self.encode_param(elem) for elem in value]
                if any(e is not elem for e, elem in zip(encoded, value, strict=True)):
                    return encoded
                return cast("list[object]", value)
            return json.dumps(value, separators=(",", ":"), default=str)
        if isinstance(value, dict):
            return json.dumps(value, separators=(",", ":"), default=str)
        return value

    def encode_params(self, params: Sequence[Any]) -> tuple[Any, ...]:
        """Encode a positional parameter sequence (idempotent for already-encoded values)."""
        if not params:
            return ()
        return tuple(self.encode_param(value) for value in params)

    def encode_params_seq(self, params_seq: Sequence[Sequence[Any]]) -> list[tuple[Any, ...]]:
        """Encode each row for ``executemany``-style APIs."""
        return [self.encode_params(row) for row in params_seq]


class PostgresDialect(Dialect):
    """PostgreSQL dialect."""

    __slots__ = ()
    name = "postgres"
    quote_char = '"'
    supports_returning = True
    supports_ilike = True
    positional_style = "dollar"
    supports_for_update = True
    supports_for_share = True
    supports_skip_locked = True
    supports_lock_of = True
    supports_distinct_on = True
    native_uuid = True
    native_decimal = True
    native_datetime = True
    native_array = True


class CockroachDialect(PostgresDialect):
    """CockroachDB dialect (PostgreSQL wire-compatible)."""

    __slots__ = ()
    name = "cockroach"


class MySQLDialect(Dialect):
    """MySQL dialect."""

    __slots__ = ()
    name = "mysql"
    quote_char = "`"
    supports_returning = False
    supports_ilike = False
    positional_style = "format"
    upsert_style = "duplicate"
    supports_for_update = True
    supports_for_share = True
    supports_skip_locked = True
    supports_lock_of = True
    supports_distinct_on = False
    native_uuid = False
    native_decimal = False
    native_datetime = True
    native_array = False


class TiDBDialect(MySQLDialect):
    """TiDB dialect (MySQL-compatible).

    TiDB accepts ``FOR UPDATE`` / ``LOCK IN SHARE MODE`` but not ``FOR SHARE``
    or ``SKIP LOCKED``; ``OF`` table lists are supported on recent versions.
    """

    __slots__ = ()
    name = "tidb"
    supports_for_share = False
    supports_skip_locked = False
    supports_lock_of = True


class SQLiteDialect(Dialect):
    """SQLite 3.35+ dialect."""

    __slots__ = ()
    name = "sqlite"
    quote_char = '"'
    supports_returning = True
    supports_ilike = False
    positional_style = "qmark"
    supports_for_update = False
    supports_for_share = False
    supports_skip_locked = False
    supports_lock_of = False
    supports_distinct_on = False
    native_uuid = False
    native_decimal = False
    native_datetime = False
    native_array = False
