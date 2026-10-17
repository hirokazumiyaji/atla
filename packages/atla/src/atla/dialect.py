"""RDBMS dialect abstractions for identifier quoting and placeholders."""

from __future__ import annotations

from typing import ClassVar, Literal

_REQUIRED_DIALECT_ATTRS = (
    "name",
    "quote_char",
    "supports_returning",
    "supports_ilike",
    "positional_style",
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
    upsert_style: ClassVar[Literal["conflict", "duplicate"]] = "conflict"

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

    def placeholder(self, index: int) -> str:
        if self.positional_style == "dollar":
            return f"${index}"
        if self.positional_style == "format":
            return "%s"
        if self.positional_style == "qmark":
            return "?"
        msg = f"unsupported positional_style: {self.positional_style!r}"
        raise ValueError(msg)


class PostgresDialect(Dialect):
    """PostgreSQL dialect."""

    __slots__ = ()
    name = "postgres"
    quote_char = '"'
    supports_returning = True
    supports_ilike = True
    positional_style = "dollar"


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


class TiDBDialect(MySQLDialect):
    """TiDB dialect (MySQL-compatible)."""

    __slots__ = ()
    name = "tidb"


class SQLiteDialect(Dialect):
    """SQLite 3.35+ dialect."""

    __slots__ = ()
    name = "sqlite"
    quote_char = '"'
    supports_returning = True
    supports_ilike = False
    positional_style = "qmark"
