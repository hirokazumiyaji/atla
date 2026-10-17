"""RDBMS dialect abstractions for identifier quoting and placeholders."""

from __future__ import annotations


class Dialect:
    """Base dialect describing engine-specific SQL surface syntax."""

    __slots__ = ()

    name: str
    quote_char: str
    supports_returning: bool
    supports_ilike: bool
    positional_style: str  # "dollar" | "format" | "qmark"

    def quote(self, ident: str) -> str:
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
