"""Map driver-specific exceptions onto the portable atla error hierarchy."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from functools import wraps
from typing import Any, ParamSpec, TypeVar

from atla.errors import (
    CheckViolationError,
    DataError,
    DatabaseError,
    ForeignKeyViolationError,
    IntegrityError,
    NotNullViolationError,
    OperationalError,
    SerializationError,
    UniqueViolationError,
)

P = ParamSpec("P")
R = TypeVar("R")

_PG_UNIQUE = {"23505"}
_PG_FK = {"23503"}
_PG_NOT_NULL = {"23502"}
_PG_CHECK = {"23514"}
_PG_SERIALIZATION = {"40001", "40P01"}

_MYSQL_UNIQUE = {1062}
_MYSQL_FK = {1451, 1452}
_MYSQL_NOT_NULL = {1048}
_MYSQL_CHECK = {3819}
_MYSQL_DEADLOCK = {1213}


def translate_exception(exc: BaseException) -> DatabaseError | None:
    """Return a normalized :class:`DatabaseError` for *exc*, or ``None`` if unknown."""
    sqlstate = _sqlstate(exc)
    errno = _errno(exc)
    message = str(exc)

    if sqlstate in _PG_UNIQUE or errno in _MYSQL_UNIQUE or _sqlite_match(message, "UNIQUE"):
        return UniqueViolationError(message)
    if sqlstate in _PG_FK or errno in _MYSQL_FK or _sqlite_match(message, "FOREIGN KEY"):
        return ForeignKeyViolationError(message)
    if sqlstate in _PG_NOT_NULL or errno in _MYSQL_NOT_NULL or _sqlite_match(message, "NOT NULL"):
        return NotNullViolationError(message)
    if sqlstate in _PG_CHECK or errno in _MYSQL_CHECK or _sqlite_match(message, "CHECK"):
        return CheckViolationError(message)
    if sqlstate in _PG_SERIALIZATION or errno in _MYSQL_DEADLOCK:
        return SerializationError(message)

    name = type(exc).__name__
    module = type(exc).__module__
    if "UniqueViolation" in name or (
        "IntegrityError" in name and "unique" in message.lower()
    ):
        return UniqueViolationError(message)
    if "ForeignKeyViolation" in name:
        return ForeignKeyViolationError(message)
    if "NotNullViolation" in name:
        return NotNullViolationError(message)
    if "CheckViolation" in name:
        return CheckViolationError(message)
    if "Serialization" in name or "Deadlock" in name:
        return SerializationError(message)
    if "DataError" in name:
        return DataError(message)
    if "OperationalError" in name or "InterfaceError" in name:
        return OperationalError(message)
    if module.startswith(("asyncpg", "asyncmy", "aiosqlite", "sqlite3", "psycopg")):
        if "IntegrityError" in name:
            return IntegrityErrorFallback(message)
        return DatabaseError(message)
    return None


class IntegrityErrorFallback(IntegrityError):
    """Generic integrity error when the driver does not expose a subtype."""


def _sqlstate(exc: BaseException) -> str | None:
    for attr in ("sqlstate", "pgcode", "sqlstate"):
        value = getattr(exc, attr, None)
        if isinstance(value, str) and value:
            return value
    diag = getattr(exc, "diag", None)
    if diag is not None:
        value = getattr(diag, "sqlstate", None)
        if isinstance(value, str) and value:
            return value
    args = getattr(exc, "args", ())
    if args and isinstance(args[0], str) and len(args[0]) == 5 and args[0].isalnum():
        return args[0]
    return None


def _errno(exc: BaseException) -> int | None:
    for attr in ("errno", "args"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
        if isinstance(value, tuple) and value and isinstance(value[0], int):
            return value[0]
    return None


def _sqlite_match(message: str, token: str) -> bool:
    upper = message.upper()
    return token in upper and "CONSTRAINT" in upper


def with_translated_errors(
    func: Callable[P, Coroutine[Any, Any, R]],
) -> Callable[P, Coroutine[Any, Any, R]]:
    """Wrap an async adapter method so driver errors become atla errors."""

    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return await func(*args, **kwargs)
        except Exception as exc:
            mapped = translate_exception(exc)
            if mapped is None:
                raise
            raise mapped from exc

    return wrapper
