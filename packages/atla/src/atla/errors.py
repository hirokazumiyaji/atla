"""Exception hierarchy for the atla runtime."""

from __future__ import annotations


class AtlaError(Exception):
    """Base exception for all atla runtime errors."""


class NotFoundError(AtlaError):
    """Raised when `.one()` returns 0 rows."""


class MultipleResultsError(AtlaError):
    """Raised when `.one()` returns more than 1 row."""


class UnsafeMutationError(AtlaError):
    """Raised when UPDATE/DELETE has no WHERE without `.allow_full_table()`."""


class DatabaseError(AtlaError):
    """Base class for database / driver errors normalized by adapters."""


class IntegrityError(DatabaseError):
    """Constraint or integrity violation."""


class UniqueViolationError(IntegrityError):
    """Unique / primary-key constraint violation (SQLSTATE 23505, MySQL 1062, …)."""


class ForeignKeyViolationError(IntegrityError):
    """Foreign-key constraint violation (SQLSTATE 23503, MySQL 1451/1452, …)."""


class NotNullViolationError(IntegrityError):
    """NOT NULL constraint violation (SQLSTATE 23502, MySQL 1048, …)."""


class CheckViolationError(IntegrityError):
    """CHECK constraint violation (SQLSTATE 23514, MySQL 3819, …)."""


class SerializationError(DatabaseError):
    """Retryable serialization failure / deadlock (SQLSTATE 40001, MySQL 1213)."""


class OperationalError(DatabaseError):
    """Connection / timeout / operational failure."""


class DataError(DatabaseError):
    """Invalid data for a column or bind parameter."""
