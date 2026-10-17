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
