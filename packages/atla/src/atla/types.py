"""Shared sentinel types for optional mutation arguments."""

from __future__ import annotations


class UnsetType:
    """Sentinel type used to omit optional INSERT/UPDATE columns."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"

    def __bool__(self) -> bool:
        return False


UNSET = UnsetType()


def is_unset(value: object) -> bool:
    """Return True when ``value`` is the :data:`UNSET` sentinel."""
    return value is UNSET or isinstance(value, UnsetType)
