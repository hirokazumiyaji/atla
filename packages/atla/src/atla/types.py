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
