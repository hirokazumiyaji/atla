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


class Json[T]:
    """Explicit wrapper marking a Python object for SQL JSON / JSONB serialization."""

    __slots__ = ("value",)

    def __init__(self, value: T) -> None:
        self.value = value

    def __repr__(self) -> str:
        return f"Json({self.value!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Json):
            return bool(self.value == other.value)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
        return False
