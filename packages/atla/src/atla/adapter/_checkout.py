"""Per-task checkout tracking for pool-backed adapters.

Pool adapters must allow concurrent coroutines to share one adapter instance.
Instance flags like ``_tx_active`` incorrectly block unrelated tasks, so checkout
state is stored in a :class:`contextvars.ContextVar` scoped to the current task.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any

_active_checkout: ContextVar[Any] = ContextVar("atla_adapter_checkout", default=None)

_CHECKOUT_MSG = (
    "connection is checked out for an active transaction; use the adapter yielded by transaction()"
)


def ensure_not_checked_out(adapter: Any) -> None:
    """Raise if *adapter* is checked out in the current task context."""
    if _active_checkout.get() is adapter:
        raise RuntimeError(_CHECKOUT_MSG)


@contextmanager
def checkout_scope(adapter: Any) -> Generator[None]:
    """Mark *adapter* as checked out for the current task until exit."""
    token: Token[Any] = _active_checkout.set(adapter)
    try:
        yield
    finally:
        _active_checkout.reset(token)
