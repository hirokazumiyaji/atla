"""Per-task checkout tracking for pool-backed adapters.

Pool adapters must allow concurrent coroutines to share one adapter instance.
Instance flags like ``_tx_active`` incorrectly block unrelated tasks, so checkout
ownership is tracked per :class:`asyncio.Task`.

Each adapter instance owns a :class:`~weakref.WeakSet` of tasks (stored on the
instance as ``_checkout_owners``), so there is no module-global registry keyed
by ``id(adapter)`` that could be reused after garbage collection. Tasks are held
weakly, so cancelled/collected tasks cannot leave stale entries.

Child tasks created with ``asyncio.create_task`` are distinct task objects and
may still acquire from the pool; only the task that entered ``transaction()``
is blocked from using the outer adapter.
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from weakref import WeakSet

CheckoutOwners = WeakSet["asyncio.Task[Any]"]

_CHECKOUT_MSG = (
    "connection is checked out for an active transaction; use the adapter yielded by transaction()"
)


def new_checkout_owners() -> CheckoutOwners:
    """Create an empty per-adapter owner set."""
    return WeakSet()


def ensure_not_checked_out(owners: CheckoutOwners) -> None:
    """Raise if the current task holds a transaction checkout recorded in *owners*."""
    task = asyncio.current_task()
    if task is not None and task in owners:
        raise RuntimeError(_CHECKOUT_MSG)


@contextmanager
def checkout_scope(owners: CheckoutOwners) -> Generator[None]:
    """Mark the current task as holding a checkout in *owners* until exit."""
    task = asyncio.current_task()
    if task is not None:
        owners.add(task)
    try:
        yield
    finally:
        if task is not None:
            owners.discard(task)
