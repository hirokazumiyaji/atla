"""Per-task checkout tracking for pool-backed adapters.

Pool adapters must allow concurrent coroutines to share one adapter instance.
Instance flags like ``_tx_active`` incorrectly block unrelated tasks, so checkout
ownership is keyed by the current :class:`asyncio.Task` object (held in a
:class:`~weakref.WeakSet` so cancelled/collected tasks cannot leave stale IDs).

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

# adapter id -> tasks that currently hold a transaction checkout on that adapter
_checkouts: dict[int, WeakSet[asyncio.Task[Any]]] = {}

_CHECKOUT_MSG = (
    "connection is checked out for an active transaction; use the adapter yielded by transaction()"
)


def ensure_not_checked_out(adapter: Any) -> None:
    """Raise if the current task holds a transaction checkout on *adapter*."""
    task = asyncio.current_task()
    if task is None:
        return
    owners = _checkouts.get(id(adapter))
    if owners is not None and task in owners:
        raise RuntimeError(_CHECKOUT_MSG)


@contextmanager
def checkout_scope(adapter: Any) -> Generator[None]:
    """Mark *adapter* as checked out by the current task until exit."""
    task = asyncio.current_task()
    adapter_key = id(adapter)
    if task is not None:
        owners = _checkouts.get(adapter_key)
        if owners is None:
            owners = WeakSet[asyncio.Task[Any]]()
            _checkouts[adapter_key] = owners
        owners.add(task)
    try:
        yield
    finally:
        if task is not None:
            owners = _checkouts.get(adapter_key)
            if owners is not None:
                owners.discard(task)
                if not owners:
                    del _checkouts[adapter_key]
