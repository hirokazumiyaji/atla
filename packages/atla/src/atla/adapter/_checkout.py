"""Per-task checkout tracking for pool-backed adapters.

Pool adapters must allow concurrent coroutines to share one adapter instance.
Instance flags like ``_tx_active`` incorrectly block unrelated tasks, so checkout
ownership is keyed by ``asyncio.current_task()`` identity.

Child tasks created with ``asyncio.create_task`` get their own task id and may
still acquire from the pool; only the task that entered ``transaction()`` is
blocked from using the outer adapter.
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

# adapter id -> set of task ids that currently hold a transaction checkout
_checkouts: dict[int, set[int]] = {}

_CHECKOUT_MSG = (
    "connection is checked out for an active transaction; use the adapter yielded by transaction()"
)


def ensure_not_checked_out(adapter: Any) -> None:
    """Raise if the current task holds a transaction checkout on *adapter*."""
    task = asyncio.current_task()
    if task is None:
        return
    owners = _checkouts.get(id(adapter))
    if owners is not None and id(task) in owners:
        raise RuntimeError(_CHECKOUT_MSG)


@contextmanager
def checkout_scope(adapter: Any) -> Generator[None]:
    """Mark *adapter* as checked out by the current task until exit."""
    task = asyncio.current_task()
    adapter_key = id(adapter)
    task_key = id(task) if task is not None else None
    if task_key is not None:
        _checkouts.setdefault(adapter_key, set()).add(task_key)
    try:
        yield
    finally:
        if task_key is not None:
            owners = _checkouts.get(adapter_key)
            if owners is not None:
                owners.discard(task_key)
                if not owners:
                    del _checkouts[adapter_key]
