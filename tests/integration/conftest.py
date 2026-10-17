"""Shared fixtures for integration tests."""

# pyright: reportUnknownMemberType=false

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest


def clear_generated_appdb() -> None:
    """Drop cached ``appdb`` modules so the next import hits the intended out dir."""
    for name in list(sys.modules):
        if name == "appdb" or name.startswith("appdb."):
            del sys.modules[name]


@pytest.fixture(autouse=True)
def _clear_generated_appdb_modules() -> Iterator[None]:
    """Ensure each E2E test imports its own generated ``appdb``, not a prior dialect's."""
    clear_generated_appdb()
    yield
    clear_generated_appdb()


@pytest.fixture
def load_appdb(monkeypatch: pytest.MonkeyPatch) -> Callable[[Path], Any]:
    """Return a loader that imports generated ``appdb`` from a tmp_path parent."""

    def _load(tmp_path: Path) -> Any:
        clear_generated_appdb()
        monkeypatch.syspath_prepend(str(tmp_path))
        return importlib.import_module("appdb")

    return _load
