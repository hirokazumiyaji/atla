"""Smoke tests for the monorepo workspace packages."""

from __future__ import annotations

import pytest

import atla
import atlagen
from atlagen.cli import main


def test_atla_version() -> None:
    assert atla.__version__ == "0.1.0"


def test_atlagen_version() -> None:
    assert atlagen.__version__ == "0.1.0"


def test_atlagen_cli_requires_command() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2
