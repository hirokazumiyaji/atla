"""Acceptance: Column[int] == str is rejected by pyright and mypy strict mode."""

from __future__ import annotations

import tempfile
from pathlib import Path

from typecheck._harness import ROOT, run_cmd, tool_cmd, write_text

FIXTURE_SOURCE = '''\
"""Negative type fixture: Column[int] compared to str must fail static checks."""

from atla import Column, Expr

age: Column[int] = Column("users", "age")


def _need_bool_expr(clause: Expr[bool]) -> None:
    raise NotImplementedError


# Wrong-type ``==`` is treated as bool (not Expr[bool]) by strict checkers.
_need_bool_expr(age == "20")
# Explicit ``.eq`` rejects the wrong operand type directly.
_need_bool_expr(age.eq("20"))
'''


def test_pyright_rejects_column_int_eq_str() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fixture = write_text(Path(tmp) / "column_eq_wrong_type.py", FIXTURE_SOURCE)
        result = run_cmd([*tool_cmd("pyright"), str(fixture)], cwd=ROOT)
    assert result.returncode != 0, result.stdout + result.stderr
    combined = result.stdout + result.stderr
    assert "bool" in combined.lower() or "eq" in combined.lower()


def test_mypy_rejects_column_int_eq_str() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fixture = write_text(Path(tmp) / "column_eq_wrong_type.py", FIXTURE_SOURCE)
        result = run_cmd(
            [
                *tool_cmd("mypy"),
                "--strict",
                "--python-version=3.12",
                str(fixture),
            ],
            cwd=ROOT,
        )
    assert result.returncode != 0, result.stdout + result.stderr
    combined = result.stdout + result.stderr
    assert "error" in combined.lower()
