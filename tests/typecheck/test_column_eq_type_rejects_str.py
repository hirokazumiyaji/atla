"""Acceptance: Column[int] == str is rejected by pyright and mypy strict mode."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

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


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def test_pyright_rejects_column_int_eq_str() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fixture = Path(tmp) / "column_eq_wrong_type.py"
        fixture.write_text(FIXTURE_SOURCE)
        result = _run([sys.executable, "-m", "pyright", str(fixture)], cwd=ROOT)
    assert result.returncode != 0, result.stdout + result.stderr
    combined = result.stdout + result.stderr
    assert "bool" in combined.lower() or "eq" in combined.lower()


def test_mypy_rejects_column_int_eq_str() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fixture = Path(tmp) / "column_eq_wrong_type.py"
        fixture.write_text(FIXTURE_SOURCE)
        result = _run(
            [
                sys.executable,
                "-m",
                "mypy",
                "--strict",
                "--python-version=3.12",
                str(fixture),
            ],
            cwd=ROOT,
        )
    assert result.returncode != 0, result.stdout + result.stderr
    combined = result.stdout + result.stderr
    assert "error" in combined.lower()
