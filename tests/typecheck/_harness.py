"""Shared helpers for pyright / mypy generated-code typecheck tests."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "atlas"

DIALECT_SCHEMAS: tuple[tuple[str, str], ...] = (
    ("postgres", "postgres_users.json"),
    ("mysql", "mysql_users.json"),
    ("sqlite", "sqlite_users.json"),
)


def run_cmd(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def tool_cmd(tool: str) -> list[str]:
    """Resolve pyright/mypy through the workspace toolchain when possible."""
    venv_bin = ROOT / ".venv" / "bin" / tool
    if venv_bin.is_file():
        return [str(venv_bin)]
    if shutil.which("uv") is not None:
        return ["uv", "run", tool]
    return [sys.executable, "-m", tool]


def generate_package(out_dir: Path, *, schema: Path, dialect: str) -> None:
    from atlagen.cli import main

    code = main(
        [
            "generate",
            "--dialect",
            dialect,
            "--schema",
            str(schema),
            "--out",
            str(out_dir),
        ]
    )
    assert code == 0, f"atlagen generate failed for {schema} ({dialect})"
    assert (out_dir / "__init__.py").is_file()


def write_typecheck_project(work: Path, *, package_dir: Path, sources: list[Path]) -> None:
    """Write pyright / mypy configs so generated packages resolve against the workspace venv."""
    pyright_config = {
        "typeCheckingMode": "strict",
        "pythonVersion": "3.12",
        "venvPath": str(ROOT),
        "venv": ".venv",
        "extraPaths": [
            str(ROOT / "packages" / "atla" / "src"),
            str(ROOT / "packages" / "atlagen" / "src"),
            str(work),
        ],
        # Fixture modules intentionally define entrypoints that are never called at runtime.
        "reportUnusedFunction": "none",
        "include": ["appdb", *[path.name for path in sources]],
        "exclude": ["**/__pycache__"],
    }
    (work / "pyrightconfig.json").write_text(
        json.dumps(pyright_config, indent=2) + "\n",
        encoding="utf-8",
    )
    mypy_path = ":".join(
        [
            str(ROOT / "packages" / "atla" / "src"),
            str(ROOT / "packages" / "atlagen" / "src"),
            str(work),
        ]
    )
    mypy_lines = [
        "[mypy]",
        "python_version = 3.12",
        "strict = True",
        "pretty = True",
        f"mypy_path = {mypy_path}",
        "explicit_package_bases = True",
        "",
    ]
    (work / "mypy.ini").write_text("\n".join(mypy_lines), encoding="utf-8")


def run_pyright(work: Path, *paths: Path) -> subprocess.CompletedProcess[str]:
    return run_cmd(
        [*tool_cmd("pyright"), "--project", str(work), *[str(p) for p in paths]],
        cwd=work,
    )


def run_mypy(work: Path, *paths: Path) -> subprocess.CompletedProcess[str]:
    return run_cmd(
        [
            *tool_cmd("mypy"),
            "--config-file",
            str(work / "mypy.ini"),
            *[str(p) for p in paths],
        ],
        cwd=work,
    )


def assert_typechecks_clean(work: Path, *paths: Path) -> None:
    pyright = run_pyright(work, *paths)
    assert pyright.returncode == 0, pyright.stdout + pyright.stderr
    mypy = run_mypy(work, *paths)
    assert mypy.returncode == 0, mypy.stdout + mypy.stderr


def assert_typechecks_fail(work: Path, *paths: Path, must_mention: tuple[str, ...]) -> None:
    # Negative checks must target app sources only so generated-package errors
    # cannot satisfy returncode != 0 / token matching.
    app_paths = tuple(p for p in paths if p.is_file() and p.parent == work)
    check_paths = app_paths or paths

    pyright = run_pyright(work, *check_paths)
    assert pyright.returncode != 0, pyright.stdout + pyright.stderr
    pyright_out = (pyright.stdout + pyright.stderr).lower()
    assert all(token.lower() in pyright_out for token in must_mention), pyright_out
    assert any(f"{path.name.lower()}:" in pyright_out for path in check_paths), pyright_out

    mypy = run_mypy(work, *check_paths)
    assert mypy.returncode != 0, mypy.stdout + mypy.stderr
    mypy_out = (mypy.stdout + mypy.stderr).lower()
    assert "error" in mypy_out
    assert all(token.lower() in mypy_out for token in must_mention), mypy_out
    assert any(f"{path.name.lower()}:" in mypy_out for path in check_paths), mypy_out


def write_text(path: Path, content: str) -> Path:
    path.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8")
    return path
