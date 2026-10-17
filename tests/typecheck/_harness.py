"""Shared helpers for pyright / mypy generated-code typecheck tests."""

from __future__ import annotations

import json
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
        [sys.executable, "-m", "pyright", "--project", str(work), *[str(p) for p in paths]],
        cwd=work,
    )


def run_mypy(work: Path, *paths: Path) -> subprocess.CompletedProcess[str]:
    return run_cmd(
        [
            sys.executable,
            "-m",
            "mypy",
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
    pyright = run_pyright(work, *paths)
    assert pyright.returncode != 0, pyright.stdout + pyright.stderr
    pyright_out = (pyright.stdout + pyright.stderr).lower()
    assert any(token.lower() in pyright_out for token in must_mention), pyright_out

    mypy = run_mypy(work, *paths)
    assert mypy.returncode != 0, mypy.stdout + mypy.stderr
    mypy_out = (mypy.stdout + mypy.stderr).lower()
    assert "error" in mypy_out
    assert any(token.lower() in mypy_out for token in must_mention), mypy_out


def write_text(path: Path, content: str) -> Path:
    path.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8")
    return path
