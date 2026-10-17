"""Load Atlas schema JSON from stdin, files, or the Atlas CLI."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import TextIO


class InspectError(RuntimeError):
    """Raised when schema input cannot be loaded."""


def load_schema_json(
    schema: str,
    *,
    dev_url: str | None = None,
    stdin: TextIO | None = None,
) -> str:
    """Return Atlas inspect JSON text for the given ``--schema`` value."""
    if schema == "-":
        handle = stdin if stdin is not None else sys.stdin
        text = handle.read()
        if not text.strip():
            raise InspectError("no Atlas JSON received on stdin")
        return text

    if "://" in schema:
        return _inspect_with_atlas(schema, dev_url=dev_url)

    path = Path(schema)
    if not path.is_file():
        raise InspectError(f"schema file not found: {path}")
    return path.read_text(encoding="utf-8")


def _inspect_with_atlas(schema: str, *, dev_url: str | None) -> str:
    atlas = shutil.which("atlas")
    if atlas is None:
        raise InspectError("atlas CLI not found on PATH; install Atlas or pass a JSON file / stdin")
    command = [
        atlas,
        "schema",
        "inspect",
        "--url",
        schema,
        "--format",
        "{{ json . }}",
    ]
    if dev_url:
        command.extend(["--dev-url", dev_url])
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise InspectError(f"failed to execute atlas: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise InspectError(f"atlas schema inspect failed (exit {completed.returncode}): {detail}")
    text = completed.stdout
    if not text.strip():
        raise InspectError("atlas schema inspect returned empty output")
    return text
