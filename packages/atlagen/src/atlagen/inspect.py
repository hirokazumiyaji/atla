"""Load Atlas schema JSON from stdin, files, or the Atlas CLI."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TextIO

# ``user:password@`` -> ``user:***@``. The userinfo starts right after ``://`` or,
# for schemeless DSNs (e.g. MySQL ``user:pass@tcp(host)/db``), at the start of a
# token. Username may be empty (``:pass@host``). The password is matched greedily
# up to the *last* ``@`` before the first ``/``, ``?``, or whitespace so passwords
# that contain ``@`` or ``#`` (legal in Go MySQL/TiDB DSNs) do not leak.
_CREDENTIAL_RE = re.compile(
    r"(?:(?<=://)|(?<![\w@:/.%+-]))"
    r"(?P<user>[^\s:/?#@'\"]*):"
    r"(?P<password>[^\s/?]*)@"
)


class InspectError(RuntimeError):
    """Raised when schema input cannot be loaded."""


def mask_credentials(text: str) -> str:
    """Replace DSN passwords with ``***`` so logs/errors never leak secrets."""
    return _CREDENTIAL_RE.sub(lambda m: f"{m.group('user')}:***@", text)


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
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        raise InspectError("atlas schema inspect timed out after 120s") from exc
    except OSError as exc:
        raise InspectError(f"failed to execute atlas: {mask_credentials(str(exc))}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise InspectError(
            mask_credentials(f"atlas schema inspect failed (exit {completed.returncode}): {detail}")
        )
    text = completed.stdout
    if not text.strip():
        raise InspectError("atlas schema inspect returned empty output")
    return text
