"""Load Atlas schema JSON from stdin, files, or the Atlas CLI."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TextIO

# ``user:password@`` -> ``user:***@``. Userinfo starts right after ``://`` or,
# for schemeless DSNs (e.g. MySQL ``user:pass@tcp(host)/db``), at the start of a
# token. Username may be empty (``:pass@host``). Password is everything after the
# first ``:`` up to the *last* ``@`` that is followed by a host-like authority
# (so ``/``, ``?``, ``@``, and ``#`` inside the password do not break masking,
# while query values like ``?x=a@b`` stay intact).
_USERINFO_START_RE = re.compile(r"(?:(?<=://)|(?<![\w@:/.%+-]))(?P<user>[^\s:/?#@'\"]*):")
_HOST_AFTER_AT_RE = re.compile(
    r"(?:"
    r"tcp\("
    r"|unix\("
    r"|\[[0-9A-Fa-f:.]+\]"  # IPv6
    r"|(?:\d{1,3}\.){3}\d{1,3}"  # IPv4
    r"|(?=.{2})[A-Za-z0-9][A-Za-z0-9._-]*"  # hostname (len >= 2)
    r")"
)


class InspectError(RuntimeError):
    """Raised when schema input cannot be loaded."""


def mask_credentials(text: str) -> str:
    """Replace DSN passwords with ``***`` so logs/errors never leak secrets."""
    parts: list[str] = []
    cursor = 0
    for match in _USERINFO_START_RE.finditer(text):
        if match.start() < cursor:
            continue
        user = match.group("user")
        pass_start = match.end()
        # Reject the scheme name itself (``postgres:`` in ``postgres://...``).
        if text.startswith("//", pass_start):
            continue
        token_end = pass_start
        while token_end < len(text) and text[token_end] not in " \t\n\r'\"":
            token_end += 1
        segment = text[pass_start:token_end]
        at_index = _userinfo_at_index(segment)
        if at_index is None:
            continue
        parts.append(text[cursor : match.start()])
        parts.append(f"{user}:***@")
        parts.append(segment[at_index + 1 :])
        cursor = token_end
    parts.append(text[cursor:])
    return "".join(parts)


def _userinfo_at_index(segment: str) -> int | None:
    """Index of the ``@`` that ends userinfo (last ``@`` before a host-like authority)."""
    chosen: int | None = None
    start = 0
    while True:
        at = segment.find("@", start)
        if at < 0:
            break
        if _HOST_AFTER_AT_RE.match(segment, at + 1) is not None:
            chosen = at
        start = at + 1
    return chosen


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
