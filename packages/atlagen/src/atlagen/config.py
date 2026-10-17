"""CLI argument and pyproject.toml configuration loading."""

from __future__ import annotations

import argparse
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

SUPPORTED_DIALECTS = frozenset({"postgres", "cockroach", "mysql", "tidb", "sqlite"})


@dataclass(slots=True, frozen=True)
class AtlagenConfig:
    """Merged runtime configuration for generate/check."""

    dialect: str
    out: Path
    schema: str
    dev_url: str | None = None
    schema_name: str | None = None
    frozen: bool = True
    # table name -> {model_name?, accessor_name?}
    table_overrides: dict[str, dict[str, str]] | None = None


class ConfigError(ValueError):
    """Raised when configuration is incomplete or invalid."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlagen")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("generate", "Generate atla client modules from an Atlas schema"),
        ("check", "Fail if generated modules are out of date with the schema"),
    ):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument(
            "--dialect",
            choices=sorted(SUPPORTED_DIALECTS),
            default=None,
            help="Target database dialect",
        )
        cmd.add_argument(
            "--out",
            "-o",
            type=Path,
            default=None,
            help="Output directory for generated Python modules",
        )
        cmd.add_argument(
            "--schema",
            "-s",
            default=None,
            help="Atlas JSON path, '-' for stdin, or Atlas URL (file://, postgres://, ...)",
        )
        cmd.add_argument(
            "--dev-url",
            default=None,
            help="Passed to atlas schema inspect --dev-url when inspecting non-JSON sources",
        )
        cmd.add_argument(
            "--schema-name",
            default=None,
            help="Database schema name when the Atlas JSON contains multiple schemas",
        )
        frozen = cmd.add_mutually_exclusive_group()
        frozen.add_argument(
            "--frozen",
            action="store_true",
            default=None,
            help="Generate frozen dataclasses (default)",
        )
        frozen.add_argument(
            "--mutable",
            action="store_true",
            default=False,
            help="Generate mutable dataclasses",
        )
    return parser


def load_pyproject_settings(path: Path | None = None) -> dict[str, Any]:
    """Load ``[tool.atlagen]`` from pyproject.toml if present."""
    pyproject = path or Path("pyproject.toml")
    if not pyproject.is_file():
        return {}
    try:
        with pyproject.open("rb") as handle:
            data: dict[str, Any] = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid pyproject.toml while reading [tool.atlagen]: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"failed to read pyproject.toml: {exc}") from exc
    tool = data.get("tool")
    if not isinstance(tool, dict):
        return {}
    section = cast(dict[str, Any], tool).get("atlagen")
    if not isinstance(section, dict):
        return {}
    return {str(key): value for key, value in cast(dict[str, Any], section).items()}


def resolve_config(
    args: argparse.Namespace,
    *,
    pyproject_settings: dict[str, Any] | None = None,
    stdin_isatty: bool | None = None,
) -> AtlagenConfig:
    """Merge CLI args with pyproject settings into a validated config."""
    settings = pyproject_settings if pyproject_settings is not None else load_pyproject_settings()
    dialect = _coalesce_str(args.dialect, settings.get("dialect"))
    out_value = args.out if args.out is not None else settings.get("out")
    schema = _coalesce_str(args.schema, settings.get("schema"), default="-") or "-"
    dev_url = _coalesce_str(args.dev_url, settings.get("dev_url"))
    schema_name = _coalesce_str(args.schema_name, settings.get("schema_name"))

    frozen = True
    if getattr(args, "mutable", False):
        frozen = False
    elif getattr(args, "frozen", None) is True:
        frozen = True
    elif "frozen" in settings:
        frozen_setting = settings["frozen"]
        if not isinstance(frozen_setting, bool):
            raise ConfigError("[tool.atlagen].frozen must be a TOML boolean (true/false)")
        frozen = frozen_setting

    if not dialect:
        raise ConfigError("--dialect is required (CLI or [tool.atlagen] in pyproject.toml)")
    if dialect not in SUPPORTED_DIALECTS:
        raise ConfigError(f"unsupported dialect: {dialect!r}")
    if out_value is None:
        raise ConfigError("--out is required (CLI or [tool.atlagen] in pyproject.toml)")

    out = Path(out_value)
    isatty = sys.stdin.isatty() if stdin_isatty is None else stdin_isatty
    if schema == "-" and isatty and args.schema is None and "schema" not in settings:
        print(
            "warning: reading Atlas JSON from stdin (interactive TTY); "
            "pass --schema PATH or pipe JSON into stdin",
            file=sys.stderr,
        )

    table_overrides = _parse_table_overrides(settings.get("tables"))

    return AtlagenConfig(
        dialect=dialect,
        out=out,
        schema=schema,
        dev_url=dev_url,
        schema_name=schema_name,
        frozen=frozen,
        table_overrides=table_overrides,
    )


def _parse_table_overrides(raw: object) -> dict[str, dict[str, str]] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ConfigError("[tool.atlagen].tables must be a TOML table")
    tables = cast(dict[str, Any], raw)
    result: dict[str, dict[str, str]] = {}
    for table_name, value in tables.items():
        if not isinstance(value, dict):
            raise ConfigError(
                f"[tool.atlagen.tables.{table_name}] must be a TOML table "
                "with optional model_name / accessor_name"
            )
        entry_raw = cast(dict[str, Any], value)
        entry: dict[str, str] = {}
        for key in ("model_name", "accessor_name"):
            if key in entry_raw:
                raw_value = entry_raw[key]
                if not isinstance(raw_value, str):
                    raise ConfigError(
                        f"[tool.atlagen.tables.{table_name}].{key} must be a string"
                    )
                text = raw_value.strip()
                if not text:
                    raise ConfigError(
                        f"[tool.atlagen.tables.{table_name}].{key} must be a non-empty string"
                    )
                entry[key] = text
        if entry:
            result[str(table_name)] = entry
    return result or None


def _coalesce_str(
    *values: object,
    default: str | None = None,
) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return default
