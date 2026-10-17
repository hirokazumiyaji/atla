"""CLI entry point for atlagen."""

from __future__ import annotations

import sys

from atlagen.config import AtlagenConfig, ConfigError, build_parser, resolve_config
from atlagen.emitter import generate_files
from atlagen.inspect import InspectError, load_schema_json
from atlagen.mapper import map_schema
from atlagen.parser import AtlasParseError, parse_atlas_json


def main(argv: list[str] | None = None) -> int:
    """Run the atlagen CLI."""
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    try:
        config = resolve_config(args)
    except ConfigError as exc:
        print(f"atlagen: {exc}", file=sys.stderr)
        return 2

    try:
        if args.command == "generate":
            return _cmd_generate(config)
        if args.command == "check":
            return _cmd_check(config)
    except ConfigError as exc:
        print(f"atlagen: {exc}", file=sys.stderr)
        return 2
    except (InspectError, AtlasParseError, OSError) as exc:
        print(f"atlagen: {exc}", file=sys.stderr)
        return 1

    parser.error(f"unknown command: {args.command}")
    return 2


def _cmd_generate(config: AtlagenConfig) -> int:
    files = _render(config)
    config.out.mkdir(parents=True, exist_ok=True)
    for relative, content in files.items():
        target = config.out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    print(f"atlagen: wrote {len(files)} file(s) to {config.out}")
    return 0


def _cmd_check(config: AtlagenConfig) -> int:
    """Verify generated output matches the schema and contains no unexpected files.

    The output directory is treated as generated-only: any file not produced by
    ``generate`` (including hand-written helpers, ``*.pyi``, ``py.typed``,
    leftover ``.sql`` / ``.json``, etc.) is reported as ``unexpected:``.
    ``__pycache__`` directories and their contents are ignored.
    """
    files = _render(config)
    mismatches: list[str] = []
    for relative, expected in files.items():
        target = config.out / relative
        if not target.is_file():
            mismatches.append(f"missing: {relative}")
            continue
        actual = target.read_text(encoding="utf-8")
        if actual != expected:
            mismatches.append(f"drift: {relative}")
    expected_paths = set(files)
    if config.out.is_dir():
        for path in config.out.rglob("*"):
            if "__pycache__" in path.parts or not path.is_file():
                continue
            relative = path.relative_to(config.out).as_posix()
            if relative not in expected_paths:
                mismatches.append(f"unexpected: {relative}")
    if mismatches:
        print("atlagen check failed:", file=sys.stderr)
        for item in mismatches:
            print(f"  - {item}", file=sys.stderr)
        return 1
    print(f"atlagen: {config.out} is up to date")
    return 0


def _render(config: AtlagenConfig) -> dict[str, str]:
    raw = load_schema_json(config.schema, dev_url=config.dev_url)
    schema = parse_atlas_json(
        raw,
        dialect=config.dialect,
        schema_name=config.schema_name,
        table_overrides=config.table_overrides,
    )
    mapped = map_schema(schema)
    return generate_files(mapped, frozen=config.frozen)


if __name__ == "__main__":
    raise SystemExit(main())
