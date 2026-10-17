"""CLI entry point for atlagen."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Run the atlagen CLI.

    The full `generate` / `check` subcommands are implemented in a later phase.
    This stub exists so the console script is installable and CI can import it.
    """
    _ = argv if argv is not None else sys.argv[1:]
    print("atlagen: not implemented yet", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
