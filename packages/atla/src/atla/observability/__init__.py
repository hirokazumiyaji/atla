"""Optional observability adapters (zero cost when unused)."""

from atla.observability.instrumented import InstrumentedAdapter
from atla.observability.sql_meta import extract_sql_meta

__all__ = [
    "InstrumentedAdapter",
    "OTelAdapter",
    "extract_sql_meta",
]


def __getattr__(name: str) -> object:
    if name == "OTelAdapter":
        from atla.observability.otel import OTelAdapter

        return OTelAdapter
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
