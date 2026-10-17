"""Optional PostgreSQL / MySQL E2E tests (skipped unless DSNs are provided)."""

from __future__ import annotations

import os

import pytest

POSTGRES_DSN = os.environ.get("ATLA_TEST_POSTGRES_DSN")
MYSQL_DSN = os.environ.get("ATLA_TEST_MYSQL_DSN")


@pytest.mark.skipif(not POSTGRES_DSN, reason="Set ATLA_TEST_POSTGRES_DSN to enable")
@pytest.mark.asyncio
async def test_postgres_smoke() -> None:
    import asyncpg  # type: ignore[import-untyped]

    from atla.adapter.asyncpg import AsyncpgAdapter

    conn = await asyncpg.connect(POSTGRES_DSN)
    try:
        adapter = AsyncpgAdapter(conn)
        value = await adapter.fetch_val("SELECT 1")
        assert int(value) == 1
    finally:
        await conn.close()


@pytest.mark.skipif(not MYSQL_DSN, reason="Set ATLA_TEST_MYSQL_DSN to enable")
@pytest.mark.asyncio
async def test_mysql_smoke() -> None:
    import asyncmy

    from atla.adapter.asyncmy import AsyncmyAdapter

    conn = await asyncmy.connect(MYSQL_DSN)
    try:
        adapter = AsyncmyAdapter(conn)
        value = await adapter.fetch_val("SELECT 1")
        assert int(value) == 1
    finally:
        conn.close()
