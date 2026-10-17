"""Type tests for join / left_join / preload return inference."""

from __future__ import annotations

from pathlib import Path

from typecheck._harness import (
    FIXTURES,
    assert_typechecks_clean,
    generate_package,
    write_text,
    write_typecheck_project,
)

JOIN_APP = '''\
"""Join / left_join / preload return types must be inferred precisely."""

from __future__ import annotations

from typing import assert_type, cast

from atla.adapter.base import DBAdapter

from appdb import AtlaClient, Post, PostRecord, User, UserRecord


async def exercise(db: AtlaClient) -> None:
    joined = await db.user.query().join(Post, Post.user_id == User.id).all()
    assert_type(joined, list[tuple[UserRecord, PostRecord]])

    left = await db.user.query().left_join(Post, Post.user_id == User.id).all()
    assert_type(left, list[tuple[UserRecord, PostRecord | None]])

    manager = User.as_("manager")
    self_joined = await db.user.query().join(manager, manager.id == User.id).all()
    assert_type(self_joined, list[tuple[UserRecord, UserRecord]])

    preloaded = await db.user.with_posts().all()
    assert_type(preloaded, list[tuple[UserRecord, list[PostRecord]]])

    filtered = await (
        db.user.where(User.id == 1)
        .with_posts(Post.title.eq("hello"), order_by=[Post.id.asc()])
        .one()
    )
    assert_type(filtered, tuple[UserRecord, list[PostRecord]])

    belongs = await db.post.with_user().all()
    assert_type(belongs, list[tuple[PostRecord, UserRecord | None]])


async def _entry() -> None:
    await exercise(AtlaClient(cast(DBAdapter, object())))
'''


def test_join_left_join_preload_types(tmp_path: Path) -> None:
    package = tmp_path / "appdb"
    generate_package(
        package,
        schema=FIXTURES / "postgres_users.json",
        dialect="postgres",
    )
    app = write_text(tmp_path / "join_app.py", JOIN_APP)
    write_typecheck_project(tmp_path, package_dir=package, sources=[app])
    assert_typechecks_clean(tmp_path, package, app)
