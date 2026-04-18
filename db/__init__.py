"""Database package — SQLAlchemy async session factory."""

from __future__ import annotations

import os
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

_engine = None
_session_factory = None


def _get_url() -> str:
    host = os.environ["POSTGRES_HOST"]
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ["POSTGRES_DB"]
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"


def get_engine():
    """Return (and lazily create) the async engine singleton."""
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            _get_url(),
            echo=False,
            pool_size=10,
            max_overflow=20,
        )
    return _engine


def get_session_factory():
    """Return the session factory, creating it if needed."""
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields a DB session and closes it on exit."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
