"""
Database session management.
"""
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

from config import config


def get_async_engine():
    """Get async database engine."""
    database_url = config.DATABASE_URL
    if database_url.startswith("sqlite:"):
        async_url = database_url.replace("sqlite:", "sqlite+aiosqlite:")
    else:
        async_url = database_url
    return create_async_engine(async_url, echo=False)


def get_sync_engine():
    """Get sync database engine."""
    return create_engine(config.DATABASE_URL, echo=False)


# Create engines
async_engine = get_async_engine()
sync_engine = get_sync_engine()

# Create session factories
AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

SyncSessionLocal = sessionmaker(
    sync_engine,
    expire_on_commit=False
)


async def get_async_session() -> AsyncSession:
    """Dependency for FastAPI to get async database session."""
    async with AsyncSessionLocal() as session:
        yield session


@asynccontextmanager
async def get_async_session_context():
    """Context manager for async database session."""
    async with AsyncSessionLocal() as session:
        yield session


def get_sync_session():
    """Get synchronous database session."""
    return SyncSessionLocal()
