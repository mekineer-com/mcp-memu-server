"""Database configuration and session management."""

import json
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.base import Base

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"


def _config_dir() -> Path:
    return _CONFIG_PATH.parent


def _read_config() -> dict:
    if not _CONFIG_PATH.exists():
        raise RuntimeError(f"Missing config file: {_CONFIG_PATH}")
    raw = _CONFIG_PATH.read_text(encoding="utf-8")
    cfg = json.loads(raw) if raw.strip() else {}
    if not isinstance(cfg, dict):
        raise RuntimeError(f"Invalid config file: {_CONFIG_PATH}")
    return cfg


def _metadata_store_dsn(cfg: dict) -> str:
    storage = cfg.get("storage") if isinstance(cfg.get("storage"), dict) else {}
    meta = storage.get("metadata_store") if isinstance(storage.get("metadata_store"), dict) else {}
    dsn = str(meta.get("dsn") or "").strip()
    if not dsn:
        raise RuntimeError("Missing storage.metadata_store.dsn in config.json")
    return dsn


def _normalize_async_dsn(raw_dsn: str) -> str:
    dsn = str(raw_dsn or "").strip()
    if not dsn:
        raise RuntimeError("Database DSN is empty")

    if dsn in {":memory:", "memory"}:
        return "sqlite+aiosqlite:///:memory:"

    if dsn.startswith("postgres://"):
        return "postgresql+psycopg://" + dsn[len("postgres://") :]
    if dsn.startswith("postgresql://") and not dsn.startswith("postgresql+"):
        return "postgresql+psycopg://" + dsn[len("postgresql://") :]
    if dsn.startswith("postgresql+asyncpg://"):
        return "postgresql+psycopg://" + dsn[len("postgresql+asyncpg://") :]
    if dsn.startswith("postgresql+psycopg://"):
        return dsn

    if dsn.startswith("sqlite+aiosqlite://"):
        return dsn
    if dsn.startswith("sqlite://"):
        return dsn.replace("sqlite://", "sqlite+aiosqlite://", 1)
    if dsn.startswith("sqlite:"):
        return dsn.replace("sqlite:", "sqlite+aiosqlite:", 1)

    p = Path(dsn).expanduser()
    if not p.is_absolute():
        p = (_config_dir() / p).resolve()
    else:
        p = p.resolve()
    return f"sqlite+aiosqlite:////{p.as_posix().lstrip('/')}"


def get_database_url() -> str:
    cfg = _read_config()
    return _normalize_async_dsn(_metadata_store_dsn(cfg))


# TODO: Consider lazy initialization to avoid executing during module import
# This would prevent database connection issues from failing tests that don't use the database
# Get database URL using the shared function
DATABASE_URL = get_database_url()

# Create SQLAlchemy async engine
engine = create_async_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)
# Async session factory
# Note: autocommit is removed as it's not supported in SQLAlchemy 2.x async_sessionmaker
SessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    autoflush=False,
    expire_on_commit=False,
    bind=engine,
)
# Re-export Base for backward compatibility
__all__ = ["Base", "SessionLocal", "engine", "get_db", "get_database_url"]


async def get_db():
    """Dependency for FastAPI to get async database session."""
    async with SessionLocal() as db:
        yield db
