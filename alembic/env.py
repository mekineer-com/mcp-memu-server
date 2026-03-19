"""Alembic migration environment configuration."""

import json
from pathlib import Path

# pylint: disable=no-member
from logging.config import fileConfig

from sqlalchemy import pool

from alembic import context

# Import Base metadata for autogenerate support
# Note: We import from app.models.base which is side-effect free
# (doesn't create database connections or read environment variables)
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


def _normalize_sync_dsn(raw_dsn: str) -> str:
    dsn = str(raw_dsn or "").strip()
    if not dsn:
        raise RuntimeError("Database DSN is empty")

    if dsn in {":memory:", "memory"}:
        return "sqlite:///:memory:"

    if dsn.startswith("postgres://"):
        return "postgresql+psycopg://" + dsn[len("postgres://") :]
    if dsn.startswith("postgresql://") and not dsn.startswith("postgresql+"):
        return "postgresql+psycopg://" + dsn[len("postgresql://") :]
    if dsn.startswith("postgresql+asyncpg://"):
        return "postgresql+psycopg://" + dsn[len("postgresql+asyncpg://") :]
    if dsn.startswith("postgresql+psycopg://"):
        return dsn

    if dsn.startswith("sqlite+aiosqlite://"):
        return dsn.replace("sqlite+aiosqlite://", "sqlite://", 1)
    if dsn.startswith("sqlite://"):
        return dsn
    if dsn.startswith("sqlite:"):
        return dsn

    p = Path(dsn).expanduser()
    if not p.is_absolute():
        p = (_config_dir() / p).resolve()
    else:
        p = p.resolve()
    return f"sqlite:////{p.as_posix().lstrip('/')}"


def get_sync_database_url() -> str:
    cfg = _read_config()
    return _normalize_sync_dsn(_metadata_store_dsn(cfg))


# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    # Get URL from environment variables
    url = get_sync_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    from sqlalchemy import create_engine

    # Get URL from environment variables and create engine directly
    url = get_sync_database_url()
    connectable = create_engine(url, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
