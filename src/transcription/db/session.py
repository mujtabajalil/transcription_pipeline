from __future__ import annotations

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


def make_engine(database_url: str, *, pool_size: int = 10) -> Engine:
    return create_engine(
        database_url,
        pool_size=pool_size,
        max_overflow=pool_size,
        pool_pre_ping=True,  # survive Postgres restarts / idle-connection reaping
        pool_recycle=1800,
    )


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)
