"""PostgreSQL connection helpers. Credentials are read from the project .env file."""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL, Engine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def get_engine(echo: bool = False) -> Engine:
    """Build a SQLAlchemy engine for the SubscribeIQ PostgreSQL database.

    Reads DB_HOST, DB_PORT, DB_NAME, DB_USER and DB_PASSWORD from the environment
    (populated from .env).

    Args:
        echo: If True, SQLAlchemy logs every emitted SQL statement.

    Returns:
        A SQLAlchemy Engine using the psycopg2 driver.

    Raises:
        RuntimeError: If any required environment variable is missing.
    """
    keys = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)} (check .env)")

    url = URL.create(
        drivername="postgresql+psycopg2",
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT")),
        database=os.getenv("DB_NAME"),
    )
    return create_engine(url, echo=echo)
