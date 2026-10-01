"""PostgreSQL connection helpers. Credentials are read from the project .env file.

Two roles:
    DB_USER           owner/admin role used by the pipeline scripts, which write tables.
    DB_READONLY_USER  least-privilege role used by the dashboard and Ask SubscribeIQ: SELECT on
                      the five SubscribeIQ tables only, read-only transactions by default.
                      Create it with `python database/create_readonly_role.py`.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL, Engine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def get_engine(echo: bool = False, readonly: bool = False) -> Engine:
    """Build a SQLAlchemy engine for the SubscribeIQ PostgreSQL database.

    Reads DB_HOST, DB_PORT and DB_NAME, plus DB_USER / DB_PASSWORD, or DB_READONLY_USER /
    DB_READONLY_PASSWORD when readonly is True, from the environment (populated from .env).

    Args:
        echo: If True, SQLAlchemy logs every emitted SQL statement.
        readonly: Connect as the least-privilege read-only role instead of the admin role.

    Returns:
        A SQLAlchemy Engine using the psycopg2 driver.

    Raises:
        RuntimeError: If any required environment variable is missing.
    """
    user_key, password_key = (("DB_READONLY_USER", "DB_READONLY_PASSWORD") if readonly
                              else ("DB_USER", "DB_PASSWORD"))
    keys = ["DB_HOST", "DB_PORT", "DB_NAME", user_key, password_key]
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        hint = " and run `python database/create_readonly_role.py`" if readonly else ""
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)} (check .env{hint})")

    url = URL.create(
        drivername="postgresql+psycopg2",
        username=os.getenv(user_key),
        password=os.getenv(password_key),
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT")),
        database=os.getenv("DB_NAME"),
    )
    return create_engine(url, echo=echo)
