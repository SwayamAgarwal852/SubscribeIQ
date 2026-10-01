"""Create or update the least-privilege role used by the dashboard and Ask SubscribeIQ.

Usage (from the project root, after database/seed.py):
    python database/create_readonly_role.py

Connects as the admin role (DB_USER) and makes DB_READONLY_USER a login role that:
    * is not a superuser and cannot create databases or roles,
    * may connect to DB_NAME and use the public schema,
    * has SELECT on the five SubscribeIQ tables and nothing else,
    * runs every transaction read-only by default (default_transaction_read_only = on),
    * has a 30-second statement timeout.

Safe to re-run: the role is created if missing, otherwise its password and settings are reset
to the values above.
"""

import os
import sys
from pathlib import Path

from psycopg2 import sql

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.db_connection import get_engine  # noqa: E402

TABLES = ["dim_customer", "dim_service", "dim_contract", "fact_subscription", "customer_segments"]
STATEMENT_TIMEOUT = "30s"


def main() -> None:
    """Create or update the read-only role and print its effective privileges."""
    user, password = os.getenv("DB_READONLY_USER"), os.getenv("DB_READONLY_PASSWORD")
    if not user or not password:
        raise SystemExit("Set DB_READONLY_USER and DB_READONLY_PASSWORD in .env first")
    role, database = sql.Identifier(user), sql.Identifier(os.getenv("DB_NAME"))

    with get_engine().begin() as conn:
        cur = conn.connection.dbapi_connection.cursor()
        cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (user,))
        verb = "ALTER" if cur.fetchone() else "CREATE"
        cur.execute(sql.SQL(verb + " ROLE {} WITH LOGIN PASSWORD %s NOSUPERUSER NOCREATEDB "
                            "NOCREATEROLE NOREPLICATION NOBYPASSRLS").format(role), (password,))
        cur.execute(sql.SQL("ALTER ROLE {} SET default_transaction_read_only = on").format(role))
        cur.execute(sql.SQL("ALTER ROLE {} SET statement_timeout = %s").format(role),
                    (STATEMENT_TIMEOUT,))
        cur.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(database, role))
        cur.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(role))
        cur.execute(sql.SQL("REVOKE ALL ON ALL TABLES IN SCHEMA public FROM {}").format(role))
        cur.execute(sql.SQL("GRANT SELECT ON {} TO {}").format(
            sql.SQL(", ").join(sql.Identifier(t) for t in TABLES), role))
        cur.execute("""SELECT table_name, string_agg(privilege_type, ', ' ORDER BY privilege_type)
                       FROM information_schema.role_table_grants WHERE grantee = %s
                       GROUP BY table_name ORDER BY table_name""", (user,))
        grants = cur.fetchall()

    print(f"{'Created' if verb == 'CREATE' else 'Updated'} role {user}: read-only by default, "
          f"{STATEMENT_TIMEOUT} statement timeout")
    for table, privileges in grants:
        print(f"  {table}: {privileges}")


if __name__ == "__main__":
    main()
