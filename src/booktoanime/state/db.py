"""SQLite connection helpers.

We intentionally use the stdlib ``sqlite3`` module synchronously rather than
``aiosqlite`` for two reasons:

1. The job database sees very low write volume (a handful of writes per stage
   transition). The cost of hopping every call to a thread pool dwarfs the
   actual SQL work.
2. ``sqlite3`` is in the stdlib; one fewer dependency to vet/license.

Connections are opened with ``check_same_thread=False`` so the orchestrator
or future ``asyncio.to_thread`` callers can use the same connection from a
worker thread. :class:`JobRepository` serializes writes with its own lock.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources
from pathlib import Path


def open_database(db_path: Path) -> sqlite3.Connection:
    """Open (or create) a SQLite database at ``db_path`` with the job schema.

    Returns a configured connection with row-as-mapping access enabled.
    Callers are expected to pass this connection into :class:`JobRepository`.
    """

    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        db_path,
        isolation_level=None,  # autocommit; transaction() drives BEGIN explicitly
        check_same_thread=False,  # JobRepository serializes via threading.Lock
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON;")
    _apply_schema(connection)
    return connection


def _apply_schema(connection: sqlite3.Connection) -> None:
    schema_sql = resources.files("booktoanime.state").joinpath("schema.sql").read_text("utf-8")
    connection.executescript(schema_sql)


@contextmanager
def transaction(connection: sqlite3.Connection) -> Iterator[None]:
    """Run the body inside a SQLite ``BEGIN IMMEDIATE`` transaction."""

    connection.execute("BEGIN IMMEDIATE;")
    try:
        yield
    except BaseException:
        connection.execute("ROLLBACK;")
        raise
    else:
        connection.execute("COMMIT;")
