from __future__ import annotations

import sqlite3
from pathlib import Path

from flask import current_app, g


def connect(path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path), timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    return connection


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = connect(current_app.config["DATABASE"])
    return g.db


def close_db(_error=None) -> None:
    connection = g.pop("db", None)
    if connection is not None:
        connection.close()


def init_database(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
    with connect(path) as connection:
        connection.executescript(schema)
        connection.execute("PRAGMA optimize")

