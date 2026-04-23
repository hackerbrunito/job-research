"""DuckDB access layer.

Exposes:
- `connect()` — returns a DuckDB connection configured from settings.
- `init_schema()` — runs every file under `sql/ddl/` in order. Idempotent.
- `insert_dataframe()` — inserts a pandas DataFrame into a named table.
- Small helpers used across the pipeline (hashes for deterministic keys,
  run-registry updates).

All read queries should go through an explicit `connect()` call — never use
the module-level `duckdb.sql(...)` API, which is not thread-safe.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from job_research.config import get_settings
from job_research.logging_setup import get_logger

log = get_logger(__name__)

# Directory layout: repo_root/sql/ddl/*.sql — resolved relative to this file
# so it works whether the app is run from repo root, tests, or an installed
# wheel (where sql/ is shipped alongside the package, see note below).
_REPO_ROOT = Path(__file__).resolve().parents[2]
DDL_DIR = _REPO_ROOT / "sql" / "ddl"
DML_DIR = _REPO_ROOT / "sql" / "dml"


# --------------------------------------------------------------------------- #
# Connection
# --------------------------------------------------------------------------- #
@contextmanager
def connect(*, read_only: bool | None = None) -> Iterator[duckdb.DuckDBPyConnection]:
    """Open a DuckDB connection using settings; yield it as a context manager."""
    settings = get_settings()
    db_path = settings.database.path
    ro = settings.database.read_only if read_only is None else read_only

    if not ro:
        db_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path), read_only=ro)
    try:
        yield con
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# Schema bootstrap
# --------------------------------------------------------------------------- #
def init_schema(con: duckdb.DuckDBPyConnection | None = None) -> None:
    """Execute every `sql/ddl/*.sql` in sorted order. Idempotent."""
    if not DDL_DIR.is_dir():
        raise FileNotFoundError(f"DDL directory not found: {DDL_DIR}")

    ddl_files = sorted(DDL_DIR.glob("*.sql"))
    if not ddl_files:
        raise RuntimeError(f"No DDL files found in {DDL_DIR}")

    def _run(c: duckdb.DuckDBPyConnection) -> None:
        for path in ddl_files:
            log.info("ddl.apply", file=path.name)
            c.execute(path.read_text(encoding="utf-8"))

    if con is None:
        with connect() as c:
            _run(c)
    else:
        _run(con)


def load_sql(name: str, *, kind: str = "dml") -> str:
    """Read a SQL file by name from sql/<kind>/. Example: load_sql('refresh_marts')."""
    base = {"ddl": DDL_DIR, "dml": DML_DIR}.get(kind)
    if base is None:
        raise ValueError(f"kind must be 'ddl' or 'dml', got {kind!r}")
    path = base / (name if name.endswith(".sql") else f"{name}.sql")
    if not path.is_file():
        raise FileNotFoundError(f"SQL file not found: {path}")
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Insert helpers
# --------------------------------------------------------------------------- #
def insert_dataframe(
    con: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    table: str,
    *,
    by_name: bool = True,
) -> int:
    """Insert a DataFrame into an existing table. Returns row count."""
    if df.empty:
        return 0
    con.register("_incoming_df", df)
    try:
        clause = "BY NAME " if by_name else ""
        # Table name is caller-supplied from internal code only, never user input.
        con.execute(f"INSERT INTO {table} {clause}SELECT * FROM _incoming_df")  # noqa: S608
    finally:
        con.unregister("_incoming_df")
    return len(df)


# --------------------------------------------------------------------------- #
# Deterministic keys
# --------------------------------------------------------------------------- #
def stable_key(*parts: Any) -> str:
    """Deterministic short hash for surrogate keys (location, salary, skill)."""
    joined = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.blake2b(joined.encode("utf-8"), digest_size=12).hexdigest()


def job_id(site: str, job_url: str) -> str:
    """Deterministic per-row id for staging — survives re-scrapes."""
    return stable_key(site, job_url)


# --------------------------------------------------------------------------- #
# Pipeline run registry
# --------------------------------------------------------------------------- #
def record_run_start(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    *,
    keywords: list[str],
    locations: list[str],
    sites: list[str],
) -> None:
    con.execute(
        """
        INSERT INTO pipeline_runs (run_id, started_at, status, keywords, locations, sites)
        VALUES (?, ?, 'running', ?, ?, ?)
        """,
        [run_id, datetime.now(UTC).replace(tzinfo=None), keywords, locations, sites],
    )


def record_run_finish(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    *,
    status: str,
    scraped_count: int,
    enriched_count: int,
    error_message: str | None = None,
) -> None:
    con.execute(
        """
        UPDATE pipeline_runs
        SET finished_at    = ?,
            status         = ?,
            scraped_count  = ?,
            enriched_count = ?,
            error_message  = ?
        WHERE run_id = ?
        """,
        [
            datetime.now(UTC).replace(tzinfo=None),
            status,
            scraped_count,
            enriched_count,
            error_message,
            run_id,
        ],
    )
