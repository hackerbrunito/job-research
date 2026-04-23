"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest

from job_research.database import DDL_DIR


@pytest.fixture
def tmp_duckdb(tmp_path: Path) -> Iterator[duckdb.DuckDBPyConnection]:
    """In-test DuckDB with the full schema applied. Yields a connection."""
    db_path = tmp_path / "test.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        for ddl_file in sorted(DDL_DIR.glob("*.sql")):
            con.execute(ddl_file.read_text(encoding="utf-8"))
        yield con
    finally:
        con.close()
