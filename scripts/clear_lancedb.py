#!/usr/bin/env python3
"""
Remove all data from the LanceDB: drop every table and delete cluster JSON files.
Use this to start fresh before re-running the pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Run from project root
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import load_settings
from pipeline import face_store


def _table_name(item) -> str | None:
    """Get table name string from list_tables() result (may be str, tuple, or nested list)."""
    if isinstance(item, str):
        return item
    if isinstance(item, (list, tuple)) and len(item) > 0:
        return _table_name(item[-1])
    return None


def main() -> int:
    settings = load_settings()
    lancedb_dir = Path(settings.lancedb_dir)
    if not lancedb_dir.exists():
        print("LanceDB dir does not exist:", lancedb_dir)
        return 0

    import lancedb
    db = lancedb.connect(str(lancedb_dir))
    tables = db.list_tables()
    for item in tables:
        name = _table_name(item)
        if not name:
            continue
        db.drop_table(name)
        print("Dropped table:", name)
    if not tables:
        print("No tables found.")

    for path in (face_store._cluster_reps_path(lancedb_dir), face_store._cluster_names_path(lancedb_dir)):
        if path.exists():
            path.unlink()
            print("Removed:", path)

    print("Done. DB is empty. Re-run the pipeline to repopulate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
