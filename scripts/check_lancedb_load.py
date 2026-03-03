#!/usr/bin/env python3
"""Check how long it takes to load the face-clustering tables (for debugging slow UI)."""
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from config import load_settings
import lancedb

def main():
    s = load_settings()
    db = lancedb.connect(str(s.lancedb_dir))
    print("Loading images table...")
    start = time.time()
    t = db.open_table("images")
    df = t.to_pandas()
    print(f"Images: {len(df)} rows in {time.time() - start:.1f}s")
    print("Loading faces table...")
    start = time.time()
    t2 = db.open_table("faces")
    df2 = t2.to_pandas()
    print(f"Faces: {len(df2)} rows in {time.time() - start:.1f}s")

if __name__ == "__main__":
    main()
