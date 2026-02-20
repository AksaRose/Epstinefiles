#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import logging
from pathlib import Path

import lancedb
import numpy as np

from config import load_settings
from pipeline.embed import embed_query


LOG = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    settings = load_settings()

    p = argparse.ArgumentParser(description="Search Epstein image captions in LanceDB.")
    p.add_argument("query", type=str, help="Text query, e.g. 'Epstein island photos'")
    p.add_argument("-k", type=int, default=10, help="Number of results")
    p.add_argument(
        "--save-images-dir",
        type=Path,
        default=None,
        help="Optional directory to save returned images as PNGs",
    )
    args = p.parse_args()

    vec = embed_query(
        args.query,
        api_key=settings.together_api_key,
        model=settings.embed_model,
    )

    db = lancedb.connect(str(settings.lancedb_dir))
    table = db.open_table(settings.table_name)

    vec_arr = np.array(vec, dtype="float32")
    try:
        q = (
            table.search(query_type="hybrid", vector_column_name="vector")
            .vector(vec_arr)
            .text(args.query)
        )
        results = q.limit(args.k).to_pandas()
    except Exception:
        results = (
            table.search(vec_arr)
            .limit(args.k)
            .to_pandas()
        )

    if results.empty:
        LOG.info("No results.")
        return 0

    if args.save_images_dir:
        args.save_images_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in results.iterrows():
        print(f"[{idx}] score={row.get('_distance', 'n/a')} id={row['id']}")
        print(f"  dataset={row['dataset_id']} file={row['source_file']} page={row['page_no']}")
        celebs = row.get("celebrities") or []
        if celebs:
            print(f"  celebrities: {', '.join(celebs)}")
        print(f"  caption: {row['caption']}")

        blob = row["image_blob"]
        if args.save_images_dir and isinstance(blob, (bytes, bytearray)):
            out_path = args.save_images_dir / f"result_{idx}.png"
            out_path.write_bytes(blob)
            print(f"  image saved to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

