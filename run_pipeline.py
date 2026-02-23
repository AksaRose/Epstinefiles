#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Set

from config import load_settings

# --fts-only: rebuild FTS index only (no cv2/opencv needed; avoids libGL on headless servers)
if "--fts-only" in sys.argv:
    import os
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--fts-only", action="store_true")
    _args, _ = _parser.parse_known_args()
    if _args.fts_only:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        _settings = load_settings()
        import lancedb
        _db = lancedb.connect(str(_settings.lancedb_dir))
        if _settings.table_name not in _db.list_tables():
            logging.error("Table %s does not exist. Run the pipeline first.", _settings.table_name)
            os._exit(1)
        _table = _db.open_table(_settings.table_name)
        try:
            _table.create_fts_index("searchable_text", replace=True, with_position=True, remove_stop_words=False)
            logging.info("FTS index on 'searchable_text' rebuilt (phrase queries enabled). Hybrid search will include all rows.")
        except Exception as e:
            logging.exception("Could not create FTS index: %s", e)
            os._exit(1)
        os._exit(0)

import lancedb
from pipeline.caption import caption_image
from pipeline.embed import embed_texts
from pipeline.face_detect import FaceAnalysisResult, analyze_image
from pipeline.lance_writer import get_or_create_table, write_batch


LOG = logging.getLogger(__name__)


def _item_id(item: Dict) -> str:
    """Same row id as in process_chunk: dataset_{id}/{path.stem}."""
    return f"dataset_{item['dataset_id']}/{item['image_path'].stem}"


# Image extensions supported by face_detect (cv2) and pipeline
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def _discover_from_dir(img_dir: Path, dataset_id: int) -> List[Dict]:
    """Discover images from a single directory. Uses dataset_id for all items.
    Accepts .png, .jpg, .jpeg, .bmp. Handles both *_page0001 naming and plain filenames (e.g. HOUSE_OVERSIGHT_083584.jpg).
    """
    items: List[Dict] = []
    paths: List[Path] = []
    for ext in _IMAGE_EXTENSIONS:
        paths.extend(img_dir.glob(f"*{ext}"))
    for path in sorted(set(paths)):
        stem = path.stem
        if "_page" in stem:
            try:
                source, page_part = stem.split("_page", 1)
                page_no = int(page_part)
            except ValueError:
                source, page_no = stem, 1
            source_file = f"{source}.pdf"
        else:
            source_file = f"{stem}{path.suffix}"
            page_no = 1
        items.append(
            {
                "dataset_id": dataset_id,
                "source_file": source_file,
                "page_no": page_no,
                "image_path": path,
            }
        )
    return items


def discover_images(base_dir: Path) -> List[Dict]:
    items: List[Dict] = []
    for dataset_id in range(1, 6):
        img_dir = base_dir / f"dataset_{dataset_id}_images"
        if not img_dir.is_dir():
            # Accept dataset_N_images2, dataset_N_images_extra, etc.
            candidates = sorted(p for p in base_dir.glob(f"dataset_{dataset_id}_images*") if p.is_dir())
            if not candidates:
                LOG.warning("Missing images directory: dataset_%d_images (or dataset_%d_images*)", dataset_id, dataset_id)
                continue
            img_dir = candidates[0]
        for path in sorted(img_dir.glob("*.png")):
            stem = path.stem  # e.g. EFTA00000001_page0001
            if "_page" not in stem:
                continue
            source, page_part = stem.split("_page", 1)
            try:
                page_no = int(page_part)
            except ValueError:
                page_no = 0
            items.append(
                {
                    "dataset_id": dataset_id,
                    "source_file": f"{source}.pdf",
                    "page_no": page_no,
                    "image_path": path,
                }
            )
    return items


def process_chunk(
    chunk: List[Dict],
    settings,
) -> List[Dict]:
    rows: List[Dict] = []

    # Step 1: face detection + local recognition + image bytes
    face_results: Dict[Path, FaceAnalysisResult] = {}
    image_bytes_map: Dict[Path, bytes] = {}
    for item in chunk:
        path: Path = item["image_path"]
        result, img_bytes = analyze_image(
            path,
            faces_db_dir=settings.faces_db_dir,
            distance_threshold=settings.face_distance_threshold,
            backend=settings.face_backend,
        )
        face_results[path] = result
        image_bytes_map[path] = img_bytes

    # Step 2: caption with Qwen (parallel for this chunk)
    captions: Dict[Path, str] = {}
    with ThreadPoolExecutor(max_workers=settings.qwen_concurrency) as ex:
        fut_to_path = {}
        for item in chunk:
            path: Path = item["image_path"]
            face = face_results[path]
            fut = ex.submit(
                caption_image,
                path,
                celebrity_names=face.celebrities,
                api_key=settings.together_api_key,
                model=settings.qwen_model,
                max_tokens=settings.caption_max_tokens,
            )
            fut_to_path[fut] = path

        for fut in as_completed(fut_to_path):
            path = fut_to_path[fut]
            try:
                captions[path] = fut.result()
            except Exception as e:  # noqa: BLE001
                LOG.warning("Caption failed for %s: %s", path, e)
                captions[path] = ""

    # Step 3: prepare texts for embedding (caption from LLM; people only from InsightFace)
    texts: List[str] = []
    for item in chunk:
        path: Path = item["image_path"]
        face = face_results[path]
        caption = captions.get(path, "").strip()
        if face.celebrities:
            celeb_part = ", ".join(face.celebrities)
            full = f"{caption} (People identified: {celeb_part})" if caption else f"People identified: {celeb_part}"
        else:
            full = caption or "Document image."
        texts.append(full)

    # Step 4: batch embed
    vectors = embed_texts(
        texts,
        api_key=settings.together_api_key,
        model=settings.embed_model,
    )

    # Step 5: build rows for LanceDB
    for idx, item in enumerate(chunk):
        path: Path = item["image_path"]
        face = face_results[path]
        caption = captions.get(path, "").strip()
        vec = vectors[idx]
        img_bytes = image_bytes_map[path]
        row_id = f"dataset_{item['dataset_id']}/{path.stem}"
        # searchable_text = caption + celebrity names for keyword/FTS in hybrid search
        celeb_part = " ".join(face.celebrities) if face.celebrities else ""
        searchable_text = f"{caption} {celeb_part}".strip() or ""
        rows.append(
            {
                "id": row_id,
                "dataset_id": int(item["dataset_id"]),
                "source_file": item["source_file"],
                "page_no": int(item["page_no"]),
                "image_path": str(path),
                "image_blob": img_bytes,
                "caption": caption,
                "celebrities": face.celebrities,
                "has_faces": bool(face.has_faces),
                "searchable_text": searchable_text,
                "vector": vec,
            }
        )

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Index images: face recognition (InsightFace), caption (Qwen), embed, write to LanceDB.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing table (clear and re-index from scratch)")
    parser.add_argument("--datasets", type=int, nargs="*", default=None, metavar="N", help="Only process these dataset IDs (e.g. --datasets 2). If omitted, process all 1-5.")
    parser.add_argument("--images-dir", type=Path, default=None, metavar="PATH", help="Only process this folder (e.g. epstein_pdfs/dataset_2_images2). Dataset ID inferred from path or use with --datasets 2.")
    parser.add_argument("--fts-only", action="store_true", help="Only rebuild the FTS index on searchable_text (no image processing). Use if hybrid search failed or index was missing.")
    parser.add_argument("--fill-empty-only", action="store_true", help="Only process images that have no caption yet (skip already-captioned; re-caption and update rows with empty caption). Ignored if --overwrite.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    settings = load_settings()

    if args.images_dir is not None:
        img_dir = args.images_dir.resolve()
        if not img_dir.is_dir():
            LOG.error("Not a directory: %s", img_dir)
            return 1
        # Infer dataset_id from path (e.g. dataset_2_images2 -> 2)
        match = re.search(r"dataset_(\d+)", img_dir.name)
        dataset_id = int(match.group(1)) if match else 2
        if args.datasets is not None and len(args.datasets) == 1:
            dataset_id = args.datasets[0]
        items = _discover_from_dir(img_dir, dataset_id)
        LOG.info("Using only %s (dataset_id=%d): %d images", img_dir, dataset_id, len(items))
    else:
        LOG.info("Using images base dir: %s", settings.images_base_dir)
        items = discover_images(settings.images_base_dir)
        if args.datasets is not None:
            items = [x for x in items if x["dataset_id"] in args.datasets]
            LOG.info("Filtered to dataset(s) %s: %d images", args.datasets, len(items))
        else:
            LOG.info("Discovered %d images", len(items))
    if not items:
        LOG.error("No images found. Run pdf_to_images.py first.")
        return 1

    # Prime embeddings to infer vector dimension
    probe_vec = embed_texts(
        ["probe text"],
        api_key=settings.together_api_key,
        model=settings.embed_model,
    )[0]
    vector_dim = len(probe_vec)
    table = get_or_create_table(settings.lancedb_dir, settings.table_name, vector_dim, overwrite=args.overwrite)

    # Fill-empty-only: skip already-captioned images; process only new or empty-caption rows (then we update by delete+add).
    if args.fill_empty_only and not args.overwrite:
        try:
            df = table.to_pandas()
        except Exception as e:
            LOG.warning("Could not scan table for --fill-empty-only, processing all items: %s", e)
            df = None
        if df is not None and len(df) > 0:
            existing_with_caption: Set[str] = set()
            existing_empty_ids: List[str] = []
            for _, row in df.iterrows():
                rid = row.get("id")
                cap = row.get("caption")
                if rid is None:
                    continue
                if cap is not None and str(cap).strip():
                    existing_with_caption.add(rid)
                else:
                    existing_empty_ids.append(rid)
            items = [i for i in items if _item_id(i) not in existing_with_caption]
            LOG.info("Fill-empty-only: skipping %d already-captioned; (re)processing %d (new or empty caption)", len(existing_with_caption), len(items))
            # Delete only empty-caption rows we are about to refill (same id); don't delete empties whose images aren't in this run.
            ids_to_process: Set[str] = {_item_id(i) for i in items}
            ids_to_delete = [rid for rid in existing_empty_ids if rid in ids_to_process]
            if ids_to_delete:
                delete_batch_size = 200
                for j in range(0, len(ids_to_delete), delete_batch_size):
                    batch_ids = ids_to_delete[j : j + delete_batch_size]
                    in_clause = ",".join("'" + str(rid).replace("'", "''") + "'" for rid in batch_ids)
                    try:
                        table.delete(where=f"id IN ({in_clause})")
                    except Exception as e:
                        LOG.warning("Delete batch failed (continuing): %s", e)
                LOG.info("Deleted %d rows with empty caption for re-captioning.", len(ids_to_delete))
        else:
            LOG.info("Fill-empty-only: table empty or unreadable, processing all %d items.", len(items))

    batch_size = settings.embed_batch_size
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        LOG.info("Processing images %d-%d", i + 1, min(len(items), i + batch_size))
        rows = process_chunk(chunk, settings)
        write_batch(table, rows)

    # Build or rebuild FTS index on searchable_text (caption + celebrity names) for hybrid search (vector + keyword/BM25)
    # with_position=True required for phrase queries (e.g. "donald trump")
    try:
        table.create_fts_index("searchable_text", replace=True, with_position=True, remove_stop_words=False)
        LOG.info("FTS index on 'searchable_text' created/updated for hybrid search (phrase queries enabled).")
    except Exception as e:
        LOG.warning("Could not create FTS index (hybrid search may be vector-only): %s", e)

    LOG.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

