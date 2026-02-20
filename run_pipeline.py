#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from config import load_settings
from pipeline.caption import caption_image
from pipeline.embed import embed_texts
from pipeline.face_detect import FaceAnalysisResult, analyze_image
from pipeline.lance_writer import get_or_create_table, write_batch


LOG = logging.getLogger(__name__)


def discover_images(base_dir: Path) -> List[Dict]:
    items: List[Dict] = []
    for dataset_id in range(1, 6):
        img_dir = base_dir / f"dataset_{dataset_id}_images"
        if not img_dir.is_dir():
            LOG.warning("Missing images directory: %s", img_dir)
            continue
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    settings = load_settings()
    LOG.info("Using images base dir: %s", settings.images_base_dir)

    items = discover_images(settings.images_base_dir)
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

    batch_size = settings.embed_batch_size
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        LOG.info("Processing images %d-%d", i + 1, min(len(items), i + batch_size))
        rows = process_chunk(chunk, settings)
        write_batch(table, rows)

    # Build FTS index on searchable_text (caption + celebrity names) for hybrid search (vector + keyword/BM25)
    try:
        table.create_fts_index("searchable_text")
        LOG.info("FTS index on 'searchable_text' created for hybrid search.")
    except Exception as e:
        LOG.warning("Could not create FTS index (hybrid search may be vector-only): %s", e)

    LOG.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

