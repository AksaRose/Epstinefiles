#!/usr/bin/env python3
"""
Tune face-clustering config on a small set (e.g. HS only). Runs extraction once,
then tries multiple DBSCAN eps values and prints cluster stats so you can pick
the best config before running the full pipeline.

Usage (run each line separately to avoid zsh "unknown file attribute" errors):
  python scripts/tune_face_clustering.py --images-dir epstein_pdfs/HS --datasets 6
  python scripts/tune_face_clustering.py --images-dir epstein_pdfs/HS --datasets 6 --eps 0.40 0.45 0.50 0.55
  DBSCAN_EPS=0.54 python run_pipeline.py --face-clustering --images-dir epstein_pdfs/HS --datasets 6
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Run from project root so config and pipeline resolve
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import load_settings
from pipeline.cluster_faces import FaceRecord, cluster_faces
from pipeline.face_detect import extract_faces_with_embeddings


def discover_hs(base_dir: Path, dataset_id: int = 6):
    """Discover images from a directory (e.g. HS). Mirrors run_pipeline _discover_from_dir."""
    items = []
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        for path in sorted(base_dir.glob(f"*{ext}")):
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
            items.append({
                "dataset_id": dataset_id,
                "source_file": source_file,
                "page_no": page_no,
                "image_path": path,
            })
    return items


def run_extraction(items, verbose=True):
    """Extract face embeddings for all images. Returns list of FaceRecord (no cluster_id set)."""
    records = []
    for i, item in enumerate(items):
        path = item["image_path"]
        row_id = f"dataset_{item['dataset_id']}/{path.stem}"
        try:
            faces_list, _ = extract_faces_with_embeddings(path)
        except Exception as e:
            if verbose:
                print(f"  Skip {path.name}: {e}", file=sys.stderr)
            continue
        for fi, (emb, bbox) in enumerate(faces_list):
            records.append(FaceRecord(
                image_id=row_id,
                face_index=fi,
                embedding=emb,
                bbox=bbox,
            ))
        if verbose and (i + 1) % 5 == 0:
            print(f"  Processed {i + 1}/{len(items)} images, {len(records)} faces so far...")
    return records


def cluster_stats(face_records):
    """Return (num_clusters, size_distribution, num_noise). size_distribution = list of cluster sizes."""
    from collections import defaultdict
    by_cluster = defaultdict(int)
    noise = 0
    for r in face_records:
        if r.cluster_id < 0:
            noise += 1
        else:
            by_cluster[r.cluster_id] += 1
    sizes = sorted(by_cluster.values(), reverse=True)
    return len(by_cluster), sizes, noise


def main():
    parser = argparse.ArgumentParser(
        description="Tune face clustering on a small image set (e.g. HS). Prints cluster stats for different eps."
    )
    parser.add_argument("--images-dir", type=Path, default=None,
                        help="Image directory (default: <base_dir>/HS)")
    parser.add_argument("--datasets", type=int, nargs="*", default=[6],
                        help="Dataset ID(s) for display (default: 6 for HS)")
    parser.add_argument("--eps", type=float, nargs="+", default=None,
                        help="DBSCAN eps values to try (default: 0.38 0.42 0.46 0.50 0.54)")
    parser.add_argument("--min-samples", type=int, default=None,
                        help="DBSCAN min_samples (default: from config)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Less progress output")
    args = parser.parse_args()

    settings = load_settings()
    base_dir = settings.images_base_dir
    img_dir = args.images_dir or base_dir / "HS"
    img_dir = img_dir.resolve() if not img_dir.is_absolute() else img_dir
    if not img_dir.is_dir():
        print(f"Error: not a directory: {img_dir}", file=sys.stderr)
        return 1

    dataset_id = args.datasets[0] if args.datasets else 6
    items = discover_hs(img_dir, dataset_id)
    if not items:
        print(f"No images in {img_dir}", file=sys.stderr)
        return 1

    print(f"Images: {len(items)} in {img_dir}")
    print("Extracting faces (this may take a minute)...")
    face_records = run_extraction(items, verbose=not args.quiet)
    print(f"Faces: {len(face_records)}")
    if not face_records:
        print("No faces detected. Try lowering FACE_DET_THRESH (e.g. 0.25).", file=sys.stderr)
        return 1

    min_samples = args.min_samples if args.min_samples is not None else settings.dbscan_min_samples
    eps_list = args.eps if args.eps is not None else [0.46, 0.50, 0.54, 0.58, 0.62, 0.66]

    print(f"\nClustering (min_samples={min_samples}) with eps sweep:")
    print("-" * 60)

    best_eps = None
    best_score = -1

    for eps in eps_list:
        # Copy records so cluster_faces can mutate cluster_id
        copy_records = [
            FaceRecord(image_id=r.image_id, face_index=r.face_index, embedding=r.embedding, bbox=r.bbox)
            for r in face_records
        ]
        cluster_faces(copy_records, eps=eps, min_samples=min_samples)
        num_clusters, sizes, noise = cluster_stats(copy_records)

        size_summary = Counter(sizes)
        parts = [f"{c}x{k}" for k, c in sorted(size_summary.items(), key=lambda x: -x[0])[:5]]
        if len(size_summary) > 5:
            parts.append("...")
        dist_str = ", ".join(parts)

        print(f"  eps={eps:.2f}  ->  {num_clusters} clusters  (noise: {noise})  sizes: {dist_str}")

        # Prefer eps that merges same person: fewer singletons, then fewer clusters.
        num_singletons = sum(1 for s in sizes if s == 1)
        if noise > len(face_records) // 2:
            score = -1
        else:
            score = -num_singletons * 1000 - num_clusters  # higher = fewer singletons, fewer clusters
        if score > best_score:
            best_score = score
            best_eps = eps

    print("-" * 60)
    if best_eps is not None:
        print(f"\nSuggested: DBSCAN_EPS={best_eps:.2f}  (best merge of same person; set in .env)")
    print("\nThen run:")
    print("  python run_pipeline.py --face-clustering --images-dir epstein_pdfs/HS --datasets 6")
    return 0


if __name__ == "__main__":
    sys.exit(main())
