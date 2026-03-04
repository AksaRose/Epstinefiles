#!/usr/bin/env python3
"""Generate smaller thumbnails for all images under epstein_pdfs.

Thumbnails are written to an epstein_thumbs/ tree that mirrors epstein_pdfs/.
The UI will automatically prefer these thumbnails when present.
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
IMAGES_ROOT = BASE_DIR / "epstein_pdfs"
THUMBS_ROOT = BASE_DIR / "epstein_thumbs"

# Max width/height for thumbnails (preserve aspect ratio)
MAX_SIZE = 1000


def make_thumbnail(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with src.open("rb") as f:
            img = Image.open(io.BytesIO(f.read()))
            img = img.convert("RGB")
            img.thumbnail((MAX_SIZE, MAX_SIZE), Image.LANCZOS)
        # Always write JPEG thumbnails to keep them small
        dst = dst.with_suffix(".jpg")
        with dst.open("wb") as f_out:
            img.save(f_out, format="JPEG", quality=85, optimize=True)
    except Exception:
        # Skip files that can't be processed; better to continue than to crash.
        return


def main() -> int:
    if not IMAGES_ROOT.is_dir():
        print(f"Images root not found: {IMAGES_ROOT}")
        return 1

    exts = {".png", ".jpg", ".jpeg", ".webp"}
    all_files = [p for p in IMAGES_ROOT.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    total = len(all_files)
    if not total:
        print("No image files found under", IMAGES_ROOT)
        return 0

    print(f"Found {total} images under {IMAGES_ROOT}")
    for idx, src in enumerate(all_files, start=1):
        rel = src.relative_to(IMAGES_ROOT)
        dst = THUMBS_ROOT / rel
        if dst.with_suffix(".jpg").is_file():
            continue
        make_thumbnail(src, dst)
        if idx % 50 == 0:
            print(f"Generated thumbnails for {idx} / {total} images")

    print("Done generating thumbnails.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

