#!/usr/bin/env python3
"""
Convert .avif files in a directory to .jpg (or .png).
Requires: pip install pillow-heif
Usage:
  python convert_avif_to_jpg.py [DIR]           # default: epstein_pdfs/HS
  python convert_avif_to_jpg.py epstein_pdfs/HS --format png
  python convert_avif_to_jpg.py epstein_pdfs/HS --remove   # delete .avif after convert
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import pillow_heif
    from PIL import Image
except ImportError as e:
    print("Install: pip install pillow-heif Pillow", file=sys.stderr)
    raise SystemExit(1) from e

pillow_heif.register_heif_opener()


def main() -> int:
    p = argparse.ArgumentParser(description="Convert .avif (and .heif) files to .jpg or .png.")
    p.add_argument("dir", type=Path, nargs="?", default=Path("epstein_pdfs/HS"), help="Directory to scan")
    p.add_argument("--format", choices=("jpg", "jpeg", "png"), default="jpg", help="Output format (default: jpg)")
    p.add_argument("--remove", action="store_true", help="Remove .avif/.heif after successful convert")
    p.add_argument("-q", "--quality", type=int, default=90, help="JPG quality 1-100 (default: 90)")
    args = p.parse_args()

    d = args.dir.resolve()
    if not d.is_dir():
        print(f"Not a directory: {d}", file=sys.stderr)
        return 1

    ext = ".jpg" if args.format in ("jpg", "jpeg") else ".png"
    paths = list(d.glob("*.avif")) + list(d.glob("*.heif"))
    if not paths:
        print(f"No .avif or .heif files in {d}")
        return 0

    ok, err = 0, 0
    for path in sorted(paths):
        stem = path.stem  # e.g. HOUSE_OVERSIGHT_065953 or HOUSE_OVERSIGHT_065953.jpg for .jpg.avif
        if stem.endswith((".jpg", ".jpeg", ".png")):
            out = path.parent / stem  # .jpg.avif -> .jpg
        else:
            out = path.parent / f"{stem}{ext}"
        try:
            img = Image.open(path)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            if ext == ".jpg":
                img.save(out, "JPEG", quality=args.quality)
            else:
                img.save(out, "PNG")
            ok += 1
            print(f"  {path.name} -> {out.name}")
            if args.remove:
                path.unlink()
        except Exception as e:
            err += 1
            print(f"  FAIL {path.name}: {e}", file=sys.stderr)

    print(f"Done: {ok} converted, {err} failed.")
    return 1 if err else 0


if __name__ == "__main__":
    raise SystemExit(main())
