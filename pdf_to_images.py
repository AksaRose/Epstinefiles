#!/usr/bin/env python3
"""
Render PDF pages to PNG images for Vision Transformer (ViT) input.
Walks dataset_1/ ... dataset_5/ (or a given base dir) and writes images to
dataset_1_images/ ... dataset_5_images/ (or alongside each PDF).
Standard ViTs expect raster images, not PDFs; use this script before feeding data to a ViT.
"""

import argparse
import logging
import sys
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

LOG = logging.getLogger(__name__)


def pdf_to_images_pymupdf(pdf_path: Path, out_dir: Path, dpi: int = 150) -> int:
    """Render each page of the PDF to a PNG in out_dir. Returns number of images written."""
    if fitz is None:
        raise RuntimeError("PyMuPDF is required: pip install pymupdf")
    doc = fitz.open(pdf_path)
    count = 0
    stem = pdf_path.stem
    for i in range(len(doc)):
        page = doc[i]
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        out_path = out_dir / f"{stem}_page{i + 1:04d}.png"
        pix.save(str(out_path))
        count += 1
    doc.close()
    return count


def process_dataset_dir(
    dataset_dir: Path,
    images_dir: Path,
    dpi: int,
) -> tuple[int, int]:
    """Process all PDFs in dataset_dir, write images to images_dir. Returns (pdfs_ok, total_pages)."""
    images_dir.mkdir(parents=True, exist_ok=True)
    pdfs_ok = 0
    total_pages = 0
    for pdf_path in sorted(dataset_dir.glob("*.pdf")):
        try:
            n = pdf_to_images_pymupdf(pdf_path, images_dir, dpi=dpi)
            pdfs_ok += 1
            total_pages += n
            LOG.debug("%s -> %d pages", pdf_path.name, n)
        except Exception as e:
            LOG.warning("Failed %s: %s", pdf_path, e)
    return pdfs_ok, total_pages


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if fitz is None:
        LOG.error("PyMuPDF not installed. Run: pip install pymupdf")
        return 1

    p = argparse.ArgumentParser(
        description="Render DOJ Epstein PDFs to PNG images for ViT input."
    )
    p.add_argument(
        "base_dir",
        type=Path,
        nargs="?",
        default=Path("epstein_pdfs"),
        help="Base directory containing dataset_1/ ... dataset_5/ (default: epstein_pdfs)",
    )
    p.add_argument(
        "-d", "--dpi",
        type=int,
        default=150,
        help="Resolution for rendered pages (default: 150)",
    )
    p.add_argument(
        "-o", "--output-base",
        type=Path,
        default=None,
        help="Base dir for image folders (default: base_dir; writes dataset_N_images/)",
    )
    args = p.parse_args()

    out_base = args.output_base or args.base_dir
    total_pdfs = 0
    total_pages = 0

    for n in range(1, 6):
        ds_dir = args.base_dir / f"dataset_{n}"
        if not ds_dir.is_dir():
            LOG.warning("Skipping missing directory: %s", ds_dir)
            continue
        img_dir = out_base / f"dataset_{n}_images"
        LOG.info("Processing dataset_%d -> %s", n, img_dir)
        ok, pages = process_dataset_dir(ds_dir, img_dir, args.dpi)
        total_pdfs += ok
        total_pages += pages
        LOG.info("  %d PDFs, %d pages", ok, pages)

    LOG.info("Total: %d PDFs, %d images", total_pdfs, total_pages)
    return 0


if __name__ == "__main__":
    sys.exit(main())
