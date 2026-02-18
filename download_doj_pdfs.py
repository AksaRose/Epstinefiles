#!/usr/bin/env python3
"""
Download DOJ Epstein disclosure PDFs from data sets 1-5 into local dataset_1/ ... dataset_5/.
Preserves original filenames (e.g. EFTA00000002.pdf). Uses session + age-verify flow where possible.
"""

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlencode

import requests
from bs4 import BeautifulSoup

LOG = logging.getLogger(__name__)

BASE = "https://www.justice.gov"
LISTING_URL_TEMPLATE = BASE + "/epstein/doj-disclosures/data-set-{n}-files"
# Pagination: first page is no param; then ?page=1, ?page=2, ...
PDF_BASE_PATTERN = re.compile(r"/epstein/files/DataSet%20(\d+)/EFTA\d+\.pdf$", re.I)
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
DELAY_SEC = 0.8
MAX_RETRIES = 2
RETRY_BACKOFF_SEC = 2.0
PDF_MAGIC = b"%PDF"


# Cookie set by DOJ after age verification (browser sends justiceGovAgeVerified=true).
AGE_VERIFIED_COOKIE = "justiceGovAgeVerified=true"


def _session(cookie_override: str | None) -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    s.headers["Accept"] = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    s.headers["Accept-Language"] = "en-US,en;q=0.9"
    if cookie_override:
        s.headers["Cookie"] = cookie_override.strip()
    else:
        s.headers["Cookie"] = AGE_VERIFIED_COOKIE
    return s


def _try_age_verify(session: requests.Session) -> None:
    """Attempt to pass age gate by requesting age-verify with destination to Epstein library."""
    destination = "/epstein/doj-disclosures/data-set-1-files"
    url = BASE + "/age-verify?" + urlencode({"destination": destination})
    try:
        r = session.get(url, timeout=30, allow_redirects=True)
        r.raise_for_status()
    except requests.RequestException as e:
        LOG.warning("Age-verify request failed: %s. Downloads may get HTML instead of PDF.", e)


def _pdf_links_from_page(soup: BeautifulSoup, dataset_n: int) -> list[tuple[str, str]]:
    """Return list of (filename, href) for links that point to PDFs in DataSet {dataset_n}."""
    out = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href.endswith(".pdf"):
            continue
        # Normalize: could be relative or absolute
        full = urljoin(BASE + "/", href)
        m = PDF_BASE_PATTERN.search(full)
        if not m:
            continue
        if int(m.group(1)) != dataset_n:
            continue
        filename = href.rsplit("/", 1)[-1].split("?")[0]
        if filename.endswith(".pdf"):
            out.append((filename, full))
    return out


def scrape_dataset(session: requests.Session, dataset_n: int) -> list[tuple[int, str, str]]:
    """Scrape all listing pages for data set N. Returns list of (dataset_n, filename, pdf_url)."""
    results: list[tuple[int, str, str]] = []
    page = 0
    seen_urls: set[str] = set()

    while True:
        if page == 0:
            url = LISTING_URL_TEMPLATE.format(n=dataset_n)
        else:
            url = LISTING_URL_TEMPLATE.format(n=dataset_n) + "?page=" + str(page)
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
        except requests.RequestException as e:
            LOG.error("Failed to fetch listing %s: %s", url, e)
            break
        soup = BeautifulSoup(r.text, "lxml")
        links = _pdf_links_from_page(soup, dataset_n)
        if not links:
            break
        added = 0
        for filename, pdf_url in links:
            if pdf_url in seen_urls:
                continue
            seen_urls.add(pdf_url)
            results.append((dataset_n, filename, pdf_url))
            added += 1
        LOG.info("Data set %d page %d: %d links (total %d)", dataset_n, page, added, len(results))
        if added == 0:
            break
        page += 1
        time.sleep(DELAY_SEC)

    return results


def scrape_all(session: requests.Session) -> list[tuple[int, str, str]]:
    """Scrape data sets 1-5. Returns list of (dataset_n, filename, pdf_url)."""
    all_entries = []
    for n in range(1, 6):
        entries = scrape_dataset(session, n)
        all_entries.extend(entries)
        if n < 5:
            time.sleep(DELAY_SEC)
    return all_entries


def is_pdf_content(data: bytes, content_type: str | None) -> bool:
    if data.startswith(PDF_MAGIC):
        return True
    if content_type and "pdf" in content_type.lower():
        return True
    return False


def download_one(
    session: requests.Session,
    dataset_n: int,
    filename: str,
    pdf_url: str,
    out_dir: Path,
    skip_existing: bool,
) -> bool:
    out_path = out_dir / f"dataset_{dataset_n}" / filename
    if skip_existing and out_path.exists():
        LOG.debug("Skip (exists): %s", out_path)
        return True
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = session.get(pdf_url, timeout=60, stream=True)
            r.raise_for_status()
            data = r.content
            ct = r.headers.get("Content-Type")
            if not is_pdf_content(data, ct):
                LOG.warning("Not PDF (dataset %d %s): Content-Type=%s, first bytes=%s",
                            dataset_n, filename, ct, data[:20] if len(data) >= 20 else data)
                return False
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(data)
            LOG.debug("Saved: %s", out_path)
            return True
        except requests.RequestException as e:
            LOG.warning("Attempt %d for %s: %s", attempt + 1, filename, e)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SEC)
    return False


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description="Download DOJ Epstein disclosure PDFs (data sets 1-5).")
    p.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("epstein_pdfs"),
        help="Base directory for dataset_1/ ... dataset_5/ (default: epstein_pdfs)",
    )
    p.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-download even if file already exists",
    )
    p.add_argument(
        "--cookie",
        type=str,
        default=None,
        help="Optional Cookie header value (e.g. from browser after passing age verification)",
    )
    p.add_argument(
        "--list-only",
        action="store_true",
        help="Only scrape and print PDF list; do not download",
    )
    args = p.parse_args()

    cookie = args.cookie or __import__("os").environ.get("DOJ_COOKIE")
    session = _session(cookie)
    _try_age_verify(session)
    time.sleep(DELAY_SEC)

    LOG.info("Scraping listing pages for data sets 1-5...")
    entries = scrape_all(session)
    LOG.info("Total PDFs to download: %d", len(entries))
    if not entries:
        LOG.error("No PDF links found. Check age verification (see README).")
        return 1

    if args.list_only:
        for n, name, url in entries:
            print(f"dataset_{n}\t{name}\t{url}")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    fail = 0
    for n, name, url in entries:
        if download_one(session, n, name, url, args.output_dir, skip_existing=not args.no_skip_existing):
            ok += 1
        else:
            fail += 1
        time.sleep(DELAY_SEC)

    LOG.info("Done: %d saved, %d failed", ok, fail)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
