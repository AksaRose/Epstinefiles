#!/usr/bin/env python3
"""Streamlit UI to search the Epstein image index (LanceDB)."""

from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

import lancedb
import numpy as np
import streamlit as st

from config import load_settings
from pipeline.embed import embed_query


def _get_download_url() -> str | None:
    """LANCEDB_DOWNLOAD_URL from env or Streamlit secrets (for cloud deploy)."""
    url = os.environ.get("LANCEDB_DOWNLOAD_URL")
    if url:
        return url.strip() or None
    try:
        return st.secrets.get("LANCEDB_DOWNLOAD_URL") or None
    except Exception:
        return None


def _ensure_lancedb(settings) -> bool:
    """
    If LanceDB dir is missing and LANCEDB_DOWNLOAD_URL is set, download zip and unzip.
    Zip must contain the table at top level (e.g. epstein_images.lance/).
    """
    table_lance = settings.lancedb_dir / f"{settings.table_name}.lance"
    if table_lance.exists():
        return True

    url = _get_download_url()
    if not url:
        return False

    settings.lancedb_dir.mkdir(parents=True, exist_ok=True)
    with st.spinner("Downloading database (first run or cold start). This may take a few minutes."):
        try:
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
                tmp = f.name
            try:
                with urlopen(url) as resp:
                    chunk_size = 1 << 20  # 1 MB
                    total = 0
                    progress = st.progress(0.0, text="Downloading...")
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        with open(tmp, "ab") as out:
                            out.write(chunk)
                        total += len(chunk)
                        # Update progress (we don't know total size)
                        progress.progress(min(1.0, total / (100 * chunk_size)), text=f"Downloaded {total // (1 << 20)} MB")
                    progress.progress(1.0, text="Extracting...")
                with zipfile.ZipFile(tmp, "r") as zf:
                    zf.extractall(settings.lancedb_dir)
            finally:
                try:
                    Path(tmp).unlink(missing_ok=True)
                except Exception:
                    pass
            return True
        except Exception as e:
            st.error(f"Failed to download database: {e}")
            return False


@st.cache_resource
def get_table():
    settings = load_settings()
    if not _ensure_lancedb(settings):
        raise FileNotFoundError(
            "LanceDB not found. Run the pipeline locally (python run_pipeline.py) or set "
            "LANCEDB_DOWNLOAD_URL to a zip of the lancedb folder for cloud deploy."
        )
    db = lancedb.connect(str(settings.lancedb_dir))
    return db.open_table(settings.table_name), settings


def search(query: str, k: int = 12, dataset_id: int | None = None):
    settings = load_settings()
    table, _ = get_table()
    vec = embed_query(
        query,
        api_key=settings.together_api_key,
        model=settings.embed_model,
    )
    q = table.search(np.array(vec, dtype="float32"))
    if dataset_id is not None:
        q = q.where(f"dataset_id = {int(dataset_id)}")
    results = q.limit(k).to_pandas()
    return results


def main():
    st.set_page_config(page_title="Epstein Image Search", layout="wide")
    st.title("Epstein case files")
    st.caption("Search by keyword (e.g. island, party, Epstein, Maxwell, house). Results are ranked by semantic similarity.")

    try:
        table, settings = get_table()
    except Exception as e:
        st.error(f"Cannot open index: {e}. Run the pipeline first (`python run_pipeline.py`).")
        return

    query = st.text_input("Search", placeholder="e.g. Epstein island photos, people at a party")
    col_k, col_ds, _ = st.columns([1, 1, 3])
    with col_k:
        k = st.number_input("Number of results", min_value=1, max_value=50, value=12)
    with col_ds:
        dataset_filter = st.selectbox(
            "Dataset",
            ["All", "1", "2", "3", "4", "5"],
            help="Filter by DOJ disclosure dataset",
        )
    dataset_id = None if dataset_filter == "All" else int(dataset_filter)

    if not query.strip():
        st.info("Enter a search query above.")
        return

    if not settings.together_api_key:
        st.error("Set TOGETHER_API_KEY in .env to run search (embedding the query).")
        return

    with st.spinner("Searching..."):
        try:
            results = search(query, k=k, dataset_id=dataset_id)
        except Exception as e:
            st.exception(e)
            return

    if results is None or results.empty:
        st.warning("No results.")
        return

    # Grid of results: 3 per row
    ncols = 3
    for start in range(0, len(results), ncols):
        row_results = results.iloc[start : start + ncols]
        cols = st.columns(ncols)
        for i, (_, row) in enumerate(row_results.iterrows()):
            with cols[i]:
                blob = row.get("image_blob")
                if blob is not None and isinstance(blob, (bytes, bytearray)):
                    st.image(blob, use_container_width=True)
                st.markdown(f"**{row.get('source_file', '')}** p.{row.get('page_no', '')}")
                _celeb = row.get("celebrities")
                if _celeb is None:
                    celebs = []
                elif isinstance(_celeb, np.ndarray):
                    celebs = _celeb.tolist()
                elif isinstance(_celeb, (list, tuple)):
                    celebs = list(_celeb)
                else:
                    celebs = [_celeb] if _celeb is not None else []
                if celebs:
                    st.caption(f"People: {', '.join(str(c) for c in celebs)}")
                cap = row.get("caption")
                cap = "" if cap is None else str(cap)
                if cap:
                    st.caption(cap[:200] + ("..." if len(cap) > 200 else ""))
                st.caption(f"Dataset {row.get('dataset_id', '')} · distance {row.get('_distance', ''):.3f}")


if __name__ == "__main__":
    main()
