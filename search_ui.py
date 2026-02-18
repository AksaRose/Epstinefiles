#!/usr/bin/env python3
"""Streamlit UI to search the Epstein image index (LanceDB)."""

from __future__ import annotations

import lancedb
import numpy as np
import streamlit as st

from config import load_settings
from pipeline.embed import embed_query


@st.cache_resource
def get_table():
    settings = load_settings()
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
                celebs = row.get("celebrities") or []
                if celebs:
                    st.caption(f"People: {', '.join(celebs)}")
                cap = row.get("caption") or ""
                if cap:
                    st.caption(cap[:200] + ("..." if len(cap) > 200 else ""))
                st.caption(f"Dataset {row.get('dataset_id', '')} · distance {row.get('_distance', ''):.3f}")


if __name__ == "__main__":
    main()
