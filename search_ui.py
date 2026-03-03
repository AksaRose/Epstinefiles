#!/usr/bin/env python3
"""Streamlit UI for the Epstein face-clustering image gallery: cluster circles, people search, dataset filter."""

from __future__ import annotations

import io
import os
import base64
from pathlib import Path

import pandas as pd
import lancedb
import streamlit as st

from config import load_settings
from pipeline import face_store


def _image_bytes_from_path(image_path: str | None) -> bytes | None:
    """Load image bytes from disk. Returns None if path missing or unreadable."""
    if not image_path:
        return None
    p = Path(image_path)
    if not p.exists() or not p.is_file():
        return None
    try:
        return p.read_bytes()
    except Exception:
        return None


def _crop_image_to_bbox(blob: bytes, bbox: list[int]) -> bytes | None:
    """Crop image to bbox [x1,y1,x2,y2]; clamp to image size. Returns PNG bytes or None."""
    if not blob or len(bbox) < 4:
        return None
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(blob)).convert("RGB")
        w, h = img.size
        x1 = max(0, min(int(bbox[0]), w - 1))
        y1 = max(0, min(int(bbox[1]), h - 1))
        x2 = max(x1 + 1, min(int(bbox[2]), w))
        y2 = max(y1 + 1, min(int(bbox[3]), h))
        cropped = img.crop((x1, y1, x2, y2))
        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


@st.cache_resource
def _get_face_tables():
    """Open LanceDB images and faces tables. Raises if not found."""
    settings = load_settings()
    lancedb_dir = Path(settings.lancedb_dir)
    if not lancedb_dir.exists():
        raise FileNotFoundError(f"LanceDB dir not found: {lancedb_dir}. Run: python run_pipeline.py --face-clustering")
    db = lancedb.connect(str(lancedb_dir))
    try:
        images_t = db.open_table(face_store.IMAGES_TABLE)
        faces_t = db.open_table(face_store.FACES_TABLE)
    except Exception as e:
        raise FileNotFoundError(
            f"Face-clustering tables not found. Run: python run_pipeline.py --face-clustering. ({e})"
        ) from e
    return images_t, faces_t, settings


def main():
    _BLANK_FAVICON = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    st.set_page_config(page_title="Epstein Image Gallery", layout="wide", page_icon=_BLANK_FAVICON)
    st.components.v1.html(
        """
        <script type="text/javascript">
            (function(c,l,a,r,i,t,y){
                c[a]=c[a]||function(){(c[a].q=c[a].q||[]).push(arguments)};
                t=l.createElement(r);t.async=1;t.src="https://www.clarity.ms/tag/"+i;
                y=l.getElementsByTagName(r)[0];y.parentNode.insertBefore(t,y);
            })(window, document, "clarity", "script", "vjtwq5oy8x");
        </script>
        """,
        height=0,
    )
    st.title("Epstein Case Image Gallery")

    with st.sidebar:
        st.markdown("### Support this project")
        st.markdown(
            "Any support from this project helps **keep this app online** and "
            "**add more documents, images, and search features** over time."
        )
        st.components.v1.html(
            """
            <script
                data-name="BMC-Widget"
                data-cfasync="false"
                src="https://cdnjs.buymeacoffee.com/1.0.0/widget.prod.min.js"
                data-id="aksarose"
                data-description="Support me on Buy me a coffee!"
                data-message=""
                data-color="#5F7FFF"
                data-position="Left"
                data-x_margin="18"
                data-y_margin="18">
            </script>
            """,
            height=80,
        )

    st.markdown(
        "Image gallery of DOJ-released materials related to the Epstein case. "
        "**Faces are clustered** — click a person below to see all images containing that face. "
        "Use the search box to filter clusters by name; you can rename any cluster."
    )
    with st.expander("About & disclaimer", expanded=False):
        st.markdown(
            "**Scope:** This gallery uses face detection and clustering (no caption/LLM). "
            "All images are from DOJ disclosure. Do not rely on this for legal or factual conclusions."
        )

    try:
        with st.spinner("Opening database..."):
            images_table, faces_table, settings = _get_face_tables()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    lancedb_dir = Path(settings.lancedb_dir)
    cluster_names = face_store.load_cluster_names(lancedb_dir)
    representatives = face_store.load_cluster_representatives(lancedb_dir)

    # Load images metadata only (exclude image_blob to avoid Lance offset-overflow decode bugs)
    _IMG_META_COLS = ["id", "dataset_id", "source_file", "page_no", "image_path"]
    with st.spinner("Loading images table..."):
        try:
            img_df = images_table.search().select(_IMG_META_COLS).limit(1_000_000).to_pandas()
        except Exception as e:
            st.error(
                "Could not load images table. If you see an 'offset overflow' error, the table may be corrupted; "
                "try re-running: python run_pipeline.py --face-clustering"
            )
            st.exception(e)
            return
    if img_df is None or img_df.empty:
        st.warning("No images in the database.")
        return
    with st.spinner("Loading faces table..."):
        try:
            faces_df = faces_table.to_pandas()
        except Exception:
            try:
                faces_df = faces_table.search().limit(2**31 - 1).to_pandas()
            except Exception:
                faces_df = faces_table.query().to_pandas() if hasattr(faces_table, "query") else faces_table.to_pandas()
    if faces_df is None:
        faces_df = pd.DataFrame()
    # Dataset 6 = HS (House oversight / high school) folder
    HS_DATASET_ID = 6
    distinct_datasets = sorted(img_df["dataset_id"].dropna().unique().tolist())
    def _dataset_label(did):
        return "HS" if int(did) == HS_DATASET_ID else str(int(did))
    dataset_options = ["All"] + [_dataset_label(d) for d in distinct_datasets]
    dataset_filter = st.selectbox("Dataset", dataset_options, help="Filter by DOJ disclosure dataset (HS = House oversight)")
    if dataset_filter == "All":
        selected_dataset_id = None
    elif dataset_filter == "HS":
        selected_dataset_id = HS_DATASET_ID
    else:
        selected_dataset_id = int(dataset_filter)

    people_search = st.text_input("Search people", placeholder="Filter clusters by name (e.g. Bill)")
    people_query = (people_search or "").strip().lower()

    # When a dataset is selected, only show clusters that appear in that dataset
    cluster_ids_in_dataset = None
    if selected_dataset_id is not None and not faces_df.empty and not img_df.empty:
        image_ids_in_dataset = set(img_df[img_df["dataset_id"] == selected_dataset_id]["id"].astype(str))
        cluster_ids_in_dataset = set(
            faces_df[faces_df["image_id"].astype(str).isin(image_ids_in_dataset)]["cluster_id"].dropna().unique().tolist()
        )

    # Filter representatives: exclude noise (-1); filter by name if search non-empty; by dataset if selected
    reps_filtered = []
    for r in representatives:
        cid = r.get("cluster_id")
        if cid is None or int(cid) < 0:
            continue
        if cluster_ids_in_dataset is not None and int(cid) not in cluster_ids_in_dataset:
            continue
        name = cluster_names.get(str(cid), f"Person {cid}")
        if people_query and people_query not in name.lower():
            continue
        reps_filtered.append(r)

    # Face circles: show representative crop per cluster; click to select
    st.subheader("People (click to see images)")
    if not reps_filtered:
        if people_query:
            st.info("No clusters match that name. Try a different search or rename a cluster.")
        else:
            st.info("No face clusters yet, or all are noise. Run the pipeline with more images.")
    else:
        # Small circular avatars in a horizontal grid
        n_cols = min(12, max(1, len(reps_filtered)))
        cols = st.columns(n_cols)
        circle_px = 48
        for idx, rep in enumerate(reps_filtered):
            cid = rep.get("cluster_id")
            # Start from global representative for this cluster
            image_id = rep.get("image_id")
            bbox = rep.get("bbox") or [0, 0, 0, 0]
            # If a dataset is selected, try to use a representative from that dataset
            if selected_dataset_id is not None and not faces_df.empty and not img_df.empty:
                try:
                    # Faces in this cluster
                    fsub = faces_df[faces_df["cluster_id"] == int(cid)]
                    if not fsub.empty:
                        # Join to images to get dataset_id per face
                        merged = fsub.merge(
                            img_df[["id", "dataset_id"]].rename(columns={"id": "img_id"}),
                            left_on="image_id",
                            right_on="img_id",
                            how="left",
                        )
                        cand = merged[merged["dataset_id"] == selected_dataset_id]
                        if not cand.empty:
                            row0 = cand.iloc[0]
                            image_id = row0.get("image_id", image_id)
                            bbox = row0.get("bbox", bbox) or bbox
                except Exception:
                    # Best-effort; fall back to original representative
                    pass

            name = cluster_names.get(str(cid), f"Person {cid}")
            col = cols[idx % n_cols]
            with col:
                rows = img_df[img_df["id"].astype(str) == str(image_id)]
                img_html = ""
                if not rows.empty:
                    path = rows.iloc[0].get("image_path")
                    blob = _image_bytes_from_path(path)
                    crop_bytes = _crop_image_to_bbox(blob, bbox) if blob else None
                    if crop_bytes:
                        try:
                            b64 = base64.b64encode(crop_bytes).decode("ascii")
                            img_html = (
                                f'<img src="data:image/png;base64,{b64}" '
                                f'style="border-radius:50%; width:{circle_px}px; height:{circle_px}px; '
                                f'object-fit:cover; display:block; margin:0 auto;" />'
                            )
                        except Exception:
                            img_html = ""

                # Render circle (if any) and name, centered
                if img_html:
                    st.markdown(f"<div style='text-align:center'>{img_html}</div>", unsafe_allow_html=True)
                button_label = name
                if st.button(button_label, key=f"cluster_btn_{cid}"):
                    st.session_state["selected_cluster_id"] = int(cid)
                if st.session_state.get("selected_cluster_id") == int(cid):
                    st.caption("✓ Selected")

    selected_cluster_id = st.session_state.get("selected_cluster_id")

    if selected_cluster_id is not None:
        st.divider()
        st.subheader(f"Images for cluster: {cluster_names.get(str(selected_cluster_id), f'Person {selected_cluster_id}')}")

        # Rename cluster
        rename_key = f"rename_{selected_cluster_id}"
        current_name = cluster_names.get(str(selected_cluster_id), f"Person {selected_cluster_id}")
        new_name = st.text_input("Rename this cluster", value=current_name, key=rename_key)
        if st.button("Save name"):
            if new_name and new_name.strip():
                cluster_names[str(selected_cluster_id)] = new_name.strip()
                face_store.save_cluster_names(lancedb_dir, cluster_names)
                st.success("Name saved.")
                st.rerun()

        # Image grid: faces where cluster_id = selected_cluster_id -> image_ids -> filter by dataset -> show images
        face_df = faces_df[faces_df["cluster_id"] == int(selected_cluster_id)] if not faces_df.empty else faces_df
        if face_df.empty:
            st.caption("No images for this cluster.")
        else:
            image_ids = face_df["image_id"].unique().tolist()
            # Subset images we already have by image_id and optional dataset filter
            id_set = set(str(i) for i in image_ids)
            img_subset = img_df[img_df["id"].astype(str).isin(id_set)].copy()
            if selected_dataset_id is not None:
                img_subset = img_subset[img_subset["dataset_id"] == selected_dataset_id]
            if img_subset.empty:
                st.caption("No images in the selected dataset for this cluster.")
            else:
                ncols = 3
                for start in range(0, len(img_subset), ncols):
                    row_imgs = img_subset.iloc[start : start + ncols]
                    cols = st.columns(ncols)
                    for i, (_, row) in enumerate(row_imgs.iterrows()):
                        with cols[i]:
                            blob = _image_bytes_from_path(row.get("image_path"))
                            if blob is not None:
                                st.image(blob, use_container_width=True)
                            else:
                                st.caption("(image file not found)")
                            st.caption(f"{row.get('source_file', '')} p.{row.get('page_no', '')}")

    if selected_cluster_id is None and not reps_filtered:
        st.caption("Select a person above to view their images.")


if __name__ == "__main__":
    main()
