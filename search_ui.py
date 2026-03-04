#!/usr/bin/env python3
"""Streamlit UI for the Epstein face-clustering image gallery: cluster circles, people search, dataset filter."""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import lancedb
import streamlit as st

from config import load_settings
from pipeline import face_store


def _thumbnail_path_for(original: Path) -> Path:
    """
    Given a full image path under epstein_pdfs, map it to the matching path under
    epstein_thumbs (same relative structure). If we can't recognize the pattern,
    fall back to the original path.
    """
    parts = list(original.parts)
    if "epstein_pdfs" in parts:
        idx = parts.index("epstein_pdfs")
        # /.../epstein_pdfs/... -> /.../epstein_thumbs/... and always use .jpg like generator
        base = Path(*parts[:idx], "epstein_thumbs", *parts[idx + 1 :])
        return base.with_suffix(".jpg")
    return original


def _image_bytes_from_path(image_path: str | None) -> bytes | None:
    """
    Load image bytes from disk.

    Prefer a pre-generated thumbnail in epstein_thumbs (if present), otherwise
    fall back to the original full-resolution image. Returns None if nothing
    is readable.
    """
    if not image_path:
        return None
    orig = Path(image_path)
    thumb = _thumbnail_path_for(orig)
    for candidate in (thumb, orig):
        if not candidate.exists() or not candidate.is_file():
            continue
        try:
            return candidate.read_bytes()
        except Exception:
            continue
    return None


def _original_image_bytes(image_path: str | None) -> bytes | None:
    """Load bytes from the original full-resolution image path."""
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

    # Task bar icon clicks: link adds ?taskbar=filters|people; we toggle panel and clear URL param.
    taskbar_val = st.query_params.get("taskbar")
    if taskbar_val:
        taskbar_val = str(taskbar_val).strip().lower()
        if taskbar_val == "filters":
            st.session_state["taskbar_show_filters"] = not st.session_state.get("taskbar_show_filters", False)
        elif taskbar_val == "people":
            st.session_state["taskbar_show_people"] = not st.session_state.get("taskbar_show_people", False)
        try:
            del st.query_params["taskbar"]
        except KeyError:
            pass
        st.rerun()

    # Clicking a person card (link ?cluster=id) selects that person and shows their images
    cluster_param = st.query_params.get("cluster")
    if cluster_param:
        try:
            cid = int(str(cluster_param).strip())
            st.session_state["selected_cluster_id"] = cid
            try:
                del st.query_params["cluster"]
            except KeyError:
                pass
            st.rerun()
        except ValueError:
            pass

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
    # Vertical task bar on the left (fixed); Lucide icons; sidebar toggle styled like task bar
    st.markdown(
        """
        <style>
            section.main .block-container { padding-left: 56px !important; }
            #epstein-taskbar {
                position: fixed;
                left: 0;
                top: 0;
                width: 56px;
                height: 100vh;
                background: #262730;
                z-index: 999;
                display: flex;
                flex-direction: column;
                align-items: center;
                padding-top: 1rem;
                gap: 0.5rem;
                border-right: 1px solid #3a3a46;
            }
            #epstein-taskbar button {
                width: 40px;
                height: 40px;
                border: none;
                border-radius: 8px;
                background: transparent;
                color: #fafafa;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            #epstein-taskbar button:hover { background: #3a3a46; }
            #epstein-taskbar button:active { background: #4a4a56; }
            #epstein-taskbar button svg { width: 22px; height: 22px; fill: #fafafa; color: #fafafa; }
            #epstein-taskbar button svg path { fill: #fafafa; }
            #epstein-taskbar a.taskbar-link {
                width: 40px; height: 40px; border-radius: 8px; display: flex; align-items: center; justify-content: center;
                background: transparent; color: #fafafa; text-decoration: none;
            }
            #epstein-taskbar a.taskbar-link:hover { background: #3a3a46; }
            #epstein-taskbar a.taskbar-link svg { width: 22px; height: 22px; }
            /* Streamlit sidebar toggle (>>) — force task bar color #262730 */
            [data-testid="collapsedControl"],
            [data-testid="collapsedControl"] button,
            [data-testid="stSidebarCollapsedControl"],
            [data-testid="stSidebarCollapsedControl"] button,
            section[data-testid="stSidebar"] button[kind="header"],
            section[data-testid="stSidebar"] [data-testid="collapsedControl"],
            div[data-testid="stSidebar"] > div:first-child > button,
            button[data-testid="baseButton-header"] {
                background: #262730 !important;
                background-color: #262730 !important;
                color: #fafafa !important;
                border-color: #3a3a46 !important;
                fill: #fafafa !important;
            }
            [data-testid="collapsedControl"] svg,
            [data-testid="collapsedControl"] path,
            section[data-testid="stSidebar"] button[kind="header"] svg,
            section[data-testid="stSidebar"] button[kind="header"] path {
                fill: #fafafa !important;
                color: #fafafa !important;
            }
            [data-testid="collapsedControl"]:hover,
            [data-testid="collapsedControl"]:hover button,
            [data-testid="stSidebarCollapsedControl"]:hover,
            section[data-testid="stSidebar"] button[kind="header"]:hover {
                background: #3a3a46 !important;
                background-color: #3a3a46 !important;
            }
        </style>
        <div id="epstein-taskbar">
            <button type="button" id="epstein-sidebar-toggle" title="Open sidebar">
                <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fafafa" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M9 3v18"/></svg>
            </button>
            <a href="?taskbar=filters" class="taskbar-link" title="Filters — Dataset & search" target="_self">
                <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fafafa" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/></svg>
            </a>
            <a href="?taskbar=people" class="taskbar-link" title="People — Choose a person" target="_self">
                <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fafafa" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
            </a>
        </div>
        <script>
            document.getElementById('epstein-sidebar-toggle').onclick = function() {
                var btn = document.querySelector('[data-testid="collapsedControl"]')
                    || document.querySelector('[data-testid="stSidebarCollapsedControl"]')
                    || document.querySelector('button[kind="header"]')
                    || document.querySelector('[aria-label*="sidebar" i]');
                if (btn) btn.click();
            };
            /* Force task bar color onto sidebar toggle (Streamlit may override CSS) */
            function styleSidebarToggle() {
                document.querySelectorAll('[data-testid="collapsedControl"], section[data-testid="stSidebar"] button[kind="header"]').forEach(function(el) {
                    el.style.setProperty('background', '#262730', 'important');
                    el.style.setProperty('background-color', '#262730', 'important');
                    el.style.setProperty('color', '#fafafa', 'important');
                    el.querySelectorAll('svg, path').forEach(function(s) { s.style.setProperty('fill', '#fafafa', 'important'); });
                });
            }
            [100, 400, 1000].forEach(function(ms) { setTimeout(styleSidebarToggle, ms); });
        </script>
        """,
        unsafe_allow_html=True,
    )

    st.title("Epstein Case Image Gallery")

    st.markdown(
        "Explore DOJ-released images related to the Epstein case. "
        "**Each circle is one person** – click to see all images containing that face. "
        "Use the search box to find people by the names you give them."
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
    taskbar_show_filters = st.session_state.get("taskbar_show_filters", False)
    taskbar_show_people = st.session_state.get("taskbar_show_people", False)

    # Main-area filters when Filter icon was clicked
    if taskbar_show_filters:
        st.selectbox("Dataset", dataset_options, key="filter_dataset", help="Limit results to one disclosure batch")
        st.text_input("Search people", placeholder="Type a name you gave a person", key="filter_people_search")
        st.markdown("---")

    # Sidebar: filters (if not in main) + people list
    with st.sidebar:
        if not taskbar_show_filters:
            st.selectbox("Dataset", dataset_options, key="filter_dataset", help="Limit results to one disclosure batch (HS = House oversight)")
            st.text_input("Search people", placeholder="Type a name you gave a person", key="filter_people_search")

        # Shared filter state from session
        filter_label = st.session_state.get("filter_dataset", "All")
        selected_dataset_id = None if filter_label == "All" else (HS_DATASET_ID if filter_label == "HS" else int(filter_label))
        people_query = (st.session_state.get("filter_people_search") or "").strip().lower()

        cluster_ids_in_dataset = None
        if selected_dataset_id is not None and not faces_df.empty and not img_df.empty:
            image_ids_in_dataset = set(img_df[img_df["dataset_id"] == selected_dataset_id]["id"].astype(str))
            cluster_ids_in_dataset = set(
                faces_df[faces_df["image_id"].astype(str).isin(image_ids_in_dataset)]["cluster_id"].dropna().unique().tolist()
            )

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

        # Named clusters first, then unnamed clusters ordered by highest image count
        named_reps = [r for r in reps_filtered if str(r.get("cluster_id")) in cluster_names]
        unnamed_reps = [r for r in reps_filtered if str(r.get("cluster_id")) not in cluster_names]

        # Compute image counts per cluster (respecting dataset filter if present)
        if not faces_df.empty:
            if selected_dataset_id is not None and not img_df.empty:
                img_ids_ds = set(
                    img_df[img_df["dataset_id"] == selected_dataset_id]["id"].astype(str)
                )
                faces_for_counts = faces_df[
                    faces_df["image_id"].astype(str).isin(img_ids_ds)
                ]
            else:
                faces_for_counts = faces_df
            cluster_image_count_sidebar = (
                faces_for_counts.groupby("cluster_id")["image_id"].nunique().to_dict()
                if not faces_for_counts.empty
                else {}
            )
        else:
            cluster_image_count_sidebar = {}

        def _count_for_rep(rep: dict) -> int:
            cid = rep.get("cluster_id")
            try:
                return int(cluster_image_count_sidebar.get(int(cid), 0))
            except Exception:
                return 0

        # Sort both named and unnamed by image count (descending), but keep named first
        named_reps = sorted(named_reps, key=_count_for_rep, reverse=True)
        unnamed_reps = sorted(unnamed_reps, key=_count_for_rep, reverse=True)
        reps_ordered = named_reps + unnamed_reps

        st.markdown("---")
        st.markdown("**People**")
        st.caption("Click a person to see their images.")
        if not reps_filtered:
            if people_query:
                st.info("No match.")
            else:
                st.info("No face clusters.")
        else:
            # Show first N, then "Show more"
            sidebar_show_limit = 10
            has_more = len(reps_ordered) > sidebar_show_limit
            show_all_key = "show_all_people"
            show_all = st.session_state.get(show_all_key, False)
            if has_more:
                if st.button("Show more people" if not show_all else "Show fewer", key="toggle_people"):
                    st.session_state[show_all_key] = not show_all
                    st.rerun()
            reps_to_show = reps_ordered if (not has_more or show_all) else reps_ordered[:sidebar_show_limit]

            circle_px = 44
            for rep in reps_to_show:
                cid = rep.get("cluster_id")
                image_id = rep.get("image_id")
                bbox = rep.get("bbox") or [0, 0, 0, 0]
                if selected_dataset_id is not None and not faces_df.empty and not img_df.empty:
                    try:
                        fsub = faces_df[faces_df["cluster_id"] == int(cid)]
                        if not fsub.empty:
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
                        pass
                name = cluster_names.get(str(cid), f"Person {cid}")
                rows = img_df[img_df["id"].astype(str) == str(image_id)]
                img_html = ""
                if not rows.empty:
                    path = rows.iloc[0].get("image_path")
                    # Use the original full-resolution page for face crops so bbox coordinates line up
                    blob = _original_image_bytes(path)
                    crop_bytes = _crop_image_to_bbox(blob, bbox) if blob else None
                    if crop_bytes:
                        try:
                            b64 = base64.b64encode(crop_bytes).decode("ascii")
                            img_html = (
                                f'<img src="data:image/png;base64,{b64}" '
                                f'style="border-radius:50%; width:{circle_px}px; height:{circle_px}px; '
                                f'object-fit:cover; vertical-align:middle;" />'
                            )
                        except Exception:
                            pass
                # Row: circle + name button (Google Photos style)
                col_thumb, col_btn = st.columns([1, 3])
                with col_thumb:
                    if img_html:
                        st.markdown(f"<div>{img_html}</div>", unsafe_allow_html=True)
                    else:
                        st.caption("")
                with col_btn:
                    is_selected = st.session_state.get("selected_cluster_id") == int(cid)
                    label = f"✓ {name}" if is_selected else name
                    if st.button(label, key=f"cluster_btn_{cid}"):
                        st.session_state["selected_cluster_id"] = int(cid)
                        st.rerun()

    # Main-area people grid: shown when People icon is active OR when there's a search query
    show_people_from_search = bool((st.session_state.get("filter_people_search") or "").strip())
    show_people_grid = taskbar_show_people or show_people_from_search

    if show_people_grid and reps_ordered:
        st.caption("People that match your current filters. Click one to open their images.")
        # Precompute image count per cluster (distinct images) once for performance,
        # respecting the current dataset filter if one is selected.
        if not faces_df.empty:
            if selected_dataset_id is not None and not img_df.empty:
                img_ids_ds_main = set(
                    img_df[img_df["dataset_id"] == selected_dataset_id]["id"].astype(str)
                )
                faces_for_counts_main = faces_df[
                    faces_df["image_id"].astype(str).isin(img_ids_ds_main)
                ]
            else:
                faces_for_counts_main = faces_df
            cluster_image_count = (
                faces_for_counts_main.groupby("cluster_id")["image_id"].nunique().to_dict()
                if not faces_for_counts_main.empty
                else {}
            )
        else:
            cluster_image_count = {}

        card_size = 100
        n_cols = 6

        # Show a limited number by default for faster initial load; allow user to expand
        main_limit = 30
        total_people = len(reps_ordered)
        show_all_key = "show_all_main_people"
        show_all = st.session_state.get(show_all_key, False)
        if total_people > main_limit:
            toggle_label = "Show all people" if not show_all else "Show fewer people"
            if st.button(toggle_label, key="toggle_main_people"):
                st.session_state[show_all_key] = not show_all
                st.rerun()
        reps_for_grid = (
            reps_ordered
            if (total_people <= main_limit or show_all)
            else reps_ordered[:main_limit]
        )
        st.caption(f"Showing {len(reps_for_grid)} of {total_people} people")

        cards_html = []
        for rep in reps_for_grid:
            cid = rep.get("cluster_id")
            image_id = rep.get("image_id")
            bbox = rep.get("bbox") or [0, 0, 0, 0]
            if selected_dataset_id is not None and not faces_df.empty and not img_df.empty:
                try:
                    fsub = faces_df[faces_df["cluster_id"] == int(cid)]
                    if not fsub.empty:
                        merged = fsub.merge(
                            img_df[["id", "dataset_id"]].rename(columns={"id": "img_id"}),
                            left_on="image_id", right_on="img_id", how="left",
                        )
                        cand = merged[merged["dataset_id"] == selected_dataset_id]
                        if not cand.empty:
                            row0 = cand.iloc[0]
                            image_id, bbox = row0.get("image_id", image_id), row0.get("bbox", bbox) or bbox
                except Exception:
                    pass
            name = cluster_names.get(str(cid), f"Person {cid}")
            n_imgs = cluster_image_count.get(int(cid), 0)
            img_src = ""
            rows = img_df[img_df["id"].astype(str) == str(image_id)]
            if not rows.empty:
                path = rows.iloc[0].get("image_path")
                # Use original image for representative face crops; thumbnails would
                # misalign bbox coordinates and produce random-looking patches.
                blob = _original_image_bytes(path)
                crop_bytes = _crop_image_to_bbox(blob, bbox) if blob else None
                if crop_bytes:
                    try:
                        b64 = base64.b64encode(crop_bytes).decode("utf-8")
                        img_src = f'data:image/png;base64,{b64}'
                    except Exception:
                        pass
            if not img_src:
                _svg = '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100"><rect fill="#333" width="100" height="100"/><text x="50" y="55" fill="#999" text-anchor="middle" font-size="12">?</text></svg>'
                img_src = "data:image/svg+xml," + quote(_svg)
            name_esc = name.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
            cards_html.append(
                f'<a href="?cluster={cid}" class="epstein-people-card" target="_self" title="{name_esc}">'
                f'<img src="{img_src}" width="{card_size}" height="{card_size}" alt=""/>'
                f'<span class="epstein-people-name">{name_esc}</span>'
                f'<span class="epstein-people-count">{n_imgs} image{"s" if n_imgs != 1 else ""}</span>'
                f"</a>"
            )
        grid_html = (
            "<style>"
            ".epstein-people-grid{display:grid;grid-template-columns:repeat(" + str(n_cols) + ",1fr);gap:1rem;margin:1rem 0;}"
            ".epstein-people-card, .epstein-people-card:link, .epstein-people-card:visited, .epstein-people-card:hover, .epstein-people-card:active{ text-decoration:none; }"
            ".epstein-people-card{display:flex;flex-direction:column;align-items:center;justify-content:flex-start;color:inherit;border-radius:10px;padding:10px;background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);transition:background 0.15s ease,border-color 0.15s ease,transform 0.1s ease;}"
            ".epstein-people-card:hover{background:#3a3a46;border-color:#565676;transform:translateY(-1px);}"
            ".epstein-people-card img{width:100px;height:100px;object-fit:cover;border-radius:10px;box-shadow:0 2px 6px rgba(0,0,0,0.4);}"
            ".epstein-people-name{font-weight:600;margin-top:6px;text-align:center;font-size:0.9rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:100%;color:#f5f5f5;}"
            ".epstein-people-count{font-size:0.8rem;color:#b3b3b3;margin-top:2px;}"
            "</style>"
            '<div class="epstein-people-grid">' + "".join(cards_html) + "</div>"
        )
        st.markdown(grid_html, unsafe_allow_html=True)
        st.markdown("---")

    # When searching by name but no people match, show a friendly message
    if show_people_grid and not reps_ordered and show_people_from_search:
        st.info("No people matched this search. Try a different name.")

    selected_cluster_id = st.session_state.get("selected_cluster_id")

    if selected_cluster_id is not None:
        st.divider()
        st.markdown(f"**Images for: {cluster_names.get(str(selected_cluster_id), f'Person {selected_cluster_id}')}**")

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
                # Light "lazy loading": show a limited number by default, allow user to expand
                images_limit = 30
                total_imgs = len(img_subset)
                show_all_key = f"show_all_images_cluster_{selected_cluster_id}"
                show_all = st.session_state.get(show_all_key, False)
                if total_imgs > images_limit:
                    label = "Show all images" if not show_all else "Show fewer images"
                    if st.button(label, key=f"toggle_images_{selected_cluster_id}"):
                        st.session_state[show_all_key] = not show_all
                        st.rerun()
                display_subset = (
                    img_subset
                    if (total_imgs <= images_limit or show_all)
                    else img_subset.iloc[:images_limit]
                )
                st.caption(f"Showing {len(display_subset)} of {total_imgs} images")

                ncols = 3
                for start in range(0, len(display_subset), ncols):
                    row_imgs = display_subset.iloc[start : start + ncols]
                    cols = st.columns(ncols)
                    for i, (_, row) in enumerate(row_imgs.iterrows()):
                        with cols[i]:
                            blob = _image_bytes_from_path(row.get("image_path"))
                            if blob is not None:
                                try:
                                    st.image(blob, use_container_width=True)
                                except Exception:
                                    st.caption("(image failed to render)")
                            else:
                                st.caption("(image file not found)")
                            st.caption(f"{row.get('source_file', '')} p.{row.get('page_no', '')}")

    if selected_cluster_id is None and not reps_filtered:
        st.caption("Click a person above to view their images.")


if __name__ == "__main__":
    main()
