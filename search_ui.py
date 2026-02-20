#!/usr/bin/env python3
"""Streamlit UI to search the Epstein image index (LanceDB)."""

from __future__ import annotations

import os
import re
import tempfile
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import lancedb
import numpy as np
import streamlit as st
from groq import Groq

from config import load_settings
from pipeline.embed import embed_query


@st.cache_resource
def _get_known_people():
    """Known person names (my_db folder names) used for face recognition at ingestion."""
    settings = load_settings()
    if not settings.faces_db_dir.is_dir():
        return []
    return [p.name for p in sorted(settings.faces_db_dir.iterdir()) if p.is_dir()]


def _detect_mentioned_people(query: str, known_people: list[str]) -> list[str]:
    """Return known person names that appear in the query or whose name contains the query (case-insensitive).
    E.g. 'trump' matches 'Donald Trump', and 'Donald Trump' matches 'photos of Donald Trump'.
    """
    if not query or not known_people:
        return []
    q = query.strip().lower()
    if not q:
        return []
    return [
        name
        for name in known_people
        if name.lower() in q or q in name.lower()
    ]


def _person_filter_where(person_names: list[str]) -> str:
    """Build LanceDB .where() predicate: celebrities list contains any of these names."""
    if not person_names:
        return ""
    # Escape single quotes in names for SQL literal
    escaped = [n.replace("'", "''") for n in person_names]
    literals = ", ".join(f"'{e}'" for e in escaped)
    return f"array_has_any(celebrities, [{literals}])"


def _get_download_url() -> str | None:
    """LANCEDB_DOWNLOAD_URL from env or Streamlit secrets (for cloud deploy)."""
    url = os.environ.get("LANCEDB_DOWNLOAD_URL")
    if url:
        return url.strip() or None
    try:
        return st.secrets.get("LANCEDB_DOWNLOAD_URL") or None
    except Exception:
        return None


# Browser User-Agent so Google Drive doesn't block the request
_UA = "Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/91.0"

# Simple persistent visitor counter
_VISITOR_FILE = Path("visitor_count.txt")


def _load_visitor_count() -> int:
    try:
        return int(_VISITOR_FILE.read_text().strip() or "0")
    except FileNotFoundError:
        return 0
    except ValueError:
        return 0


def _increment_visitor_count() -> int:
    """
    Increment the global visitor count once per browser session.
    """
    count = _load_visitor_count()
    if not st.session_state.get("visitor_counted"):
        count += 1
        try:
            _VISITOR_FILE.write_text(str(count))
        except Exception:
            pass
        st.session_state["visitor_counted"] = True
    return count


def _resolve_google_drive_url(url: str) -> str:
    """
    If Google Drive returns HTML (virus scan / confirm page), parse it and return
    the URL with the confirm token so the next request gets the actual file.
    """
    if "drive.google.com" not in url:
        return url
    mid = re.search(r"id=([0-9A-Za-z_.-]+)", url)
    if not mid:
        return url
    file_id = mid.group(1)

    req = Request(url, headers={"User-Agent": _UA})
    with urlopen(req, timeout=30) as resp:
        head = resp.read(64 * 1024)  # first 64 KB
    # Zip files start with PK
    if head.startswith(b"PK"):
        return url
    try:
        body = head.decode("utf-8", errors="ignore")
    except Exception:
        return _usercontent_drive_url(file_id)
    # Look for confirm token in Google's "virus scan" page
    m = re.search(r"confirm=([^\"'\s&]+)", body)
    if m:
        token = m.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}&confirm={token}"
    m = re.search(r"/uc\?export=download[^\"']*confirm=([^\"'\s&]+)", body)
    if m:
        return f"https://drive.google.com/uc?export=download&id={file_id}&confirm={m.group(1)}"
    # Fallback: usercontent endpoint with confirm=t (often works for public files)
    return _usercontent_drive_url(file_id)


def _usercontent_drive_url(file_id: str) -> str:
    """Alternative Drive download URL that sometimes bypasses virus scan."""
    return f"https://drive.usercontent.google.com/download?export=download&confirm=t&id={file_id}"


def _download_to_file(url: str, tmp_path: str, progress_callback=None) -> None:
    """Download url to tmp_path, handling Google Drive confirm. progress_callback(total_mb) optional."""
    url = _resolve_google_drive_url(url)
    req = Request(url, headers={"User-Agent": _UA})
    chunk_size = 1 << 20  # 1 MB
    with urlopen(req, timeout=60) as resp:
        with open(tmp_path, "wb") as out:
            total = 0
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                total += len(chunk)
                if progress_callback:
                    progress_callback(total // (1 << 20))


def _ensure_lancedb(settings) -> bool:
    """
    If LanceDB dir is missing and LANCEDB_DOWNLOAD_URL is set, download zip and unzip.
    Zip must contain the table at top level (e.g. epstein_images.lance/).
    Supports Google Drive (handles virus-scan confirm page).
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
                def on_progress(mb: int):
                    st.progress(min(1.0, mb / 200.0), text=f"Downloaded {mb} MB")

                _download_to_file(url, tmp, progress_callback=on_progress)
                st.progress(1.0, text="Extracting...")
                with zipfile.ZipFile(tmp, "r") as zf:
                    zf.extractall(settings.lancedb_dir)
            finally:
                try:
                    Path(tmp).unlink(missing_ok=True)
                except Exception:
                    pass
            return True
        except zipfile.BadZipFile as e:
            st.error(f"Downloaded file is not a valid zip (often means Google Drive confirm failed): {e}")
            return False
        except Exception as e:
            st.error(f"Failed to download database: {e}")
            return False


def _analyze_query(user_query: str, settings) -> tuple[str, str]:
    """
    Use Groq to classify intent (question vs keyword search) and produce a search-optimized query
    for better retrieval. Returns (intent, search_query).
    """
    groq_key = getattr(settings, "groq_api_key", None) or os.environ.get("GROQ_API_KEY")
    if not groq_key or not user_query.strip():
        return ("keyword_search", user_query.strip())

    model = getattr(settings, "groq_summary_model", None) or os.environ.get("GROQ_SUMMARY_MODEL", "llama-3.3-70b-versatile")
    prompt = (
        "You are analyzing a user input for an image search gallery about DOJ-released materials related to the Epstein case.\n\n"
        f"User input: \"{user_query.strip()}\"\n\n"
        "Reply with exactly two lines:\n"
        "Line 1: Either QUESTION or KEYWORD (whether the user asked a question or is searching by topic/keywords).\n"
        "Line 2: A short search query (keywords or key phrases) that would best find relevant images. For questions, turn the question into search keywords (e.g. \"Who was at the party?\" -> \"party, people, guests, gathering\"). For keyword search, use or lightly expand the user's words. Do not add specific names or places the user did not mention. Keep line 2 under 15 words."
    )
    try:
        client = Groq(api_key=groq_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0,
        )
        text = (resp.choices[0].message.content or "").strip()
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if len(lines) >= 2:
            intent = "question" if "QUESTION" in lines[0].upper() else "keyword_search"
            search_query = lines[1].strip()
            if search_query:
                return (intent, search_query)
    except Exception:
        pass
    return ("keyword_search", user_query.strip())


def _summarize_results(query: str, captions: list[str], settings) -> str | None:
    """
    Use Groq (Llama) to generate a brief, answer-like summary grounded in the top results.
    """
    if not captions:
        return None
    groq_key = getattr(settings, "groq_api_key", None) or os.environ.get("GROQ_API_KEY")
    if not groq_key:
        return None

    model = getattr(settings, "groq_summary_model", None) or os.environ.get("GROQ_SUMMARY_MODEL", "llama-3.3-70b-versatile")
    client = Groq(api_key=groq_key)
    top_caps = captions[: min(20, len(captions))]
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(top_caps))
    prompt = (
        "You are answering a user's query using only what appears in a set of image captions from DOJ-released materials "
        "related to the Epstein case.\n\n"
        f"User query:\n{query}\n\n"
        "Captions of the most relevant images:\n"
        f"{numbered}\n\n"
        "Based only on these captions, write a short, neutral answer (2–4 sentences) that speaks directly to the user's query. "
        "If the captions do not provide enough information to fully answer the query, say that clearly and instead describe what "
        "the images do show that is relevant. Do not speculate, infer, or introduce people, places, or events that are not "
        "explicitly mentioned in the captions. Do not hallucinate; when uncertain, say so."
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=220,
            temperature=0,
        )
    except Exception:
        return None

    choice = resp.choices[0]
    text = choice.message.content
    cleaned = (text or "").strip()
    return cleaned or None


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


def search(
    query: str,
    k: int = 12,
    dataset_id: int | None = None,
    person_filter: list[str] | None = None,
):
    settings = load_settings()
    table, _ = get_table()
    vec = embed_query(
        query,
        api_key=settings.together_api_key,
        model=settings.embed_model,
    )
    vec_arr = np.array(vec, dtype="float32")
    try:
        q = (
            table.search(query_type="hybrid", vector_column_name="vector")
            .vector(vec_arr)
            .text(query)
        )
    except Exception:
        q = table.search(vec_arr)
    if dataset_id is not None:
        q = q.where(f"dataset_id = {int(dataset_id)}")
    if person_filter:
        where_expr = _person_filter_where(person_filter)
        if where_expr:
            q = q.where(where_expr)
    results = q.limit(k).to_pandas()
    return results


def main():
    # Transparent 1x1 PNG so the tab doesn't show the Streamlit favicon
    _BLANK_FAVICON = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    st.set_page_config(page_title="Epstein Image Search", layout="wide", page_icon=_BLANK_FAVICON)
    # Microsoft Clarity analytics
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
        "This is an image gallery of materials related to the Epstein case. You can search across all images by keyword or topic. "
        "**All images displayed here are from documents released by the U.S. Department of Justice (DOJ)** as part of its disclosure. "
        "Use the search box below to find images (e.g., by place, person, or subject)."
    )

    with st.expander("About this gallery & disclaimer", expanded=False):
        st.markdown(
            "**Current scope:** This gallery currently includes approximately **250 documents** across DOJ disclosure datasets 1–5. "
            "That is roughly one fifth of the material released so far. The focus here is on **photographs** rather than emails or text-heavy images; "
            "the indexed set contains **2,000+ images**."
        )
        st.markdown(
            "**Disclaimer:** Captions and metadata are generated with AI (LLM) and **may contain errors**. "
            "This material is sensitive. Do not rely on it for legal or factual conclusions; refer to official DOJ sources when accuracy matters."
        )

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

    groq_configured = bool(getattr(settings, "groq_api_key", None) or os.environ.get("GROQ_API_KEY"))
    with st.spinner("Analyzing query..." if groq_configured else "Preparing search..."):
        intent, search_query = _analyze_query(query, settings)
    if not groq_configured:
        st.caption("Query analysis unavailable (set GROQ_API_KEY in .env for intent detection and summaries).")
    # If the user query mentions a known person (from my_db / face recognition), filter to those rows
    known_people = _get_known_people()
    mentioned_people = _detect_mentioned_people(query.strip(), known_people)
    if mentioned_people:
        display_names = [n.replace("_", " ") for n in mentioned_people]
        st.info(f"Filtering to images where **{', '.join(display_names)}** was identified (face recognition).")
    with st.spinner("Searching..."):
        try:
            results = search(
                search_query,
                k=k,
                dataset_id=dataset_id,
                person_filter=mentioned_people if mentioned_people else None,
            )
        except Exception as e:
            st.exception(e)
            return

    if results is None or results.empty:
        st.warning("No results.")
        return

    # High-level Groq summary: use original query so the answer matches what the user asked
    captions_for_summary: list[str] = []
    if "caption" in results.columns:
        for c in results["caption"].tolist():
            if c is None:
                continue
            s = str(c).strip()
            if s:
                captions_for_summary.append(s)
    summary = _summarize_results(query, captions_for_summary, settings)
    if summary:
        st.subheader("Summary of results")
        st.markdown(summary)

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
                # Caption (no heading)
                cap = row.get("caption")
                if cap is not None and str(cap).strip():
                    st.caption(str(cap).strip())
                # People present (from face recognition); may be list or numpy array from LanceDB
                celebs = row.get("celebrities")
                if celebs is not None:
                    try:
                        names = [str(x).strip() for x in celebs if x is not None and str(x).strip()]
                        if names:
                            st.caption(", ".join(n.replace("_", " ") for n in names))
                    except (TypeError, ValueError):
                        st.caption(str(celebs).replace("_", " "))


if __name__ == "__main__":
    main()
