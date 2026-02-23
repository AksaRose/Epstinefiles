"""
Chroma text RAG for Epstein Files 20K document chunks.
Used to generate a short 1–2 line description from the document corpus (instead of image captions).
Set CHROMA_DIR to a local chroma_db path, or CHROMA_HF_DATASET to auto-download from Hugging Face.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List

from config import load_settings
from groq import Groq

LOG = logging.getLogger(__name__)

CHROMA_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_QUERY_K = 8
CHROMA_FETCH_K = 20

# Cache Chroma collection by base path so we don't re-open DB or re-load the embedding model every query.
_chroma_collection_cache: dict = {}


def _chroma_base_dir(settings) -> Path | None:
    """Resolve Chroma DB directory: CHROMA_DIR or download from CHROMA_HF_DATASET."""
    if settings.chroma_dir:
        resolved = settings.chroma_dir.resolve()
        if resolved.is_dir():
            return resolved
        LOG.warning("Chroma: CHROMA_DIR=%s does not exist or is not a directory", resolved)
    hf = getattr(settings, "chroma_hf_dataset", None) or os.environ.get("CHROMA_HF_DATASET")
    if not hf:
        LOG.debug("Chroma: no CHROMA_DIR and no CHROMA_HF_DATASET")
        return None
    try:
        from huggingface_hub import snapshot_download
        cache = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))).resolve()
        dest = snapshot_download(repo_id=hf, repo_type="dataset", cache_dir=str(cache), local_dir=cache / "chroma_epstein")
        base = Path(dest)
        chroma_db = base / "chroma_db"
        if chroma_db.is_dir():
            return chroma_db
        return base
    except Exception as e:
        LOG.warning("Chroma: HF download failed: %s", e)
        return None


def get_chroma_chunks(query: str, k: int = CHROMA_QUERY_K, fetch_k: int = CHROMA_FETCH_K) -> List[str]:
    """
    Query the Chroma Epstein text store and return the text of the top-k chunks.
    Returns [] if Chroma is not configured or query fails.
    Chroma client/collection is cached so later searches are fast (no re-download, no re-load of embedding model).
    """
    settings = load_settings()
    base = _chroma_base_dir(settings)
    if not base:
        # Log why Chroma wasn't used (stderr so it shows in systemd journalctl)
        env_val = os.environ.get("CHROMA_DIR")
        msg = (
            f"Chroma: no base dir. CHROMA_DIR env={repr(env_val)}, "
            f"settings.chroma_dir={getattr(settings, 'chroma_dir', None)}"
        )
        LOG.warning(msg)
        print(msg, file=sys.stderr, flush=True)
        return []

    base_str = str(base.resolve())
    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        if base_str not in _chroma_collection_cache:
            client = chromadb.PersistentClient(path=base_str)
            ef = SentenceTransformerEmbeddingFunction(model_name=CHROMA_EMBED_MODEL)
            colls = client.list_collections()
            if not colls:
                LOG.warning("Chroma: no collections in %s", base_str)
                return []
            _chroma_collection_cache[base_str] = client.get_collection(name=colls[0].name, embedding_function=ef)
        coll = _chroma_collection_cache[base_str]
        res = coll.query(query_texts=[query], n_results=min(k, fetch_k))
        texts: List[str] = []
        if res and res.get("documents") and res["documents"][0]:
            for doc in res["documents"][0][:k]:
                if doc and isinstance(doc, str) and doc.strip():
                    texts.append(doc.strip())
        if not texts:
            msg = f"Chroma: query returned 0 chunks from {base_str}"
            LOG.warning(msg)
            print(msg, file=sys.stderr, flush=True)
        return texts
    except Exception as e:
        msg = f"Chroma: query failed: {e}"
        LOG.warning(msg)
        print(msg, file=sys.stderr, flush=True)
        return []


def description_from_chroma(query: str, max_tokens: int = 280) -> str | None:
    """
    Query Chroma for relevant document chunks and use Groq to produce a 3–4 sentence
    description answering the query. Returns None if Chroma or Groq is unavailable.
    """
    chunks = get_chroma_chunks(query, k=CHROMA_QUERY_K, fetch_k=CHROMA_FETCH_K)
    if not chunks:
        return None

    settings = load_settings()
    groq_key = getattr(settings, "groq_api_key", None) or os.environ.get("GROQ_API_KEY")
    if not groq_key:
        return None

    model = getattr(settings, "groq_summary_model", None) or os.environ.get("GROQ_SUMMARY_MODEL", "llama-3.3-70b-versatile")
    client = Groq(api_key=groq_key)
    context = "\n\n".join(chunks[:6])
    prompt = (
        "You are answering based only on the following text from Epstein case files (OCR from released documents).\n\n"
        f"From the files:\n{context}\n\n"
        f"User query: {query}\n\n"
        "Write a short paragraph of 3 to 4 sentences that directly answers the query using only the files above. "
        "Do not speculate or add information not in the files. If the files do not answer the query, say so clearly. Refer to the source as 'the files' or 'these files', not 'excerpts'."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text or None
    except Exception:
        return None
