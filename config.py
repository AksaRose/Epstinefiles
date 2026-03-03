from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (directory containing config.py) so it works when cwd differs (e.g. systemd/Streamlit)
_project_root = Path(__file__).resolve().parent
load_dotenv(_project_root / ".env")


@dataclass
class Settings:
    base_dir: Path
    images_base_dir: Path
    lancedb_dir: Path
    table_name: str
    faces_db_dir: Path
    chroma_dir: Path | None  # Local path to Chroma DB (Epstein text RAG); None = disabled
    chroma_hf_dataset: str | None  # If set, download this HF dataset and use its chroma_db/
    face_backend: str  # "insightface" (buffalo_l) or "deepface"
    face_det_thresh: float  # InsightFace detection confidence threshold; lower = more faces (e.g. 0.35)
    face_distance_threshold: float  # max cosine distance to accept a match (e.g. 0.8)
    dbscan_eps: float  # DBSCAN eps (cosine); higher = merge more (same person across photos), lower = fewer merges
    dbscan_min_samples: int  # DBSCAN min_samples; 1 = every face gets a cluster (no noise), 2+ = only groups
    # Optional for face-only mode (caption/embed/RAG/LLM):
    together_api_key: str | None
    embed_model: str
    qwen_model: str
    embed_batch_size: int  # larger = fewer embed API calls (e.g. 64 or 128)
    qwen_concurrency: int  # parallel caption requests per chunk
    caption_max_tokens: int  # lower = cheaper captions (e.g. 64–96)
    groq_api_key: str | None
    groq_summary_model: str


def load_settings() -> Settings:
    cwd = Path.cwd()
    base_dir = Path(os.getenv("EPSTEIN_PDFS_DIR", cwd / "epstein_pdfs"))
    images_base = base_dir
    lancedb_dir = Path(os.getenv("LANCEDB_DIR", cwd / "lancedb"))
    faces_db_dir = Path(os.getenv("FACES_DB_DIR", cwd / "my_db"))
    chroma_dir_env = os.getenv("CHROMA_DIR")
    chroma_dir = Path(chroma_dir_env) if chroma_dir_env else None
    chroma_hf_dataset = os.getenv("CHROMA_HF_DATASET") or None  # e.g. devankit7873/EpsteinFiles-Vector-Embeddings-ChromaDB

    return Settings(
        base_dir=base_dir,
        images_base_dir=images_base,
        lancedb_dir=lancedb_dir,
        table_name=os.getenv("LANCEDB_TABLE", "epstein_images"),
        faces_db_dir=faces_db_dir,
        chroma_dir=chroma_dir,
        chroma_hf_dataset=chroma_hf_dataset,
        face_backend=os.getenv("FACE_BACKEND", "insightface"),
        face_det_thresh=float(os.getenv("FACE_DET_THRESH", "0.35")),
        face_distance_threshold=float(os.getenv("FACE_DISTANCE_THRESHOLD", "0.8")),
        dbscan_eps=float(os.getenv("DBSCAN_EPS", "0.58")),
        dbscan_min_samples=int(os.getenv("DBSCAN_MIN_SAMPLES", "1")),
        together_api_key=os.getenv("TOGETHER_API_KEY"),
        embed_model=os.getenv("EMBED_MODEL", "Alibaba-NLP/gte-modernbert-base"),
        qwen_model=os.getenv("QWEN_MODEL", "qwen/qwen3-vl-32b-instruct"),
        embed_batch_size=int(os.getenv("EMBED_BATCH_SIZE", "64")),
        qwen_concurrency=int(os.getenv("QWEN_CONCURRENCY", "8")),
        caption_max_tokens=int(os.getenv("CAPTION_MAX_TOKENS", "96")),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        groq_summary_model=os.getenv("GROQ_SUMMARY_MODEL", "llama-3.3-70b-versatile"),
    )

