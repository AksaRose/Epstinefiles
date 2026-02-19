from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    base_dir: Path
    images_base_dir: Path
    lancedb_dir: Path
    table_name: str
    faces_db_dir: Path
    together_api_key: str | None
    embed_model: str
    qwen_model: str
    embed_batch_size: int
    qwen_concurrency: int
    groq_api_key: str | None
    groq_summary_model: str


def load_settings() -> Settings:
    cwd = Path.cwd()
    base_dir = Path(os.getenv("EPSTEIN_PDFS_DIR", cwd / "epstein_pdfs"))
    images_base = base_dir
    lancedb_dir = Path(os.getenv("LANCEDB_DIR", cwd / "lancedb"))
    faces_db_dir = Path(os.getenv("FACES_DB_DIR", cwd / "my_db"))

    return Settings(
        base_dir=base_dir,
        images_base_dir=images_base,
        lancedb_dir=lancedb_dir,
        table_name=os.getenv("LANCEDB_TABLE", "epstein_images"),
        faces_db_dir=faces_db_dir,
        together_api_key=os.getenv("TOGETHER_API_KEY"),
        embed_model=os.getenv("EMBED_MODEL", "Alibaba-NLP/gte-modernbert-base"),
        qwen_model=os.getenv("QWEN_MODEL", "qwen/qwen3-vl-32b-instruct"),
        embed_batch_size=int(os.getenv("EMBED_BATCH_SIZE", "32")),
        qwen_concurrency=int(os.getenv("QWEN_CONCURRENCY", "5")),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        groq_summary_model=os.getenv("GROQ_SUMMARY_MODEL", "llama-3.3-70b-versatile"),
    )

