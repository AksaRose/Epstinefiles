#!/usr/bin/env python3
"""Download the Epstein Chroma DB from Hugging Face to ./chroma_epstein (run from project root)."""
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = "devankit7873/EpsteinFiles-Vector-Embeddings-ChromaDB"
root = Path(__file__).resolve().parents[1]
local_dir = root / "chroma_epstein"
local_dir.mkdir(parents=True, exist_ok=True)
print("Downloading Chroma dataset from Hugging Face...")
path = snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(local_dir))
print("Downloaded to:", path)
chroma_db = Path(path) / "chroma_db"
if chroma_db.is_dir():
    print("Set in .env on the server: CHROMA_DIR=" + str(chroma_db))
else:
    print("Chroma DB path (if different):", path)
