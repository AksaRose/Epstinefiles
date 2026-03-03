"""
LanceDB storage for face-clustering pipeline: images table, faces table,
cluster_representatives.json, and cluster_names.json (user renames).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import lancedb
import pyarrow as pa

FACE_EMBED_DIM = 512
IMAGES_TABLE = "images"
FACES_TABLE = "faces"


def get_db(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(db_path))


def _images_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("dataset_id", pa.int32()),
            pa.field("source_file", pa.string()),
            pa.field("page_no", pa.int32()),
            pa.field("image_path", pa.string()),
            pa.field("image_blob", pa.binary()),
        ]
    )


def _faces_schema(vector_dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("face_id", pa.string()),
            pa.field("image_id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), vector_dim)),
            pa.field("cluster_id", pa.int32()),
            pa.field("bbox", pa.list_(pa.int32(), 4)),
        ]
    )


def get_or_create_images_table(db_path: Path, overwrite: bool = False):
    db = get_db(db_path)
    if overwrite and IMAGES_TABLE in db.list_tables():
        db.drop_table(IMAGES_TABLE)
    if IMAGES_TABLE in db.list_tables():
        return db.open_table(IMAGES_TABLE)
    schema = _images_schema()
    empty = pa.Table.from_arrays([pa.array([]) for _ in schema], schema=schema)
    return db.create_table(IMAGES_TABLE, data=empty, schema=schema, mode="overwrite")


def get_or_create_faces_table(db_path: Path, vector_dim: int = FACE_EMBED_DIM, overwrite: bool = False):
    db = get_db(db_path)
    if overwrite and FACES_TABLE in db.list_tables():
        db.drop_table(FACES_TABLE)
    if FACES_TABLE in db.list_tables():
        return db.open_table(FACES_TABLE)
    schema = _faces_schema(vector_dim)
    empty = pa.Table.from_arrays([pa.array([]) for _ in schema], schema=schema)
    return db.create_table(FACES_TABLE, data=empty, schema=schema, mode="overwrite")


def write_images_batch(table, rows: Iterable[Mapping[str, Any]]):
    data = list(rows)
    if not data:
        return
    table.add(data)


def write_faces_batch(table, rows: Iterable[Mapping[str, Any]]):
    data = list(rows)
    if not data:
        return
    table.add(data)


def _cluster_reps_path(lancedb_dir: Path) -> Path:
    return Path(lancedb_dir) / "cluster_representatives.json"


def _cluster_names_path(lancedb_dir: Path) -> Path:
    return Path(lancedb_dir) / "cluster_names.json"


def save_cluster_representatives(
    lancedb_dir: Path,
    representatives: List[tuple],
) -> None:
    """Save list of (cluster_id, image_id, bbox) for UI thumbnails."""
    path = _cluster_reps_path(lancedb_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {"cluster_id": cid, "image_id": iid, "bbox": bbox}
        for cid, iid, bbox in representatives
    ]
    path.write_text(json.dumps(data, indent=2))


def load_cluster_representatives(lancedb_dir: Path) -> List[Dict[str, Any]]:
    path = _cluster_reps_path(lancedb_dir)
    if not path.is_file():
        return []
    return json.loads(path.read_text())


def load_cluster_names(lancedb_dir: Path) -> Dict[str, str]:
    """cluster_id (as str) -> display name."""
    path = _cluster_names_path(lancedb_dir)
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def save_cluster_names(lancedb_dir: Path, names: Dict[str, str]) -> None:
    path = _cluster_names_path(lancedb_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(names, indent=2))
