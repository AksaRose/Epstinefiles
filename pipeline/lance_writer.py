from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Mapping

import lancedb
import numpy as np
import pyarrow as pa


def get_db(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(db_path))


def _schema(vector_dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("dataset_id", pa.int32()),
            pa.field("source_file", pa.string()),
            pa.field("page_no", pa.int32()),
            pa.field("image_path", pa.string()),
            pa.field("image_blob", pa.binary()),
            pa.field("caption", pa.string()),
            pa.field("celebrities", pa.list_(pa.string())),
            pa.field("has_faces", pa.bool_()),
            pa.field("vector", pa.list_(pa.float32(), vector_dim)),
        ]
    )


def get_or_create_table(db_path: Path, table_name: str, vector_dim: int):
    db = get_db(db_path)
    if table_name in db.table_names():
        return db.open_table(table_name)
    # Create empty table with schema
    schema = _schema(vector_dim)
    empty_batch = pa.Table.from_arrays([pa.array([]) for _ in schema], schema=schema)
    return db.create_table(table_name, data=empty_batch, schema=schema, mode="overwrite")


def write_batch(table, rows: Iterable[Mapping]):
    data = list(rows)
    if not data:
        return
    table.add(data)

