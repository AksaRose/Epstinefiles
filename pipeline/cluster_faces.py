"""
Face embedding clustering with DBSCAN.
Input: list of (image_id, face_index, embedding, bbox).
Output: same list with cluster_id per face; cluster representatives for UI thumbnails.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.cluster import DBSCAN


@dataclass
class FaceRecord:
    image_id: str
    face_index: int
    embedding: np.ndarray
    bbox: List[int]
    cluster_id: int = -1


def cluster_faces(
    face_records: List[FaceRecord],
    eps: float = 0.58,
    min_samples: int = 1,
) -> Tuple[List[FaceRecord], List[Tuple[int, str, List[int]]]]:
    """
    Run DBSCAN on face embeddings. Modifies face_records in place with cluster_id.
    Returns (face_records, representatives) where representatives is
    list of (cluster_id, image_id, bbox) for first face in each cluster (for UI thumbnail).
    With min_samples=1 every face gets a cluster (no noise); with min_samples>=2 only groups of 2+ are clusters.
    """
    if not face_records:
        return face_records, []

    embeddings = np.stack([r.embedding for r in face_records], axis=0).astype(np.float32)
    clustering = DBSCAN(metric="cosine", eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(embeddings)

    for i, rec in enumerate(face_records):
        rec.cluster_id = int(labels[i])

    # Representatives: first face per cluster (excluding noise -1)
    seen: set[int] = set()
    representatives: List[Tuple[int, str, List[int]]] = []
    for rec in face_records:
        if rec.cluster_id < 0:
            continue
        if rec.cluster_id not in seen:
            seen.add(rec.cluster_id)
            representatives.append((rec.cluster_id, rec.image_id, rec.bbox))

    return face_records, representatives
