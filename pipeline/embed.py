from __future__ import annotations

import os
from typing import Iterable, List, Optional

from together import Together

# Together API has a request size limit; sending too many texts at once returns 413 Payload Too Large.
# Sub-batch so each request stays under the limit (e.g. 24 texts per call).
_MAX_TEXTS_PER_EMBED_REQUEST = int(os.environ.get("EMBED_REQUEST_BATCH_SIZE", "24"))


def _get_client(api_key: Optional[str] = None) -> Together:
    return Together(api_key=api_key)


def embed_texts(
    texts: Iterable[str],
    api_key: Optional[str] = None,
    model: str = "Alibaba-NLP/gte-modernbert-base",
) -> List[List[float]]:
    """
    Batch embed a list of texts using Together embeddings API.
    Splits into sub-batches to avoid 413 Payload Too Large (API request size limit).
    """
    client = _get_client(api_key)
    inputs = [t for t in texts]
    if not inputs:
        return []
    vectors: List[List[float]] = []
    for i in range(0, len(inputs), _MAX_TEXTS_PER_EMBED_REQUEST):
        chunk = inputs[i : i + _MAX_TEXTS_PER_EMBED_REQUEST]
        resp = client.embeddings.create(model=model, input=chunk)
        for item in resp.data:
            vectors.append(list(item.embedding))
    return vectors


def embed_query(
    text: str,
    api_key: Optional[str] = None,
    model: str = "Alibaba-NLP/gte-modernbert-base",
) -> List[float]:
    return embed_texts([text], api_key=api_key, model=model)[0]

