from __future__ import annotations

from typing import Iterable, List, Optional

from together import Together


def _get_client(api_key: Optional[str] = None) -> Together:
    return Together(api_key=api_key)


def embed_texts(
    texts: Iterable[str],
    api_key: Optional[str] = None,
    model: str = "Alibaba-NLP/gte-modernbert-base",
) -> List[List[float]]:
    """
    Batch embed a list of texts using Together embeddings API.
    """
    client = _get_client(api_key)
    inputs = [t for t in texts]
    if not inputs:
        return []
    resp = client.embeddings.create(model=model, input=inputs)
    # Assume OpenAI-compatible shape: data list with embedding field
    vectors: List[List[float]] = []
    for item in resp.data:
        vectors.append(list(item.embedding))
    return vectors


def embed_query(
    text: str,
    api_key: Optional[str] = None,
    model: str = "Alibaba-NLP/gte-modernbert-base",
) -> List[float]:
    return embed_texts([text], api_key=api_key, model=model)[0]

