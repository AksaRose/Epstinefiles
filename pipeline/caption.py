from __future__ import annotations

import base64
from pathlib import Path
from typing import Iterable, List, Optional

from together import Together


def _get_client(api_key: Optional[str] = None) -> Together:
    return Together(api_key=api_key)


def _build_prompt(celebrity_names: List[str] | None) -> str:
    if celebrity_names:
        names = ", ".join(celebrity_names)
        return (
            f"These people were identified in this image by face recognition only: {names}. "
            "Write one concise caption describing only the setting and what is happening. "
            "Do not name, identify, or guess any other person. Only the names above are confirmed. "
            "Describe only what is visible; do not infer or speculate about events or context not shown."
        )
    return (
        "Write one concise caption for this image describing the scene and setting only. "
        "Do not identify or name any person. Describe only what is visible; do not infer or speculate."
    )


def caption_image(
    image_path: Path,
    celebrity_names: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    model: str = "qwen/qwen3-vl-32b-instruct",
    max_tokens: int = 96,
) -> str:
    """
    Generate a caption for an image using Qwen3-VL via Together AI.
    Lower max_tokens reduces cost (e.g. 64–96 for short captions).
    """
    client = _get_client(api_key)
    img_bytes = image_path.read_bytes()
    b64 = base64.b64encode(img_bytes).decode("utf-8")

    prompt = _build_prompt(celebrity_names or [])

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
        max_tokens=max_tokens,
        temperature=0,
    )
    choice = resp.choices[0]
    text = choice.message.content
    if isinstance(text, list):
        # Some Together responses may structure content as list of parts.
        parts = [p.get("text", "") for p in text if isinstance(p, dict)]
        return " ".join(p.strip() for p in parts if p.strip())
    return str(text).strip()

