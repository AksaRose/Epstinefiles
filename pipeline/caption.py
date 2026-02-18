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
            f"The following image contains people identified as: {names}. "
            "Write one concise caption that uses their names and clearly describes "
            "the setting and what is happening."
        )
    return (
        "Write one concise caption for this image describing the scene, setting, "
        "and what people (if any) are doing."
    )


def caption_image(
    image_path: Path,
    celebrity_names: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    model: str = "qwen/qwen3-vl-32b-instruct",
) -> str:
    """
    Generate a caption for an image using Qwen3-VL via Together AI.
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
        max_tokens=128,
    )
    choice = resp.choices[0]
    text = choice.message.content
    if isinstance(text, list):
        # Some Together responses may structure content as list of parts.
        parts = [p.get("text", "") for p in text if isinstance(p, dict)]
        return " ".join(p.strip() for p in parts if p.strip())
    return str(text).strip()

