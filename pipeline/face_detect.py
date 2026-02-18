from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
from deepface import DeepFace


@dataclass
class FaceAnalysisResult:
    has_faces: bool
    celebrities: List[str]


def _identity_to_name(identity_path: str, db_path: Path) -> str:
    """Extract person name from path like my_db/Jeffrey_Epstein/je1.jpeg -> Jeffrey_Epstein."""
    p = Path(identity_path)
    # Parent of the image file is the person folder
    if p.parent.name and p.parent != db_path:
        return p.parent.name
    return p.stem


def analyze_image(
    image_path: Path,
    faces_db_dir: Path | None = None,
) -> Tuple[FaceAnalysisResult, bytes]:
    """
    Run DeepFace detection + recognition against a local my_db-style folder.

    Returns (FaceAnalysisResult, image_bytes).
    """
    img = cv2.imread(str(image_path))
    if img is None:
        data = image_path.read_bytes()
        import numpy as np
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    success, buf = cv2.imencode(".png", img)
    if not success:
        raise ValueError(f"Failed to encode image as PNG: {image_path}")
    image_bytes = buf.tobytes()

    has_faces = False
    celebrities: List[str] = []

    if faces_db_dir is not None and faces_db_dir.is_dir():
        try:
            # DeepFace.find returns list of DataFrame (one per detected face)
            result = DeepFace.find(
                img_path=str(image_path),
                db_path=str(faces_db_dir),
                model_name="ArcFace",
                detector_backend="opencv",
                enforce_detection=False,
                silent=True,
            )
            # DeepFace.find returns list of DataFrames (one per face) or a single DataFrame
            if result is not None:
                dfs = result if isinstance(result, list) else [result]
                if dfs:
                    has_faces = True
                    names: set[str] = set()
                    for df in dfs:
                        if df is None or (hasattr(df, "empty") and df.empty):
                            continue
                        row = df.iloc[0]
                        if row.get("distance", 1.0) > 0.6:
                            continue
                        identity = row.get("identity")
                        if isinstance(identity, str):
                            names.add(_identity_to_name(identity, faces_db_dir))
                    celebrities = sorted(names)
        except Exception:
            # Fallback: only detect, no recognition
            try:
                objs = DeepFace.extract_faces(str(image_path), detector_backend="opencv", enforce_detection=False)
                has_faces = bool(objs and len(objs) > 0)
            except Exception:
                pass
    else:
        try:
            objs = DeepFace.extract_faces(str(image_path), detector_backend="opencv", enforce_detection=False)
            has_faces = bool(objs and len(objs) > 0)
        except Exception:
            pass

    return FaceAnalysisResult(has_faces=has_faces, celebrities=celebrities), image_bytes
