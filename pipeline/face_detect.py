from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from deepface import DeepFace

# Optional: InsightFace (better accuracy). Install: pip install insightface onnxruntime
# If install fails on Mac (e.g. "cmath not found"): run xcode-select --install, or use
# --backend deepface --model Facenet512 for a stronger DeepFace model without compiling.
_INSIGHTFACE_APP = None

def _get_insightface_app():
    global _INSIGHTFACE_APP
    if _INSIGHTFACE_APP is None:
        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=0, det_size=(640, 640))
            _INSIGHTFACE_APP = app
        except Exception:
            raise RuntimeError("InsightFace not available. Install: pip install insightface onnxruntime")
    return _INSIGHTFACE_APP


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two vectors (0 = same, 2 = opposite)."""
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(1.0 - np.dot(a, b))


# Cache: db_path -> list of (identity_name, embedding)
_INSIGHTFACE_DB_CACHE: dict[str, List[Tuple[str, np.ndarray]]] = {}


def _build_insightface_db(faces_db_dir: Path, app) -> List[Tuple[str, np.ndarray]]:
    key = str(faces_db_dir.resolve())
    if key in _INSIGHTFACE_DB_CACHE:
        return _INSIGHTFACE_DB_CACHE[key]
    entries: List[Tuple[str, np.ndarray]] = []
    for person_dir in sorted(faces_db_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        for img_path in person_dir.iterdir():
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            faces = app.get(img)
            for f in faces:
                if hasattr(f, "embedding") and f.embedding is not None:
                    entries.append((name, f.embedding))
    _INSIGHTFACE_DB_CACHE[key] = entries
    return entries


def _analyze_insightface(
    img: np.ndarray,
    faces_db_dir: Path,
    distance_threshold: float,
) -> Tuple[bool, List[str], List[Tuple[str, float]]]:
    """Returns (has_faces, celebrities, match_distances). match_distances = one (best_name, distance) per detected face."""
    app = _get_insightface_app()
    db = _build_insightface_db(faces_db_dir, app)
    if not db:
        return False, [], []
    faces = app.get(img)
    if not faces:
        return False, [], []
    names: set[str] = set()
    match_distances: List[Tuple[str, float]] = []
    for f in faces:
        if not hasattr(f, "embedding") or f.embedding is None:
            continue
        best_dist = float("inf")
        best_name: str | None = None
        for name, ref_emb in db:
            d = _cosine_distance(f.embedding, ref_emb)
            if d < best_dist:
                best_dist = d
                best_name = name
        match_distances.append((best_name or "unknown", best_dist))
        if best_name is not None and best_dist <= distance_threshold:
            names.add(best_name)
    return True, sorted(names), match_distances


@dataclass
class FaceAnalysisResult:
    has_faces: bool
    celebrities: List[str]
    match_distances: List[Tuple[str, float]] | None = None  # (identity, distance) per face when using InsightFace


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
    distance_threshold: float = 0.8,
    model_name: str = "ArcFace",
    backend: str = "insightface",
) -> Tuple[FaceAnalysisResult, bytes]:
    """
    Run face detection + recognition against a local my_db-style folder.
    backend: "deepface" or "insightface" (better accuracy; pip install insightface onnxruntime).
    Returns (FaceAnalysisResult, image_bytes).
    """
    img = cv2.imread(str(image_path))
    if img is None:
        data = image_path.read_bytes()
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
        if backend == "insightface":
            try:
                has_faces, celebrities, match_distances = _analyze_insightface(img, faces_db_dir, distance_threshold)
                return FaceAnalysisResult(has_faces=has_faces, celebrities=celebrities, match_distances=match_distances), image_bytes
            except RuntimeError:
                pass
        try:
            result = DeepFace.find(
                img_path=str(image_path),
                db_path=str(faces_db_dir),
                model_name=model_name,
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
                        distance = float(row.get("distance", 1.0))
                        if distance > distance_threshold:
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test face detection and optional recognition")
    parser.add_argument("image", type=Path, help="Path to image file")
    parser.add_argument("--db", type=Path, default=None, help="Faces DB dir (e.g. my_db) for recognition; omit for detection only")
    parser.add_argument("--threshold", type=float, default=0.8, help="Max distance to accept a match (default 0.8 for InsightFace)")
    parser.add_argument("--model", type=str, default="ArcFace", choices=["ArcFace", "Facenet", "Facenet512", "VGG-Face", "OpenFace", "DeepFace", "DeepID", "Dlib"], help="Recognition model (DeepFace only)")
    parser.add_argument("--backend", type=str, default="insightface", choices=["deepface", "insightface"], help="Backend: insightface buffalo_l (default) or deepface")
    parser.add_argument("--verbose", action="store_true", help="Print distance for each detected face (DeepFace only)")
    args = parser.parse_args()

    if not args.image.is_file():
        raise SystemExit(f"Not a file: {args.image}")

    if args.verbose and args.db and args.db.is_dir() and args.backend == "deepface":
        try:
            r = DeepFace.find(img_path=str(args.image), db_path=str(args.db), model_name=args.model, detector_backend="opencv", enforce_detection=False, silent=True)
            dfs = r if isinstance(r, list) else [r]
            for i, df in enumerate(dfs):
                if df is not None and not (hasattr(df, "empty") and df.empty):
                    row = df.iloc[0]
                    dist = row.get("distance", None)
                    ident = row.get("identity", "")
                    name = _identity_to_name(ident, args.db) if isinstance(ident, str) else ident
                    print(f"  face {i+1}: best match = {name} (distance = {dist})")
        except Exception as e:
            print("  verbose failed:", e)

    result, _ = analyze_image(args.image, faces_db_dir=args.db, distance_threshold=args.threshold, model_name=args.model, backend=args.backend)
    if result.match_distances:
        print("Distances (InsightFace buffalo_l, cosine):")
        for i, (name, dist) in enumerate(result.match_distances):
            print(f"  face {i+1}: best match = {name} (distance = {dist:.4f})")
    print("has_faces:", result.has_faces)
    print("celebrities:", result.celebrities)
