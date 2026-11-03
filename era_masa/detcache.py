"""Detector-mode cache: YOLOX boxes + scores, MASA embeddings and camera motion.

Schema 2 is separate from the fixed-GT-box schema 1: no GT identities are
stored, so tracker code cannot read them by construction.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .data import sha256


@dataclass
class DetSequence:
    name: str
    length: int
    fps: float
    width: int
    height: int
    frames: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray
    embeddings: np.ndarray
    gmc: np.ndarray  # length x 2 x 3 affine, previous frame -> current frame
    metadata: dict

    def indices(self, frame):
        return slice(np.searchsorted(self.frames, frame, "left"), np.searchsorted(self.frames, frame, "right"))


def save_det_cache(path, *, metadata, frames, boxes, scores, embeddings, gmc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, metadata=np.array(json.dumps({"schema": 2, **metadata})),
                            frames=frames, boxes=boxes.astype(np.float32), scores=scores.astype(np.float32),
                            embeddings=embeddings.astype(np.float32), gmc=gmc.astype(np.float32))
    tmp.replace(path)


def load_det_cache(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as cache:
        meta = json.loads(str(cache["metadata"]))
        if meta.get("schema") != 2:
            raise ValueError("Not a detector cache")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", meta["sequence"]):
            raise ValueError("Unsafe sequence name")
        frames = cache["frames"].astype(int)
        n = len(frames)
        boxes, scores = cache["boxes"].astype(float), cache["scores"].astype(float)
        embeddings, gmc = cache["embeddings"].astype(float), cache["gmc"].astype(float)
    if boxes.shape != (n, 4) or scores.shape != (n,) or embeddings.shape != (n, 256):
        raise ValueError("Misaligned detector cache")
    if gmc.shape != (meta["length"], 2, 3) or np.any(np.diff(frames) < 0):
        raise ValueError("Invalid camera motion or frame order")
    meta = {**meta, "cache_path": str(path.resolve()), "cache_sha256": sha256(path)}
    return DetSequence(meta["sequence"], int(meta["length"]), float(meta["fps"]), int(meta["width"]),
                       int(meta["height"]), frames, boxes, scores, embeddings, gmc, meta)
