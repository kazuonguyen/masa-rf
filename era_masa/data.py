import configparser
from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Sequence:
    name: str
    length: int
    fps: float
    frames: np.ndarray
    boxes: np.ndarray
    embeddings: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    gt_ids: np.ndarray | None
    visibility: np.ndarray | None
    metadata: dict

    def indices(self, frame):
        # Cache is sorted by frame. Empty frames must still reach the tracker.
        return slice(np.searchsorted(self.frames, frame, "left"), np.searchsorted(self.frames, frame, "right"))


def load_cache(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata"]))
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", metadata["sequence"]):
            raise ValueError("Sequence name must be a safe filename")
        if metadata["length"] < 1 or metadata["fps"] <= 0:
            raise ValueError("Sequence length and fps must be positive")
        frames = cache["frames"].astype(int)
        boxes = cache["boxes"].astype(float)
        embeddings = cache["embeddings"].astype(float)
        n = len(frames)
        if metadata.get("schema") != 1:
            raise ValueError("Unsupported embedding cache schema")
        if embeddings.shape != (n, 256) or boxes.shape != (n, 4):
            raise ValueError("Cache requires Nx256 MASA embeddings and Nx4 boxes")
        if np.any(np.diff(frames) < 0) or np.any(frames < 1) or np.any(frames > metadata["length"]):
            raise ValueError("Invalid or unsorted frame numbers")
        scores, labels = cache["scores"].copy(), cache["labels"].copy()
        gt_ids = cache["gt_ids"].copy() if "gt_ids" in cache else None
        visibility = cache["visibility"].copy() if "visibility" in cache else None
        for array in (scores, labels, gt_ids, visibility):
            if array is not None and array.shape != (n,):
                raise ValueError("Cache row metadata is misaligned")
        metadata = {**metadata, "cache_path": str(path.resolve()), "cache_sha256": sha256(path)}
        return Sequence(metadata["sequence"], int(metadata["length"]), float(metadata["fps"]),
                        frames, boxes, embeddings, scores, labels, gt_ids, visibility, metadata)


def save_cache(path, *, metadata, frames, boxes, embeddings, scores, labels, gt_ids=None, visibility=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(metadata=np.array(json.dumps({"schema": 1, **metadata})), frames=frames,
                  boxes=boxes, embeddings=embeddings, scores=scores, labels=labels)
    if gt_ids is not None:
        arrays["gt_ids"] = gt_ids
    if visibility is not None:
        arrays["visibility"] = visibility
    # Write through a file handle so np.savez does not silently alter filenames.
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)


def read_ground_truth(sequence_path, benchmark, policy="valid"):
    """GT is an offline data source; its identities never enter tracker.update."""
    sequence_path = Path(sequence_path)
    info = configparser.ConfigParser()
    info.read(sequence_path / "seqinfo.ini")
    gt_path = sequence_path / "gt" / "gt.txt"
    rows = np.loadtxt(gt_path, delimiter=",", ndmin=2)
    if not np.isfinite(rows).all():
        raise ValueError("Nonfinite values in GT annotations")
    if rows.shape[1] < 7:
        raise ValueError("GT must have at least seven MOT columns")
    valid = (rows[:, 4] > 0) & (rows[:, 5] > 0)
    if policy == "valid":
        valid &= rows[:, 6] > 0
        if benchmark in ("MOT16", "MOT17", "MOT20"):
            if rows.shape[1] < 8:
                raise ValueError("MOT16/17/20 GT must contain a class column")
            valid &= rows[:, 7] == 1  # pedestrian, excludes distractor/ignored classes
    rows = rows[valid]
    rows = rows[np.argsort(rows[:, 0], kind="stable")]
    section = info["Sequence"]
    boxes = rows[:, 2:6].copy()
    boxes[:, 2:] += boxes[:, :2]
    visibility = rows[:, 8] if benchmark in ("MOT16", "MOT17", "MOT20", "CTMC-v1") and rows.shape[1] > 8 else None
    return dict(rows=rows, boxes=boxes, frames=rows[:, 0].astype(int), gt_ids=rows[:, 1].astype(int),
                visibility=visibility, length=int(section["seqLength"]), fps=float(section["frameRate"]),
                metadata={"sequence": sequence_path.name, "benchmark": benchmark,
                    "length": int(section["seqLength"]), "fps": float(section["frameRate"]),
                    "gt_path": str(gt_path.resolve()), "gt_sha256": sha256(gt_path), "gt_policy": policy})
