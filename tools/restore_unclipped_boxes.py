"""Fix for the v1 extraction bug: v1 caches stored detector boxes clipped to the image, unlike the
official ByteTrack pipeline (and MOT GT, which extends beyond the frame). This tool re-runs only the
YOLOX detector, re-applies exactly the v1 keep-mask, verifies that clip(new boxes) == cached boxes
and scores are equal (alignment check), and writes a v2 cache whose `boxes` are the unclipped
detector output. MASA embeddings (computed on the in-image part of each box) and GMC are reused.
`boxes_clipped` keeps the v1 boxes for provenance.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from era_masa.data import sha256
from era_masa.detcache import load_det_cache
from era_masa.runlog import RunLog
from extract_detections import Detector


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, nargs="+", required=True, help="v1 cache dirs")
    p.add_argument("--dst-root", type=Path, default=ROOT / "cache_det_v2")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    with RunLog(a.output, {"src": [str(s) for s in a.src], "dst_root": str(a.dst_root)}) as log:
        detectors = {}
        for src in a.src:
            for path in sorted(src.glob("*.npz")):
                target = a.dst_root / src.name / path.name
                if target.exists():
                    log.message(f"exists, kept: {target}")
                    continue
                seq = load_det_cache(path)
                meta = seq.metadata
                det = detectors.setdefault(meta["detector"], Detector(meta["detector"], "cuda:0"))
                split_root = {"MOT17": ROOT / "data/MOT17-full/MOT17", "MOT20": ROOT / "data/MOT20-full/MOT20"}.get(meta["benchmark"])
                if split_root is not None:
                    img_dir = split_root / meta["split"] / seq.name / "img1"
                else:  # DanceTrack: sequence lives in one of the downloaded split folders
                    img_dir = next(d / seq.name / "img1" for d in (ROOT / "data/DanceTrack").iterdir() if (d / seq.name).exists())
                if meta["image_fingerprint"] is None:
                    raise ValueError("missing fingerprint")
                new_boxes = np.empty_like(seq.boxes)
                max_box_err = max_score_err = 0.0
                t0 = time.perf_counter()
                pool = ThreadPoolExecutor(4)
                read = lambda f: cv2.imread(str(img_dir / f"{f:06d}.jpg")) if (img_dir / f"{f:06d}.jpg").exists() \
                    else cv2.imread(str(img_dir / f"{f:08d}.jpg"))
                pending = {f: pool.submit(read, f) for f in range(1, min(seq.length, 8) + 1)}
                for frame in range(1, seq.length + 1):
                    img = pending.pop(frame).result()
                    if frame + 8 <= seq.length:
                        pending[frame + 8] = pool.submit(read, frame + 8)
                    b, s = det(img, meta["conf_floor"])
                    torch.cuda.synchronize()
                    h, w = img.shape[:2]
                    c = b.copy()
                    c[:, [0, 2]] = c[:, [0, 2]].clip(0, w - 1)
                    c[:, [1, 3]] = c[:, [1, 3]].clip(0, h - 1)
                    keep = (c[:, 2] - c[:, 0] > 1) & (c[:, 3] - c[:, 1] > 1) & (s >= meta["conf_floor"])
                    sl = seq.indices(frame)
                    if keep.sum() != sl.stop - sl.start:
                        raise RuntimeError(f"{seq.name}/{frame}: {keep.sum()} kept vs {sl.stop - sl.start} cached")
                    if keep.any():
                        max_box_err = max(max_box_err, float(np.abs(c[keep] - seq.boxes[sl]).max()))
                        max_score_err = max(max_score_err, float(np.abs(s[keep] - seq.scores[sl]).max()))
                    new_boxes[sl] = b[keep]
                if max_box_err > 0.05 or max_score_err > 1e-4:
                    raise RuntimeError(f"{seq.name}: alignment check failed box {max_box_err} score {max_score_err}")
                target.parent.mkdir(parents=True, exist_ok=True)
                meta2 = {k: v for k, v in meta.items() if k not in ("cache_path", "cache_sha256")}
                meta2.update(schema=2, boxes="unclipped detector output (v2 fix)", v1_cache_sha256=meta["cache_sha256"],
                             alignment_max_box_err=max_box_err, alignment_max_score_err=max_score_err)
                tmp = target.with_suffix(".partial")
                with tmp.open("wb") as f:
                    np.savez_compressed(f, metadata=np.array(json.dumps(meta2)), frames=seq.frames,
                                        boxes=new_boxes.astype(np.float32), boxes_clipped=seq.boxes.astype(np.float32),
                                        scores=seq.scores.astype(np.float32), embeddings=seq.embeddings.astype(np.float32),
                                        gmc=seq.gmc.astype(np.float32))
                tmp.replace(target)
                changed = float((np.abs(new_boxes - seq.boxes) > 0.5).any(axis=1).mean())
                log.save(f"{src.name}-{seq.name}.json", {**meta2, "cache_sha256": sha256(target),
                         "fraction_boxes_changed": changed, "seconds": time.perf_counter() - t0})
                log.message(f"{src.name}/{seq.name}: aligned (box err {max_box_err:.4f}, score err {max_score_err:.2e}); "
                            f"{changed:.1%} boxes were clipped in v1; {seq.length / (time.perf_counter() - t0):.1f} FPS")


if __name__ == "__main__":
    main()
