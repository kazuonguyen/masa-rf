"""Run one tracker on detector caches, post-process uniformly, evaluate with TrackEval.

Uniform output protocol for every method (ByteTrack tools/track.py):
  keep boxes with area > min_box_area and, for MOT benchmarks, w/h <= 1.6.
Optional linear interpolation of gaps <= 20 frames is evaluated as a separate
tracker ("<name>-interp"), never mixed into the raw numbers.
"""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
from .detcache import load_det_cache
from .evaluation import flatten_metrics, load_trackeval
from .runlog import RunLog

ROOT = Path(__file__).resolve().parents[1]


CLIP_BENCHMARKS = ("MOT20", "DanceTrack")


def clip_boxes(boxes, width, height):
    """Clip to [0, W-1] x [0, H-1] (identical to the v1 cache convention)."""
    b = np.array(boxes, dtype=float, copy=True).reshape(-1, 4)
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, width - 1)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, height - 1)
    return b


def postfilter(rows, min_area, aspect):
    if not len(rows):
        return rows
    w, h = rows[:, 4] - rows[:, 2], rows[:, 5] - rows[:, 3]
    keep = (w * h > min_area) & (w > 0) & (h > 0)
    if aspect:
        keep &= w / np.maximum(h, 1e-9) <= 1.6
    return rows[keep]


def interpolate(rows, max_gap=20):
    """rows: frame,id,x1,y1,x2,y2. Linear fill of gaps <= max_gap inside each track."""
    if not len(rows):
        return rows
    out = [rows]
    for tid in np.unique(rows[:, 1]):
        tr = rows[rows[:, 1] == tid]
        tr = tr[np.argsort(tr[:, 0])]
        for a, b in zip(tr[:-1], tr[1:]):
            gap = int(b[0] - a[0])
            if 1 < gap <= max_gap:
                for k in range(1, gap):
                    f = k / gap
                    out.append(np.r_[a[0] + k, tid, a[2:] + f * (b[2:] - a[2:])][None])
    rows = np.concatenate(out)
    return rows[np.lexsort((rows[:, 1], rows[:, 0]))]


def write_mot(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(f"{int(r[0])},{int(r[1])},{r[2]:.2f},{r[3]:.2f},{r[4]-r[2]:.2f},{r[5]-r[3]:.2f},1,-1,-1,-1\n")


def evaluate(gt_root, trackers_root, names, seq_lengths, benchmark):
    metrics, Dataset, _ = load_trackeval(ROOT / "third_party/TrackEval")
    ds = Dataset({"GT_FOLDER": str(gt_root), "TRACKERS_FOLDER": str(trackers_root), "TRACKERS_TO_EVAL": names,
                  "SEQ_INFO": dict(seq_lengths), "SKIP_SPLIT_FOL": True, "DO_PREPROC": True,
                  "PRINT_CONFIG": False, "BENCHMARK": "MOT20" if benchmark == "MOT20" else "MOT17"})
    rows = []
    for name in names:
        per = {m.get_name(): {} for m in metrics}
        for seq in seq_lengths:
            data = ds.get_preprocessed_seq_data(ds.get_raw_seq_data(name, seq), "pedestrian")
            res = {}
            for m in metrics:
                res[m.get_name()] = per[m.get_name()][seq] = m.eval_sequence(data)
            rows.append({"tracker": name, "sequence": seq, **flatten_metrics(res)})
        comb = {m.get_name(): m.combine_sequences(per[m.get_name()]) for m in metrics}
        rows.append({"tracker": name, "sequence": "COMBINED", **flatten_metrics(comb)})
    return rows


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--params", default="{}")
    p.add_argument("--name")
    p.add_argument("--caches", type=Path, required=True)
    p.add_argument("--gt-root", type=Path, help="prepare_gt output; omit for test splits")
    p.add_argument("--benchmark", required=True, choices=("MOT17", "MOT20", "DanceTrack"))
    p.add_argument("--sequences", nargs="+")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--min-box-area", type=float, default=10)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no-eval", action="store_true", help="skip TrackEval (timing-only runs)")
    p.add_argument("--clip", choices=("auto", "on", "off"), default="auto",
                   help="clip detector boxes to the image; auto follows the benchmark's GT convention "
                        "(MOT17 GT extends beyond the frame -> off; MOT20 and DanceTrack GT are in-frame -> on)")
    a = p.parse_args(argv)
    params = json.loads(a.params)
    name = a.name or a.method
    mot20 = a.benchmark == "MOT20"
    aspect = a.benchmark != "DanceTrack"
    clip = a.clip == "on" or (a.clip == "auto" and a.benchmark in CLIP_BENCHMARKS)
    if a.gt_root:
        info = json.loads((a.gt_root / "seqinfo.json").read_text())["sequences"]
        if a.sequences:
            info = {k: v for k, v in info.items() if k in a.sequences}
    else:
        info = None
    caches = sorted(a.caches.glob("*.npz"))
    if a.sequences:
        caches = [c for c in caches if c.stem in a.sequences]
    if info is not None:
        caches = [c for c in caches if c.stem in info]
        missing = set(info) - {c.stem for c in caches}
        if missing:
            raise FileNotFoundError(f"Missing caches: {sorted(missing)}")
    from .det_baselines import make
    with RunLog(a.output, {**vars(a), "caches": str(a.caches), "gt_root": str(a.gt_root), "output": str(a.output),
                            "resolved_params": params, "clip_boxes": clip}) as log:
        lengths, timing = {}, {}
        for path in caches:
            seq = load_det_cache(path)
            offset = info[seq.name]["offset"] if info else 0
            end = offset + info[seq.name]["length"] if info else seq.length
            tracker = make(a.method, params, seq.fps, mot20)
            log.event("sequence_started", sequence=seq.name, frames=end - offset, offset=offset,
                      cache_sha256=seq.metadata["cache_sha256"])
            rows, total, seen = [], 0.0, set()
            frame_log = (a.output / "logs" / f"{seq.name}-frames.csv")
            frame_log.parent.mkdir(parents=True, exist_ok=True)
            with frame_log.open("w", newline="", buffering=1) as fl:
                fw = csv.writer(fl)
                stats_keys = None
                for frame in range(offset + 1, end + 1):
                    sl = seq.indices(frame)
                    warp = seq.gmc[frame - 1] if frame > offset + 1 else np.eye(2, 3)
                    t0 = time.perf_counter()
                    boxes = clip_boxes(seq.boxes[sl], seq.width, seq.height) if clip else seq.boxes[sl]
                    out = tracker.update(boxes, seq.scores[sl], seq.embeddings[sl], warp, (seq.height, seq.width))
                    dt = time.perf_counter() - t0
                    total += dt
                    ids = {int(r[4]) for r in out}
                    new_ids = len(ids - seen)
                    seen |= ids
                    for r in out:
                        rows.append([frame - offset, r[4], *r[:4]])
                    sc = seq.scores[sl]
                    stats = getattr(getattr(tracker, "t", None), "last_stats", None) or {}
                    if stats_keys is None:
                        stats_keys = sorted(stats)
                        fw.writerow(["frame", "detections", "det_score_gt_0.1", "det_score_gt_0.6", "outputs",
                                     "new_ids", "total_ids", "seconds", *stats_keys])
                    fw.writerow([frame - offset, len(sc), int((sc > 0.1).sum()), int((sc > 0.6).sum()), len(out),
                                 new_ids, len(seen), f"{dt:.6f}", *[stats.get(k, "") for k in stats_keys]])
            rows = np.array(rows, float).reshape(-1, 6)
            log.event("sequence_finished", sequence=seq.name, raw_boxes=len(rows), ids=len(seen), seconds=total)
            rows = postfilter(rows, a.min_box_area, aspect)
            write_mot(a.output / "trackers" / name / "data" / f"{seq.name}.txt", rows)
            write_mot(a.output / "trackers" / f"{name}-interp" / "data" / f"{seq.name}.txt", interpolate(rows))
            lengths[seq.name] = end - offset
            timing[seq.name] = {"frames": end - offset, "seconds": total,
                                "association_fps": (end - offset) / max(total, 1e-9)}
            if not a.quiet:
                log.message(f"{name} | {seq.name}: {len(rows)} boxes after filter, {len(seen)} IDs, "
                            f"{timing[seq.name]['association_fps']:.1f} association FPS")
        log.save("timing.json", timing)
        if a.gt_root and not a.no_eval:
            res = evaluate(a.gt_root, a.output / "trackers", [name, f"{name}-interp"], lengths, a.benchmark)
            with (a.output / "metrics.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(res[0]))
                w.writeheader()
                w.writerows(res)
            log.save("metrics.json", res)
            for r in res:
                if r["sequence"] == "COMBINED":
                    log.message(f"{r['tracker']}: HOTA={r['HOTA']:.2f} AssA={r['AssA']:.2f} DetA={r['DetA']:.2f} "
                                f"IDF1={r['IDF1']:.2f} MOTA={r['MOTA']:.2f} IDSW={r['IDSW']}")


if __name__ == "__main__":
    main()
