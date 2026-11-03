import csv
import json
import time
from dataclasses import replace
from pathlib import Path
import numpy as np
from .config import SUPPORTED_VARIANTS
from .geometry import crowding
from .tracker import ReliabilityTracker
from .data import sha256


def mot_line(frame, tid, box, confidence=1):
    x, y, x2, y2 = box
    return f"{frame},{tid},{x:.6f},{y:.6f},{x2-x:.6f},{y2-y:.6f},{confidence:.6f},-1,-1,-1\n"


def run_sequence(sequence, config, directory, log, diagnostics="matched"):
    directory = Path(directory)
    sequence_dir = directory / "logs" / sequence.name
    sequence_dir.mkdir(parents=True, exist_ok=True)
    result_path = directory / "data" / f"{sequence.name}.txt"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    config = replace(config, fps=sequence.fps)
    if config.variant == "author-oa":
        from .upstream_tracker import AuthorTracker
        tracker = AuthorTracker(config)
    else:
        tracker = ReliabilityTracker(config)
    camera = None
    camera_hash = None
    if config.camera_motion:
        camera_path = Path(__file__).resolve().parents[1] / "cache" / sequence.metadata["benchmark"] / "gmc" / (sequence.name + ".npz")
        with np.load(camera_path, allow_pickle=False) as cache:
            camera = cache["translations"]
            metadata = json.loads(str(cache["metadata"]))
        if metadata["embedding_cache_sha256"] != sequence.metadata["cache_sha256"]:
            raise ValueError("Camera motion and embedding cache source mismatch")
        if camera.shape != (sequence.length, 2) or not np.isfinite(camera).all():
            raise ValueError("Camera translation cache must be length x 2 and finite")
        camera_hash = sha256(camera_path)
    predicted_ids = np.full(len(sequence.frames), -1, dtype=int)
    total_seconds, pair_count, frozen, updates = 0.0, 0, 0, 0
    origins = {}  # Analysis-only GT mapping; never passed into the tracker.
    log.event("sequence_started", variant=config.variant, sequence=sequence.name,
              frames=sequence.length, detections=len(sequence.frames))
    with result_path.open("w", encoding="utf-8", buffering=1) as result, \
         (sequence_dir / "frames.csv").open("w", newline="", encoding="utf-8", buffering=1) as frame_file, \
         (sequence_dir / "associations.jsonl").open("w", encoding="utf-8", buffering=1) as assoc_file, \
         (sequence_dir / "candidates.jsonl").open("w", encoding="utf-8", buffering=1) as candidate_file:
        frames = csv.DictWriter(frame_file, fieldnames=["frame", "detections", "active_tracks", "matches",
                    "new_tracks", "frozen_updates", "association_seconds", "association_fps"])
        frames.writeheader()
        for frame in range(1, sequence.length + 1):
            sl = sequence.indices(frame)
            boxes = sequence.boxes[sl]
            start = time.perf_counter()
            extra = {} if camera is None else {"camera_translation": camera[frame - 1]}
            ids = tracker.update(frame, boxes, sequence.embeddings[sl], sequence.scores[sl], sequence.labels[sl], diagnostics, **extra)
            elapsed = time.perf_counter() - start
            total_seconds += elapsed
            predicted_ids[sl] = ids
            for tid, box in zip(ids, boxes):
                if tid > 0:
                    result.write(mot_line(frame, tid, box))
            def annotate(row):
                row = dict(row)
                if sequence.gt_ids is not None:
                    truth = int(sequence.gt_ids[sl][row["detection_index"]])
                    row["gt_id"] = truth
                    row["track_origin_gt_id"] = origins.get(row["track_id"])
                    row["identity_correct"] = (truth == origins[row["track_id"]]) if row["track_id"] in origins else None
                return row
            for row in tracker.pair_diagnostics:
                candidate_file.write(json.dumps(annotate(row), allow_nan=False) + "\n")
            for row in tracker.diagnostics:
                assoc_file.write(json.dumps(annotate(row), allow_nan=False) + "\n")
            # Only establish GT origin after logging candidate decisions.
            if sequence.gt_ids is not None:
                for tid, truth in zip(ids, sequence.gt_ids[sl]):
                    if tid > 0:
                        origins.setdefault(int(tid), int(truth))
            freeze_count = tracker.statistics["frozen_updates"]
            frozen += freeze_count
            updates += tracker.statistics["matches"]
            pair_count += len(tracker.pair_diagnostics)
            frames.writerow({"frame": frame, "detections": len(boxes), "active_tracks": len(tracker.tracks),
                "matches": tracker.statistics["matches"], "new_tracks": tracker.statistics["new_tracks"],
                "frozen_updates": freeze_count, "association_seconds": elapsed,
                "association_fps": 1 / max(elapsed, 1e-12)})
            if frame == 1 or frame % 100 == 0 or frame == sequence.length:
                log.message(f"{config.variant} | {sequence.name} | {frame}/{sequence.length} frames | "
                            f"{len(boxes)} boxes | {sequence.length if total_seconds == 0 else frame/total_seconds:.1f} association FPS")
            log.event("frame_completed", variant=config.variant, sequence=sequence.name, frame=frame,
                      detections=len(boxes), active_tracks=len(tracker.tracks), seconds=elapsed)
    summary = {"variant": config.variant, "sequence": sequence.name, "frames": sequence.length,
        "detections": len(sequence.frames), "association_seconds": total_seconds,
        "association_fps": sequence.length / max(total_seconds, 1e-12),
        "fps_scope": "association only; excludes MASA extraction, IO and diagnostics serialization",
        "frozen_updates": frozen, "matched_updates": updates, "candidate_rows": pair_count,
        "config": config.to_dict(), "source": sequence.metadata}
    summary["camera_translation_cache_sha256"] = camera_hash
    (sequence_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.event("sequence_completed", **{k: v for k, v in summary.items() if k not in ("source", "config")})
    return summary, predicted_ids


def subset_masks(sequence):
    iou = np.zeros(len(sequence.frames))
    for frame in range(1, sequence.length + 1):
        sl = sequence.indices(frame)
        iou[sl], _ = crowding(sequence.boxes[sl])
    masks = {"iou-low": iou < 0.3, "iou-medium": (iou >= 0.3) & (iou <= 0.7), "iou-high": iou > 0.7}
    if sequence.visibility is not None:
        visibility = sequence.visibility
        known = np.isfinite(visibility) & (visibility >= 0) & (visibility <= 1)
        masks.update({"visibility-high": known & (visibility > 0.7),
                      "visibility-medium": known & (visibility >= 0.3) & (visibility <= 0.7),
                      "visibility-low": known & (visibility < 0.3)})
    return masks


def write_conditioned_subset(sequence, ids, mask, tracker_root, gt_root):
    """Select GT and its fixed-box predictions after full causal tracking.

    Identity histories keep original frame numbers. These are conditioned
    observation metrics, not MOTChallenge leaderboard scores.
    """
    if sequence.gt_ids is None:
        raise ValueError("Subset evaluation requires analysis-only gt_ids")
    gt_dir = Path(gt_root) / sequence.name / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    pred_file = Path(tracker_root) / "data" / f"{sequence.name}.txt"
    pred_file.parent.mkdir(parents=True, exist_ok=True)
    with (gt_dir / "gt.txt").open("w", encoding="utf-8") as gt, pred_file.open("w", encoding="utf-8") as pred:
        for idx in np.flatnonzero(mask):
            frame, box = int(sequence.frames[idx]), sequence.boxes[idx]
            x, y, x2, y2 = box
            gt.write(f"{frame},{int(sequence.gt_ids[idx])},{x},{y},{x2-x},{y2-y},1,1,1\n")
            if ids[idx] > 0:
                pred.write(mot_line(frame, ids[idx], box))
    (gt_dir.parent / "seqinfo.ini").write_text(
        f"[Sequence]\nname={sequence.name}\nseqLength={sequence.length}\nframeRate={sequence.fps}\n", encoding="utf-8")


def run_suite(sequences, config, variants, log, diagnostics="matched", subsets=True):
    summary = []
    if len({s.name for s in sequences}) != len(sequences):
        raise ValueError("Sequence names must be unique within a run")
    if len({s.metadata.get('benchmark') for s in sequences}) != 1:
        raise ValueError("Run one benchmark at a time")
    for variant in variants:
        if variant not in SUPPORTED_VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        profile = replace(config, variant=variant)
        for seq in sequences:
            directory = log.directory / "trackers" / variant
            stats, ids = run_sequence(seq, profile, directory, log, diagnostics)
            summary.append(stats)
            # Unified fixed-box GT is explicitly a protocol of this project.
            if seq.gt_ids is not None:
                write_conditioned_subset(seq, ids, np.ones(len(ids), dtype=bool), directory,
                                         log.directory / "gt" / "all")
                if subsets:
                    for name, mask in subset_masks(seq).items():
                        write_conditioned_subset(seq, ids, mask,
                            log.directory / "subsets" / name / "trackers" / variant,
                            log.directory / "gt" / name)
        log.save(f"trackers/{variant}/config.json", profile.to_dict())
    log.save("summary.json", summary)
    log.save("manifest.json", {"protocol": "fixed valid GT boxes; paper-described reference baselines",
        "variants": list(variants), "sequences": [{"name": s.name, "length": s.length,
            "source": s.metadata} for s in sequences], "subsets": subsets,
        "diagnostics": diagnostics, "synthetic": any(s.metadata.get("synthetic", False) for s in sequences)})
    return summary
