"""Clean run-time measurement for the paper: one process at a time on an otherwise idle machine.

Association: every locked method, single-threaded numerics, repeated; outputs are compared byte-for-byte with
the predictions of the main suite, which also checks that the results reproduce.
Front end: YOLOX-X, MASA-R50 and camera motion with one worker on a sample of each split.
"""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv-gpu/Scripts/python.exe")
sys.path.insert(0, str(ROOT))
from tools.run_det_suite import SPLITS

METHODS = ["bytetrack", "ocsort", "botsort", "botsort-reid", "deepocsort", "masa", "masa-oa-author", "masa-rf"]
REPORTING = ["MOT17-val", "MOT20-val", "DanceTrack-val"]
FRONT = {  # split: (benchmark, split root, split name, detector, sequences)
    "MOT17-val": ("MOT17", "data/MOT17-full/MOT17/train", "train", "mot17-ablation", ["MOT17-04-FRCNN", "MOT17-09-FRCNN"]),
    "MOT20-val": ("MOT20", "data/MOT20-full/MOT20/train", "train", "mot17", ["MOT20-01", "MOT20-05"]),
    "DanceTrack-val": ("DanceTrack", "data/DanceTrack/val", "val", "dancetrack", ["dancetrack0004", "dancetrack0073"]),
}


def digest(folder):
    h = hashlib.sha256()
    for f in sorted(folder.glob("*.txt")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def association(out, repeats):
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    table = {}
    for split in REPORTING:
        caches, gt, bench, family = SPLITS[split]
        locked = {e["name"]: e for e in json.loads((ROOT / "configs/det_locked.json").read_text())[family]}
        for method in METHODS:
            fps = []
            for rep in range(repeats):
                target = out / "association" / split / method / f"rep{rep}"
                if not (target / "timing.json").exists():
                    cmd = [PY, "-m", "era_masa.det_run", "--method", method, "--params", json.dumps(locked[method]["params"]),
                           "--name", method, "--caches", caches, "--gt-root", gt, "--benchmark", bench,
                           "--output", str(target), "--quiet", "--no-eval"]
                    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, env=env)
                t = json.loads((target / "timing.json").read_text())
                fps.append(sum(v["frames"] for v in t.values()) / sum(v["seconds"] for v in t.values()))
            same = digest(out / "association" / split / method / "rep0" / "trackers" / method / "data") == digest(
                ROOT / "outputs/det-suite" / split / method / "trackers" / method / "data")
            table.setdefault(split, {})[method] = {"fps_median": statistics.median(fps), "fps": fps,
                                                   "ms_per_frame": 1000 / statistics.median(fps), "identical_to_suite": same}
            print(split, method, f"{statistics.median(fps):.1f} FPS", "identical" if same else "DIFFERENT", flush=True)
            (out / "association.json").write_text(json.dumps(table, indent=1))


def front_end(out, max_frames):
    summary = {}
    for split, (bench, root, name, det, seqs) in FRONT.items():
        target = out / "frontend" / split
        cmd = [PY, "tools/extract_detections.py", "--benchmark", bench, "--split-root", root, "--split", name,
               "--detector", det, "--cache-root", str(out / "frontend-cache" / split), "--sequences", *seqs,
               "--max-frames", str(max_frames), "--output", str(target)]
        if not target.exists():
            subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)
        import csv
        d, m, g = [], [], []
        for f in sorted(target.glob("*-frames.csv")):
            rows = list(csv.DictReader(f.open()))[10:]  # skip model warm-up frames
            d += [float(r["detector_s"]) for r in rows]
            m += [float(r["masa_s"]) for r in rows]
            g += [float(r["gmc_s"]) for r in rows]
        summary[split] = {"frames": len(d), "detector_ms": 1000 * statistics.mean(d), "masa_ms": 1000 * statistics.mean(m),
                          "cmc_ms": 1000 * statistics.mean(g)}
        print(split, summary[split], flush=True)
        (out / "frontend.json").write_text(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=ROOT / "outputs/paper-timing")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--skip-association", action="store_true")
    ap.add_argument("--skip-frontend", action="store_true")
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if not a.skip_frontend:
        front_end(a.output, a.max_frames)
    if not a.skip_association:
        association(a.output, a.repeats)


if __name__ == "__main__":
    main()
