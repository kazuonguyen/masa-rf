"""Post-hoc experiments for the paper, run on the reporting splits after the main results were fixed.

They are reported separately from the locked comparison and never change a locked configuration:
  pure-*      remove a cue completely (the locked-config ablations keep GAW/gates, see the paper)
  ema-*       appearance-memory sweep of MASA-RF, ORM off
  freeze-*    hard freeze of the memory (alpha = 0) when occlusion risk >= 0.3
  w-*         appearance-weight sweep of MASA-RF
  oa-*        mechanism ablation of the authors' MASA-OA file with real detections

Each run goes through era_masa.det_run, so it has the same logs, predictions and metrics as the main suite.
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv-gpu/Scripts/python.exe")
sys.path.insert(0, str(ROOT))
from tools.run_det_suite import SPLITS

REPORTING = ["MOT17-val", "MOT20-val", "DanceTrack-val"]
EMA = [0.0, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0]
WEIGHTS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
OA_VARIANTS = ["original", "kf-once", "freeze", "single", "thr03", "single-thr03"]
OA_SOURCES = "outputs/mot15-author-mechanism-ablation/sources"


def plan(split):
    locked = {e["name"]: e for e in json.loads((ROOT / "configs/det_locked.json").read_text())[SPLITS[split][3]]}
    rf = locked["masa-rf"]["params"]
    oa = locked["masa-oa-author"]["params"]
    runs = [("masa-rf", "pure-geometry", {**rf, "appearance_weight": 0.0, "adaptive_weight": 0.0, "appearance_gate": 0.0}),
            ("masa-rf", "pure-appearance", {**rf, "appearance_weight": 1.0, "adaptive_weight": 0.0, "geometry_gate": 0.0,
                                             "center_weight": 0.0})]
    no_orm = {**rf, "risk_memory": 0.0}
    runs += [("masa-rf", f"ema-{x:g}", {**no_orm, "ema_alpha": x}) for x in EMA]
    runs += [("masa-rf", f"freeze-ema-{x:g}", {**no_orm, "ema_alpha": x, "freeze_risk": 0.3}) for x in (0.1, 0.8)]
    runs += [("masa-rf", f"w-{x:g}", {**rf, "appearance_weight": x}) for x in WEIGHTS]
    runs += [("masa-oa-author", f"oa-{v}", {**oa, "source_root": f"{OA_SOURCES}/{v}"}) for v in OA_VARIANTS]
    return runs


def execute(job):
    split, method, name, params, out = job
    target = out / split / name
    if (target / "metrics.json").exists():
        return name, "cached"
    caches, gt, bench, _ = SPLITS[split]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    cmd = [PY, "-m", "era_masa.det_run", "--method", method, "--params", json.dumps(params), "--name", name,
           "--caches", caches, "--gt-root", gt, "--benchmark", bench, "--output", str(target), "--quiet"]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    if p.returncode:
        print(f"FAILED {split}/{name}: {p.stderr[-600:]}", flush=True)
        return name, "failed"
    print(f"done {split}/{name}", flush=True)
    return name, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=ROOT / "outputs/paper-posthoc")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", help="substring filter on run names")
    a = ap.parse_args()
    jobs = [(s, m, n, p, a.output) for s in REPORTING for m, n, p in plan(s) if not a.only or a.only in n]
    print(f"{len(jobs)} runs", flush=True)
    with ThreadPoolExecutor(a.workers) as pool:
        results = list(pool.map(execute, jobs))
    failed = [n for n, s in results if s == "failed"]
    print("failed:", failed, flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
