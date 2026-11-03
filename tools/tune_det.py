"""Grid/random search of tracker hyper-parameters on a TUNING split only.

Each trial is a separate process (official baselines mutate sys.path). Results
are appended to <output>/trials.csv as they finish; nothing is overwritten.
"""
import argparse
import csv
import itertools
import json
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv-gpu/Scripts/python.exe")


def trials_from(spec, limit, seed):
    keys = list(spec)
    grid = [dict(zip(keys, v)) for v in itertools.product(*(spec[k] for k in keys))]
    if limit and len(grid) > limit:
        random.Random(seed).shuffle(grid)
        grid = grid[:limit]
    return grid


def run_trial(idx, params, a):
    out = a.output / f"trial{idx:03d}"
    if not (out / "metrics.json").exists():
        cmd = [PY, "-m", "era_masa.det_run", "--method", a.method, "--params", json.dumps(params),
               "--caches", str(a.caches), "--gt-root", str(a.gt_root), "--benchmark", a.benchmark,
               "--output", str(out), "--name", a.method]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if proc.returncode:
            return {"trial": idx, "params": json.dumps(params), "error": proc.stderr[-400:]}
    rows = json.loads((out / "metrics.json").read_text())
    row = next(r for r in rows if r["sequence"] == "COMBINED" and r["tracker"] == a.method)
    return {"trial": idx, "params": json.dumps(params), **{k: row[k] for k in
            ("HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDSW")}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--space", required=True, help="JSON dict param -> list of values")
    p.add_argument("--base", default="{}", help="fixed params merged into every trial")
    p.add_argument("--caches", type=Path, required=True)
    p.add_argument("--gt-root", type=Path, required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--seed", type=int, default=20261006)
    a = p.parse_args()
    space, base = json.loads(a.space), json.loads(a.base)
    trials = [{**base, **t} for t in trials_from(space, a.limit, a.seed)]
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "plan.json").write_text(json.dumps({"method": a.method, "space": space, "base": base,
        "caches": str(a.caches), "gt_root": str(a.gt_root), "trials": trials}, indent=1))
    results = []
    with ThreadPoolExecutor(a.workers) as pool:
        for r in pool.map(lambda x: run_trial(x[0], x[1], a), enumerate(trials)):
            results.append(r)
            print(json.dumps(r)[:300], flush=True)
    ok = [r for r in results if "error" not in r]
    ok.sort(key=lambda r: (-r["HOTA"], -r["IDF1"]))
    with (a.output / "ranking.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trial", "params", "HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDSW"])
        w.writeheader()
        w.writerows(ok)
    errors = [r for r in results if "error" in r]
    (a.output / "errors.json").write_text(json.dumps(errors, indent=1))
    if ok:
        (a.output / "best.json").write_text(json.dumps({**ok[0], "params": json.loads(ok[0]["params"])}, indent=1))
        print("BEST", json.dumps(ok[0]))
    print(f"{len(ok)} ok, {len(errors)} errors")


if __name__ == "__main__":
    main()
