"""Run locked configurations of all methods on a reporting split; summarise with paired bootstrap.

Locked configs come from configs/det_locked.json (written from tuning results
before this script is run on any reporting split).
"""
import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv-gpu/Scripts/python.exe")
sys.path.insert(0, str(ROOT))
SPLITS = {
    "MOT17-val": ("cache_det_v2/MOT17-train-ablation", "gt/MOT17-val-half", "MOT17", "mot"),
    "MOT20-val": ("cache_det_v2/MOT20-train-mot17det", "gt/MOT20-train", "MOT20", "mot"),
    "DanceTrack-val": ("cache_det_v2/DanceTrack-val", "gt/DanceTrack-val", "DanceTrack", "dancetrack"),
    "MOT17-train-half": ("cache_det_v2/MOT17-train-ablation", "gt/MOT17-train-half", "MOT17", "mot"),
    "DanceTrack-tune": ("cache_det_v2/DanceTrack-train", "gt/DanceTrack-train-tune", "DanceTrack", "dancetrack"),
}


def run(entry, split, out):
    caches, gt, bench, _ = SPLITS[split]
    target = out / entry["name"]
    if not (target / "metrics.json").exists():
        cmd = [PY, "-m", "era_masa.det_run", "--method", entry["method"], "--params", json.dumps(entry["params"]),
               "--name", entry["name"], "--caches", caches, "--gt-root", gt, "--benchmark", bench,
               "--output", str(target)]
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if p.returncode:
            raise RuntimeError(f"{entry['name']} failed: {p.stderr[-800:]}")
    return json.loads((target / "metrics.json").read_text())


def per_seq_hota_objects(target, name, gt_root, bench):
    """Recompute per-sequence TrackEval HOTA result objects for bootstrap (combine_sequences)."""
    from era_masa.evaluation import load_trackeval
    metrics, Dataset, _ = load_trackeval(ROOT / "third_party/TrackEval")
    hota = metrics[0]
    info = json.loads((ROOT / gt_root / "seqinfo.json").read_text())["sequences"]
    ds = Dataset({"GT_FOLDER": str(ROOT / gt_root), "TRACKERS_FOLDER": str(target / "trackers"),
                  "TRACKERS_TO_EVAL": [name], "SEQ_INFO": {k: v["length"] for k, v in info.items()},
                  "SKIP_SPLIT_FOL": True, "DO_PREPROC": True, "PRINT_CONFIG": False,
                  "BENCHMARK": "MOT20" if bench == "MOT20" else "MOT17"})
    return {s: hota.eval_sequence(ds.get_preprocessed_seq_data(ds.get_raw_seq_data(name, s), "pedestrian"))
            for s in info}, hota


def bootstrap(a_res, b_res, hota, samples=2000, seed=20261006):
    rng = np.random.default_rng(seed)
    names = list(a_res)
    deltas = []
    for _ in range(samples):
        pick = rng.choice(names, len(names), replace=True)
        va = hota.combine_sequences({str(i): a_res[n] for i, n in enumerate(pick)})
        vb = hota.combine_sequences({str(i): b_res[n] for i, n in enumerate(pick)})
        deltas.append(float(np.mean(va["HOTA"]) - np.mean(vb["HOTA"])) * 100)
    return [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=sorted(SPLITS))
    p.add_argument("--locked", type=Path, default=ROOT / "configs/det_locked.json")
    p.add_argument("--output", type=Path)
    p.add_argument("--only", nargs="+")
    p.add_argument("--reference", default="masa-rf")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    out = a.output or ROOT / "outputs" / "det-suite" / a.split
    out.mkdir(parents=True, exist_ok=True)
    family = SPLITS[a.split][3]
    locked = json.loads(a.locked.read_text())[family]
    entries = [e for e in locked if not a.only or e["name"] in a.only]
    with ThreadPoolExecutor(a.workers) as pool:
        results = dict(zip([e["name"] for e in entries], pool.map(lambda e: run(e, a.split, out), entries)))
    rows = []
    for name, res in results.items():
        for interp in ("", "-interp"):
            r = next(x for x in res if x["sequence"] == "COMBINED" and x["tracker"] == name + interp)
            rows.append({"method": name + interp, **{k: round(r[k], 3) if isinstance(r[k], float) else r[k]
                         for k in ("HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDSW", "FP", "FN")}})
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # paired per-sequence bootstrap of the reference vs every other method (raw outputs)
    _, gt, bench, _ = SPLITS[a.split]
    boots = {}
    if a.reference in results:
        ref, hota = per_seq_hota_objects(out / a.reference, a.reference, gt, bench)
        for name in results:
            if name == a.reference:
                continue
            other, _ = per_seq_hota_objects(out / name, name, gt, bench)
            boots[name] = bootstrap(ref, other, hota)
        (out / f"bootstrap-{a.reference}.json").write_text(json.dumps({"reference": a.reference, "HOTA_delta_CI95": boots,
            "unit": "sequence", "samples": 2000}, indent=1))
    for r in sorted(rows, key=lambda r: -r["HOTA"]):
        ci = boots.get(r["method"])
        print(f"{r['method']:28s} HOTA {r['HOTA']:6.2f} AssA {r['AssA']:6.2f} DetA {r['DetA']:6.2f} "
              f"IDF1 {r['IDF1']:6.2f} MOTA {r['MOTA']:6.2f} IDSW {r['IDSW']:5d}" + (f"  ref-this CI95 {ci}" if ci else ""))


if __name__ == "__main__":
    main()
