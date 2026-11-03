"""Statistics for the paper, computed from the per-sequence TrackEval results of the locked runs.

Writes outputs/paper-analysis/analysis.json. Tests are paired over sequences:
  - exact two-sided sign-flip permutation test on the mean per-sequence HOTA difference,
  - Holm correction over the seven baselines of a split,
  - comparison against the better of the tuned and the published-default configuration of each baseline,
  - marginal analysis of the MASA-RF hyper-parameters from the tuning logs (tuning splits only).
"""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "outputs/det-suite"
SPLITS = ["MOT17-val", "MOT20-val", "DanceTrack-val"]
BASELINES = ["bytetrack", "ocsort", "botsort", "botsort-reid", "deepocsort", "masa", "masa-oa-author"]


def load(split, name, root=SUITE):
    """Return {sequence: row} for the raw (not interpolated) output, floats where possible."""
    rows = {}
    for r in csv.DictReader((root / split / name / "metrics.csv").open()):
        if r["tracker"].endswith("-interp"):
            continue
        rows[r["sequence"]] = {k: (float(v) if k not in ("tracker", "sequence") else v) for k, v in r.items()}
    return rows


def sign_flip_p(diff):
    """Exact two-sided p-value of the mean difference under random sign flips (meet in the middle)."""
    d = np.asarray(diff, float)
    d = d[np.abs(d) > 1e-12]
    n = len(d)
    if n == 0:
        return 1.0
    observed = abs(d.sum()) - 1e-9

    def sums(x):
        s = np.zeros(1)
        for v in x:
            s = np.concatenate([s + v, s - v])
        return np.sort(s)

    a, b = sums(d[: n // 2]), sums(d[n // 2:])
    hits = np.searchsorted(b, observed - a, side="left")
    upper = len(b) - hits
    lower = np.searchsorted(b, -observed - a, side="right")
    return float((upper + lower).sum() / (len(a) * len(b)))


def holm(ps):
    order = np.argsort(ps)
    adjusted, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        adjusted[i] = min(running, 1.0)
    return adjusted


def compare(ref, other):
    seqs = [s for s in ref if s != "COMBINED"]
    diff = [ref[s]["HOTA"] - other[s]["HOTA"] for s in seqs]
    return {"combined_delta": ref["COMBINED"]["HOTA"] - other["COMBINED"]["HOTA"],
            "mean_seq_delta": float(np.mean(diff)), "wins": int(sum(x > 0 for x in diff)),
            "losses": int(sum(x < 0 for x in diff)), "p_sign_flip": sign_flip_p(diff), "n": len(seqs)}


def tuning_marginals():
    out = {}
    for split in ("MOT17-train-half", "DanceTrack-tune"):
        trials = []
        for sub in ("masa-rf", "masa-rf-r2"):
            f = ROOT / "outputs/det-tune" / split / sub / "ranking.csv"
            for r in csv.DictReader(f.open()):
                trials.append({**json.loads(r["params"]), "HOTA": float(r["HOTA"])})
        marg = {}
        for key in ("ema_alpha", "appearance_weight", "center_weight", "risk_memory", "adaptive_weight", "appearance_mode"):
            vals = sorted({t[key] for t in trials if key in t}, key=str)
            marg[key] = {str(v): {"n": sum(t.get(key) == v for t in trials),
                                  "best": max(t["HOTA"] for t in trials if t.get(key) == v),
                                  "median": float(np.median([t["HOTA"] for t in trials if t.get(key) == v]))} for v in vals}
        out[split] = {"trials": len(trials), "marginals": marg}
    return out


def main():
    result = {"splits": {}, "tuning": tuning_marginals()}
    for split in SPLITS:
        ref = load(split, "masa-rf")
        block = {"n": len(ref) - 1, "min_possible_p": 2.0 ** (1 - (len(ref) - 1)), "per_sequence": {}, "baselines": {}}
        rows = {b: load(split, b) for b in BASELINES}
        defaults = {b: load(split, f"{b}-default") for b in BASELINES}
        for b in BASELINES:
            block["baselines"][b] = compare(ref, rows[b])
            better = defaults[b] if defaults[b]["COMBINED"]["HOTA"] > rows[b]["COMBINED"]["HOTA"] else rows[b]
            block["baselines"][b]["best_config"] = "default" if better is defaults[b] else "tuned"
            block["baselines"][b]["best_config_HOTA"] = better["COMBINED"]["HOTA"]
            block["baselines"][b]["vs_best_config"] = compare(ref, better)
        adjusted = holm([block["baselines"][b]["p_sign_flip"] for b in BASELINES])
        adjusted_best = holm([block["baselines"][b]["vs_best_config"]["p_sign_flip"] for b in BASELINES])
        for b, p, q in zip(BASELINES, adjusted, adjusted_best):
            block["baselines"][b]["p_holm"] = p
            block["baselines"][b]["vs_best_config"]["p_holm"] = q
        for s in ref:
            if s != "COMBINED":
                block["per_sequence"][s] = {"masa-rf": ref[s]["HOTA"], **{b: rows[b][s]["HOTA"] for b in BASELINES}}
        result["splits"][split] = block
    out = ROOT / "outputs/paper-analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(result, indent=1))
    for split, block in result["splits"].items():
        print(f"== {split} n={block['n']} smallest possible p={block['min_possible_p']:.4f}")
        for b, c in block["baselines"].items():
            v = c["vs_best_config"]
            print(f"  {b:16s} d={c['combined_delta']:+6.2f} mean={c['mean_seq_delta']:+6.2f} W/L={c['wins']}/{c['losses']} "
                  f"p={c['p_sign_flip']:.5f} holm={c['p_holm']:.5f} | best={c['best_config']:7s} d={v['combined_delta']:+6.2f} "
                  f"W/L={v['wins']}/{v['losses']} holm={v['p_holm']:.5f}")
    for split, block in result["tuning"].items():
        print(f"== tuning {split}: {block['trials']} trials")
        for key in ("ema_alpha", "appearance_weight", "risk_memory"):
            print(" ", key, {k: (v["n"], round(v["best"], 2), round(v["median"], 2)) for k, v in block["marginals"][key].items()})


if __name__ == "__main__":
    main()
