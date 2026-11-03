"""Build tables for docs/DETECTOR_EXPERIMENTS.md from det-suite outputs (no new tracking)."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ["MOT17-val", "MOT20-val", "DanceTrack-val"]
MAIN = ["bytetrack", "ocsort", "botsort", "botsort-reid", "deepocsort", "masa", "masa-oa-author",
        "botsort-reid-rf", "deepocsort-rf", "masa-rf"]
LABEL = {"bytetrack": "ByteTrack", "ocsort": "OC-SORT", "botsort": "BoT-SORT", "botsort-reid": "BoT-SORT-ReID (MASA emb.)",
         "deepocsort": "Deep OC-SORT (MASA emb.)", "masa": "MASA (greedy/Hungarian assoc.)",
         "masa-oa-author": "MASA-OA (authors' code)", "botsort-reid-rf": "BoT-SORT-ReID + ORM",
         "deepocsort-rf": "Deep OC-SORT + ORM/GAW (tuned)", "masa-rf": "**MASA-RF (ours)**"}
COLS = ("HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDSW")


def load(split):
    f = ROOT / "outputs/det-suite" / split / "summary.csv"
    return {r["method"]: r for r in csv.DictReader(f.open())} if f.exists() else {}


def table(rows, names, label=lambda n: n):
    out = ["| Method | " + " | ".join(COLS) + " |", "|---" * (len(COLS) + 1) + "|"]
    for n in names:
        if n in rows:
            r = rows[n]
            out.append(f"| {label(n)} | " + " | ".join(
                (f"{float(r[c]):.2f}" if c != "IDSW" else r[c]) for c in COLS) + " |")
    return "\n".join(out)


def fps(split, name):
    f = ROOT / "outputs/det-suite" / split / name / "timing.json"
    if not f.exists():
        return None
    t = json.loads(f.read_text())
    return sum(v["frames"] for v in t.values()) / sum(v["seconds"] for v in t.values())


def tuning_vs_val():
    prov = json.loads((ROOT / "configs/det_locked_provenance.json").read_text())
    lines = []
    for fam, split in (("mot", "MOT17-val"), ("dancetrack", "DanceTrack-val")):
        rows = load(split)
        pairs = [(prov[f"{fam}/{m}"]["tuning_HOTA"], float(rows[m]["HOTA"])) for m in MAIN if m in rows]
        a, b = np.array(pairs).T
        rank = lambda x: np.argsort(np.argsort(-x))
        rho = np.corrcoef(rank(a), rank(b))[0, 1]
        lines.append(f"- {split}: Spearman rank correlation between tuning HOTA and reporting HOTA over "
                     f"{len(pairs)} locked methods = {rho:.2f}")
    return "\n".join(lines)


def main():
    parts = []
    for split in SPLITS:
        rows = load(split)
        if not rows:
            continue
        parts.append(f"### {split}\n\nRaw output (no interpolation):\n\n" + table(rows, MAIN, lambda n: LABEL[n]))
        parts.append("\nWith linear interpolation (gap ≤ 20):\n\n" +
                     table(rows, [m + "-interp" for m in MAIN], lambda n: LABEL[n[:-7]]))
        parts.append("\nPublished-default hyper-parameters (untuned):\n\n" +
                     table(rows, [m + "-default" for m in MAIN], lambda n: LABEL[n[:-8]]))
        abl = [n for n in rows if n.startswith("masa-rf-no-") and not n.endswith("interp")]
        parts.append("\nMASA-RF component ablation (locked config with one component removed):\n\n" +
                     table(rows, ["masa-rf"] + abl))
        boot = ROOT / "outputs/det-suite" / split / "bootstrap-masa-rf.json"
        if boot.exists():
            b = json.loads(boot.read_text())["HOTA_delta_CI95"]
            parts.append("\nPaired sequence bootstrap, HOTA(MASA-RF) − HOTA(method), 95% CI:\n\n" + "\n".join(
                f"- {LABEL.get(k, k)}: [{v[0]:+.2f}, {v[1]:+.2f}]" for k, v in b.items() if k in MAIN))
        parts.append("\nAssociation-only throughput (frames/s, CPU, cached detections+embeddings):\n\n" + ", ".join(
            f"{LABEL[m].strip('*')}: {fps(split, m):.0f}" for m in MAIN if fps(split, m)) + "\n")
    parts.append("### Tuning-split vs reporting-split agreement\n\n" + tuning_vs_val())
    (ROOT / "outputs/det-suite/tables.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {ROOT / 'outputs/det-suite/tables.md'}")


if __name__ == "__main__":
    main()
