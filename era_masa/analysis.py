import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata


def auc(labels, scores):
    labels, scores = np.asarray(labels, dtype=bool), np.asarray(scores, dtype=float)
    positives, negatives = labels.sum(), (~labels).sum()
    if positives == 0 or negatives == 0:
        return None
    return float((rankdata(scores)[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def analyze_run(directory, log):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for variant in manifest["variants"]:
        for sequence in manifest["sequences"]:
            source = directory / "trackers" / variant / "logs" / sequence["name"]
            # All candidates are less selection-biased; fall back to accepted pairs.
            path = source / "candidates.jsonl"
            sample = "all_candidates"
            if not path.exists() or path.stat().st_size == 0:
                path = source / "associations.jsonl"
                sample = "accepted_pairs_only"
            labels, signals = [], {k: [] for k in ("risk", "overlap_iou", "temporal", "rival", "geometry", "motion")}
            if not path.is_file():
                continue
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    pair = json.loads(line)
                    if pair.get("identity_correct") is None or pair.get("risk") is None or pair.get("valid") is False:
                        continue
                    labels.append(not pair["identity_correct"])
                    for k in signals:
                        signals[k].append(pair[k])
            rows.append({"variant": variant, "sequence": sequence["name"], "sample": sample,
                "pairs": len(labels), "wrong_origin_identity_pairs": sum(labels),
                **{f"{k}_auc": auc(labels, v) for k, v in signals.items()}})
    log.save("reliability_analysis.json", {"target": "pair identity differs from track origin GT identity",
        "limitation": "Association-risk proxy; does not directly label MASA embedding contamination. "
                      "Accepted-only samples are selection-biased; origin IDs can become stale after switches.", "rows": rows})
    if rows:
        with (directory / "reliability_analysis.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    log.message(f"Reliability diagnostic report: {len(rows)} sequence/variant rows")
    return rows
