"""Lock one configuration per method and benchmark family from TUNING results only.

family "mot"        <- outputs/det-tune/MOT17-train-half  (used for MOT17-val and MOT20-val)
family "dancetrack" <- outputs/det-tune/DanceTrack-tune   (used for DanceTrack-val)
Selection: highest tuning HOTA over all tuning rounds of that method. Published-default
entries (params {}) and component ablations of the proposed tracker are added.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {"mot": "MOT17-train-half", "dancetrack": "DanceTrack-tune"}
ROUNDS = {
    "bytetrack": ["bytetrack"], "ocsort": ["ocsort"], "botsort": ["botsort"],
    "botsort-reid": ["botsort-reid", "botsort-reid-ext", "botsort-reid-r2"],
    "deepocsort": ["deepocsort", "deepocsort-ext", "deepocsort-r2"],
    "masa": ["masa"], "masa-oa-author": ["masa-oa-author"], "masa-rf": ["masa-rf", "masa-rf-r2"],
    "botsort-reid-rf": ["botsort-reid-rf"], "deepocsort-rf": ["deepocsort-rf", "deepocsort-rf-r3"],
}
PROPOSED = "deepocsort-rf"
MODULE_OFF = {"orm": {"orm_kappa": 0.0}, "gaw": {"gaw_kappa": 0.0}, "bisoftmax": {"bisoftmax_beta": 0.0},
              "recovery": {"recovery_cos": 0.0}}
RF_OFF = {"cmc": {"cmc": False}, "byte": {"byte_stage": False}, "risk-memory": {"risk_memory": 0.0},
          "center": {"center_weight": 0.0}, "age-penalty": {"age_penalty": 0.0},
          "appearance": {"appearance_weight": 0.0}, "geometry": {"appearance_weight": 1.0},
          "adaptive-weight": {"adaptive_weight": 0.0}, "appearance-gate": {"appearance_gate": 0.0}}
import sys
sys.path.insert(0, str(ROOT))
from era_masa.det_tracker import RFConfig
RF_DEFAULTS = RFConfig().to_dict()  # values used when a key is absent from the tuned params


def best(split, method):
    cands = []
    for d in ROUNDS[method]:
        f = ROOT / "outputs/det-tune" / split / d / "best.json"
        if f.exists():
            r = json.loads(f.read_text())
            p = r["params"] if isinstance(r["params"], dict) else json.loads(r["params"])
            cands.append((r["HOTA"], d, p))
    if not cands:
        raise FileNotFoundError(f"No tuning result for {method} on {split}")
    return max(cands, key=lambda c: c[0])


def main():
    locked, provenance = {}, {}
    for fam, split in FAMILIES.items():
        entries = []
        for m in ROUNDS:
            hota, src, params = best(split, m)
            entries.append({"name": m, "method": m, "params": params})
            provenance[f"{fam}/{m}"] = {"tuning_split": split, "round": src, "tuning_HOTA": hota}
        for m in ("bytetrack", "ocsort", "botsort", "botsort-reid", "deepocsort", "masa", "masa-oa-author"):
            entries.append({"name": f"{m}-default", "method": m, "params": {}})
        prop = next(e for e in entries if e["name"] == PROPOSED)["params"]
        for k, off in MODULE_OFF.items():
            if any(prop.get(x, 0) for x in off):  # ablate only modules the tuning actually switched on
                entries.append({"name": f"{PROPOSED}-no-{k}", "method": PROPOSED, "params": {**prop, **off}})
        rf = next(e for e in entries if e["name"] == "masa-rf")["params"]
        for k, off in RF_OFF.items():
            # skip no-op ablations: the component is already off in the locked configuration
            if all(rf.get(x, RF_DEFAULTS[x]) == v for x, v in off.items()):
                continue
            entries.append({"name": f"masa-rf-no-{k}", "method": "masa-rf", "params": {**rf, **off}})
        locked[fam] = entries
    (ROOT / "configs/det_locked.json").write_text(json.dumps(locked, indent=1))
    (ROOT / "configs/det_locked_provenance.json").write_text(json.dumps(provenance, indent=1))
    for k, v in provenance.items():
        print(f"{k:28s} {v['round']:20s} tuning HOTA {v['tuning_HOTA']:.2f}")


if __name__ == "__main__":
    main()
