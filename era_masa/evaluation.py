"""Use official TrackEval metric implementations, with NumPy alias compatibility.

Only np.float/np.int aliases are substituted in memory; upstream files and the
HOTA/CLEAR/Identity equations remain unchanged. Load only required modules so
unrelated segmentation datasets do not require optional dependencies.
"""
import csv
import importlib.util
import json
import re
import sys
import types
from pathlib import Path
import numpy as np
from .data import sha256


def load_trackeval(root):
    root = Path(root).resolve() / "trackeval"
    if not (root / "metrics" / "hota.py").is_file():
        raise FileNotFoundError("TrackEval source missing. Run tools/bootstrap.py first.")
    prefix = "_era_trackeval"
    for name, path in ((prefix, root), (prefix + ".metrics", root / "metrics"),
                       (prefix + ".datasets", root / "datasets")):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    files = ["utils", "_timing", "metrics._base_metric", "metrics.hota", "metrics.clear", "metrics.identity",
             "datasets._base_dataset", "datasets.mot_challenge_2d_box"]
    provenance = {}
    for relative in files:
        path = root.joinpath(*relative.split(".")).with_suffix(".py")
        name = prefix + "." + relative
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        source = path.read_text(encoding="utf-8")
        source = re.sub(r"\bnp\.float\b", "np.float64", source)
        source = re.sub(r"\bnp\.int\b", "np.int64", source)
        exec(compile(source, str(path), "exec"), module.__dict__)
        parent_name, child = name.rsplit(".", 1)
        setattr(sys.modules[parent_name], child, module)
        provenance[relative] = sha256(path)
    metrics = [sys.modules[prefix + ".metrics.hota"].HOTA(),
               sys.modules[prefix + ".metrics.clear"].CLEAR({"PRINT_CONFIG": False}),
               sys.modules[prefix + ".metrics.identity"].Identity({"PRINT_CONFIG": False})]
    dataset = sys.modules[prefix + ".datasets.mot_challenge_2d_box"].MotChallenge2DBox
    return metrics, dataset, provenance


def flatten_metrics(results):
    hota, clear, identity = results["HOTA"], results["CLEAR"], results["Identity"]
    return {**{k: float(np.mean(hota[k]) * 100) for k in ("HOTA", "AssA", "AssRe", "AssPr", "DetA", "DetRe", "DetPr", "LocA")},
            "IDSW": int(clear["IDSW"]), "IDF1": float(identity["IDF1"] * 100),
            "IDP": float(identity["IDP"] * 100), "IDR": float(identity["IDR"] * 100),
            **{k: int(identity[k]) for k in ("IDTP", "IDFP", "IDFN")},
            "MOTA": float(clear["MOTA"] * 100), "Frag": int(clear["Frag"]),
            "FP": int(clear["CLR_FP"]), "FN": int(clear["CLR_FN"]),
            **{k: int(clear[k]) for k in ("MT", "PT", "ML")},
            **{k: float(clear[k]) * 100 for k in ("MTR", "MLR", "CLR_Re", "CLR_Pr")},
            "IDSW_per_1000_GT": float(clear["IDSW"] * 1000 / max(clear["CLR_TP"] + clear["CLR_FN"], 1)),
            "GT_Dets": int(clear["CLR_TP"] + clear["CLR_FN"])}


def evaluate_run(directory, trackeval_root, log, include_subsets=True):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    metrics, Dataset, provenance = load_trackeval(trackeval_root)
    log.save("trackeval_provenance.json", {"source": str(Path(trackeval_root).resolve()),
        "file_sha256": provenance, "compatibility": "np.float -> np.float64; np.int -> np.int64 (in memory)"})
    seq_info = {s["name"]: s["length"] for s in manifest["sequences"]}
    conditions = ["all"]
    if include_subsets and manifest["subsets"]:
        conditions += sorted(p.name for p in (directory / "subsets").iterdir())
    rows = []
    for condition in conditions:
        gt_root = directory / "gt" / condition
        if not gt_root.is_dir():
            raise ValueError("Evaluation requires cached gt_ids; tracking without GT remains supported")
        tracker_root = directory / "trackers" if condition == "all" else directory / "subsets" / condition / "trackers"
        dataset = Dataset({"GT_FOLDER": str(gt_root), "TRACKERS_FOLDER": str(tracker_root),
            "TRACKERS_TO_EVAL": manifest["variants"], "SEQ_INFO": dict(seq_info),
            "SKIP_SPLIT_FOL": True, "DO_PREPROC": False, "PRINT_CONFIG": False,
            "BENCHMARK": "MOT17"})
        for variant in manifest["variants"]:
            per_metric = {m.get_name(): {} for m in metrics}
            for seq in seq_info:
                raw = dataset.get_raw_seq_data(variant, seq)
                data = dataset.get_preprocessed_seq_data(raw, "pedestrian")
                results = {}
                for metric in metrics:
                    result = metric.eval_sequence(data)
                    per_metric[metric.get_name()][seq] = result
                    results[metric.get_name()] = result
                row = {"condition": condition, "variant": variant, "sequence": seq, **flatten_metrics(results)}
                rows.append(row)
                log.event("evaluation_sequence_completed", **row)
            combined = {m.get_name(): m.combine_sequences(per_metric[m.get_name()]) for m in metrics}
            row = {"condition": condition, "variant": variant, "sequence": "COMBINED", **flatten_metrics(combined)}
            rows.append(row)
            log.message(f"{condition} | {variant} | HOTA={row['HOTA']:.2f} AssA={row['AssA']:.2f} "
                        f"IDSW={row['IDSW']} IDF1={row['IDF1']:.2f}")
    target = directory / "metrics.csv"
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    log.save("metrics.json", rows)
    return rows
