"""Evaluate author-supplied MOT15 predictions without regenerating embeddings."""
import argparse
import configparser
import csv
import json
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from era_masa.data import sha256
from era_masa.evaluation import load_trackeval, flatten_metrics
from era_masa.runlog import RunLog, run_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    destination = args.output or run_directory(name="published-mot15")
    variants = {"published-masa": "masa", "published-hung": "masa-hung",
        "published-hung-kf": "masa-hung-kf", "published-masa-oa": "masa-hung-kf-OcclusionAware"}
    with RunLog(destination, {"source": "author supplied MOT15 prediction files", "variants": variants}) as log:
        sources = ROOT / "third_party/MASA-OA"
        benchmark = sources / "benchmarks/MOT15"
        seq_info = {}
        checksums = {}
        for seq in sorted(benchmark.iterdir()):
            if not (seq / "seqinfo.ini").exists():
                continue
            cfg = configparser.ConfigParser()
            cfg.read(seq / "seqinfo.ini")
            seq_info[seq.name] = int(cfg["Sequence"]["seqLength"])
            checksums[f"gt/{seq.name}"] = sha256(seq / "gt/gt.txt")
            for variant, suffix in variants.items():
                source = sources / "evaluation/MOT15" / seq.name / f"{seq.name}_{suffix}.txt"
                target = destination / "trackers" / variant / "data" / f"{seq.name}.txt"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                checksums[f"{variant}/{seq.name}"] = sha256(source)
        metrics, Dataset, provenance = load_trackeval(ROOT / "third_party/TrackEval")
        dataset = Dataset({"GT_FOLDER": str(benchmark), "TRACKERS_FOLDER": str(destination / "trackers"),
            "SEQ_INFO": seq_info, "TRACKERS_TO_EVAL": list(variants), "SKIP_SPLIT_FOL": True,
            "BENCHMARK": "MOT15", "DO_PREPROC": True, "PRINT_CONFIG": False})
        rows = []
        for variant in variants:
            by_metric = {m.get_name(): {} for m in metrics}
            for seq in seq_info:
                data = dataset.get_preprocessed_seq_data(dataset.get_raw_seq_data(variant, seq), "pedestrian")
                results = {}
                for metric in metrics:
                    result = metric.eval_sequence(data)
                    by_metric[metric.get_name()][seq] = result
                    results[metric.get_name()] = result
                row = {"variant": variant, "sequence": seq, **flatten_metrics(results)}
                rows.append(row)
                log.message(f"{variant} | {seq} | HOTA={row['HOTA']:.3f} AssA={row['AssA']:.3f} IDSW={row['IDSW']}")
                log.event("published_sequence_evaluated", **row)
            combined = {m.get_name(): m.combine_sequences(by_metric[m.get_name()]) for m in metrics}
            row = {"variant": variant, "sequence": "COMBINED", **flatten_metrics(combined)}
            rows.append(row)
            log.message(f"{variant} COMBINED | HOTA={row['HOTA']:.3f} AssA={row['AssA']:.3f} IDSW={row['IDSW']}")
        with (destination / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        log.save("metrics.json", rows)
        log.save("provenance.json", {"file_sha256": checksums, "trackeval_sha256": provenance,
            "note": "Author supplied predictions; source code run and extraction environment not reproduced yet."})


if __name__ == "__main__":
    main()
