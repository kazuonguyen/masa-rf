import csv
import json
from pathlib import Path
import numpy as np
import pytest
from era_masa.cli import main
from era_masa.data import load_cache, read_ground_truth, save_cache
from era_masa.runlog import RunLog

ROOT = Path(__file__).resolve().parents[1]


def test_failed_run_retains_traceback_and_structured_event(tmp_path):
    target = tmp_path / "failure"
    with pytest.raises(RuntimeError):
        with RunLog(target, {"seed": 7}) as log:
            log.message("Progress before failure")
            assert "Progress before failure" in (target / "console.log").read_text(encoding="utf-8")
            raise RuntimeError("controlled diagnostic failure")
    assert "RuntimeError: controlled diagnostic failure" in (target / "console.log").read_text(encoding="utf-8")
    status = json.loads((target / "status.json").read_text())
    assert status["status"] == "failed"
    events = [json.loads(s) for s in (target / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "run_failed"
    assert events[0]["time"].endswith("+07:00")


def test_existing_run_is_never_overwritten(tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    with pytest.raises(FileExistsError):
        with RunLog(target, {}):
            pass


def test_gt_id_changes_cannot_change_tracking(tmp_path):
    from era_masa.synthetic import create_synthetic
    from era_masa import TrackerConfig
    from era_masa.experiment import run_sequence
    source = create_synthetic(tmp_path / "cache")[0]
    sequence = load_cache(source)
    with RunLog(tmp_path / "logs", {}) as log:
        _, first = run_sequence(sequence, TrackerConfig(), tmp_path / "a", log)
        sequence.gt_ids = np.flip(sequence.gt_ids) + 500
        _, second = run_sequence(sequence, TrackerConfig(), tmp_path / "b", log)
    np.testing.assert_array_equal(first, second)


def test_end_to_end_demo_tracks_evaluates_and_logs(tmp_path):
    trackeval = ROOT / "third_party/TrackEval"
    if not trackeval.exists():
        pytest.skip("Run tools/bootstrap.py to test official TrackEval integration")
    output = tmp_path / "demo"
    main(["demo", "--output", str(output), "--variants", "masa-oa", "era-full", "--evaluate",
          "--diagnostics", "all", "--trackeval-root", str(trackeval)])
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["synthetic"] is True
    assert json.loads((output / "status.json").read_text())["status"] == "completed"
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert sum(e["event"] == "frame_completed" for e in events) == 2 * 3 * 80
    for variant in ("masa-oa", "era-full"):
        logs = output / "trackers" / variant / "logs" / "clean"
        with (logs / "frames.csv").open() as stream:
            frames = list(csv.DictReader(stream))
        assert len(frames) == 80
        assert int(frames[1]["matches"]) == 2
        pairs = [json.loads(line) for line in (logs / "associations.jsonl").read_text().splitlines()]
        assert len(pairs) == 160
        assert all(0 <= p["risk"] <= 1 for p in pairs if p["risk"] is not None)
        assert (logs / "candidates.jsonl").stat().st_size > 0
    with (output / "metrics.csv").open() as stream:
        metrics = list(csv.DictReader(stream))
    clean = [m for m in metrics if m["condition"] == "all" and m["sequence"] == "clean"]
    assert all(float(m["HOTA"]) == pytest.approx(100) and int(m["IDSW"]) == 0 for m in clean)
    assert (output / "reliability_analysis.csv").exists()


def test_trackeval_reports_known_identity_switch(tmp_path):
    from era_masa.evaluation import load_trackeval, flatten_metrics
    root = ROOT / "third_party/TrackEval"
    if not root.exists():
        pytest.skip("TrackEval source is not installed")
    metrics, Dataset, _ = load_trackeval(root)
    gt = tmp_path / "gt" / "seq" / "gt"
    pred = tmp_path / "trackers" / "toy" / "data"
    gt.mkdir(parents=True)
    pred.mkdir(parents=True)
    (gt / "gt.txt").write_text("1,1,0,0,10,10,1,1,1\n2,1,0,0,10,10,1,1,1\n")
    (pred / "seq.txt").write_text("1,1,0,0,10,10,1,-1,-1,-1\n2,2,0,0,10,10,1,-1,-1,-1\n")
    dataset = Dataset({"GT_FOLDER": str(tmp_path / "gt"), "TRACKERS_FOLDER": str(tmp_path / "trackers"),
        "TRACKERS_TO_EVAL": ["toy"], "SEQ_INFO": {"seq": 2}, "SKIP_SPLIT_FOL": True,
        "DO_PREPROC": False, "PRINT_CONFIG": False})
    data = dataset.get_preprocessed_seq_data(dataset.get_raw_seq_data("toy", "seq"), "pedestrian")
    result = flatten_metrics({m.get_name(): m.eval_sequence(data) for m in metrics})
    assert result["IDSW"] == 1
    assert result["IDF1"] == pytest.approx(50)
    assert result["HOTA"] == pytest.approx(100 / np.sqrt(2))


def test_real_mot15_annotation_reader():
    path = ROOT / "third_party/MASA-OA/benchmarks/MOT15/TUD-Campus"
    if not path.exists():
        pytest.skip("Official repository data missing")
    gt = read_ground_truth(path, "MOT15")
    assert gt["length"] == 71
    assert gt["fps"] == 25
    assert gt["visibility"] is None  # MOT15 last columns are not visibility.
    assert np.all(gt["boxes"][:, 2:] > gt["boxes"][:, :2])
