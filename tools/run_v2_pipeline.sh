#!/usr/bin/env bash
# Full v2 pipeline on unclipped-box caches (cache_det_v2). Same procedure as v1, fully scripted:
# tuning (round 1, edge extension, round 2, plug-ins, round 3) on tuning splits only -> lock ->
# reporting splits once -> bootstrap -> submissions -> tables.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
GPY=.venv-gpu/Scripts/python.exe
wait_for() { until [ "$(ls "$1"/*.npz 2>/dev/null | wc -l)" -ge "$2" ]; do sleep 20; done; }
step() { echo "[$(date +%H:%M:%S)] $*"; }

M="cache_det_v2/MOT17-train-ablation gt/MOT17-train-half MOT17"
D="cache_det_v2/DanceTrack-train gt/DanceTrack-train-tune DanceTrack"
ALL="bytetrack ocsort botsort botsort-reid deepocsort masa masa-oa-author masa-rf"

wait_for cache_det_v2/MOT17-train-ablation 7
step "tuning MOT17-train-half"
bash tools/tune_all.sh MOT17-train-half $M $ALL
bash tools/tune_edges.sh MOT17-train-half $M
bash tools/tune_round2.sh MOT17-train-half $M
bash tools/tune_plugins.sh MOT17-train-half $M
bash tools/tune_round3.sh MOT17-train-half $M

wait_for cache_det_v2/DanceTrack-train 6
step "tuning DanceTrack-tune"
bash tools/tune_all.sh DanceTrack-tune $D $ALL
bash tools/tune_edges.sh DanceTrack-tune $D
bash tools/tune_round2.sh DanceTrack-tune $D
bash tools/tune_plugins.sh DanceTrack-tune $D
bash tools/tune_round3.sh DanceTrack-tune $D

step "lock"
$PY tools/lock_configs.py
date > configs/det_locked.timestamp

wait_for cache_det_v2/MOT20-train-mot17det 4
wait_for cache_det_v2/DanceTrack-val 25
for s in MOT17-val DanceTrack-val MOT20-val; do
  step "reporting $s"
  $GPY tools/run_det_suite.py --split $s --reference masa-rf --workers 8 > outputs/det-suite-$s.stdout 2>&1
  $GPY tools/run_det_suite.py --split $s --reference deepocsort-rf --workers 8 >> outputs/det-suite-$s.stdout 2>&1
done

# test-set files are optional: the report uses validation splits only
if [ -d cache_det_v2/MOT17-test ]; then
  wait_for cache_det_v2/MOT17-test 7
  wait_for cache_det_v2/MOT20-test 4
  wait_for cache_det_v2/DanceTrack-test 35
  step "submissions"
  $GPY tools/make_submissions.py --methods masa-rf deepocsort-rf > outputs/submissions.stdout 2>&1
fi
$PY tools/report_det.py
step "V2_PIPELINE_DONE"
