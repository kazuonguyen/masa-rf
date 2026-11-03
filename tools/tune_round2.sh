#!/usr/bin/env bash
# Round 2 (tuning split only): lifecycle/gating parameters, equal budget (48) for ours and the two
# strongest appearance baselines. Base = round-1 best of each method.
set -u
cd "$(dirname "$0")/.."
SPLIT=$1; CACHES=$2; GT=$3; BENCH=$4
best() { .venv/Scripts/python.exe -c "import json,sys;print(json.dumps(json.load(open(sys.argv[1]))['params']))" "$1"; }
pick() {  # best.json of round1 or its edge extension, whichever has higher HOTA
  .venv/Scripts/python.exe - "$@" <<'EOF'
import json, sys, os
c = [json.load(open(p)) for p in sys.argv[1:] if os.path.exists(p)]
print(json.dumps(max(c, key=lambda r: r["HOTA"])["params"]))
EOF
}
T=outputs/det-tune/$SPLIT
.venv/Scripts/python.exe tools/tune_det.py --method masa-rf --base "$(pick $T/masa-rf/best.json)" \
  --space '{"new_track_thresh":[0.6,0.7],"high_thresh":[0.5,0.6],"track_buffer":[30,60],"appearance_gate":[0.0,0.3,0.5],"geometry_gate":[0.05,0.15],"fuse_score":[true,false]}' \
  --caches $CACHES --gt-root $GT --benchmark $BENCH --output $T/masa-rf-r2 --limit 48 --workers 8 | grep -E "BEST|ok,"
.venv/Scripts/python.exe tools/tune_det.py --method botsort-reid --base "$(pick $T/botsort-reid/best.json $T/botsort-reid-ext/best.json)" \
  --space '{"new_track_thresh":[0.6,0.7],"track_high_thresh":[0.5,0.6],"track_buffer":[30,60],"proximity_thresh":[0.4,0.5,0.6],"match_thresh":[0.7,0.8],"track_low_thresh":[0.1,0.2]}' \
  --caches $CACHES --gt-root $GT --benchmark $BENCH --output $T/botsort-reid-r2 --limit 48 --workers 8 | grep -E "BEST|ok,"
.venv/Scripts/python.exe tools/tune_det.py --method deepocsort --base "$(pick $T/deepocsort/best.json $T/deepocsort-ext/best.json)" \
  --space '{"det_thresh":[0.5,0.6],"max_age":[30,60],"min_hits":[1,3],"iou_threshold":[0.2,0.3],"inertia":[0.1,0.2],"delta_t":[1,3]}' \
  --caches $CACHES --gt-root $GT --benchmark $BENCH --output $T/deepocsort-r2 --limit 48 --workers 8 | grep -E "BEST|ok,"
echo ROUND2_DONE $SPLIT
