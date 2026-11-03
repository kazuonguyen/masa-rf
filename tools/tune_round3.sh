#!/usr/bin/env bash
# Round 3 (tuning split only): MASA bi-softmax similarity and appearance recovery on Deep OC-SORT(+ORM/GAW best).
set -u
cd "$(dirname "$0")/.."
SPLIT=$1; CACHES=$2; GT=$3; BENCH=$4
T=outputs/det-tune/$SPLIT
base=$(.venv/Scripts/python.exe -c "import json;print(json.dumps(json.load(open('$T/deepocsort-rf/best.json'))['params']))")
.venv/Scripts/python.exe tools/tune_det.py --method deepocsort-rf --base "$base" \
  --space '{"bisoftmax_beta":[0.0,0.3,0.6],"temperature":[0.05,0.1],"recovery_cos":[0.0,0.7,0.8,0.9],"recovery_dist":[1.0,2.0]}' \
  --caches $CACHES --gt-root $GT --benchmark $BENCH --output $T/deepocsort-rf-r3 --workers 12 | grep -E "BEST|ok,"
echo ROUND3_DONE $SPLIT
