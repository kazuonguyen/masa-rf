#!/usr/bin/env bash
# Tune every method on one tuning split with the pre-registered spaces (configs/det_tuning_spaces.json).
set -u
cd "$(dirname "$0")/.."
SPLIT=$1; CACHES=$2; GT=$3; BENCH=$4; shift 4
for m in "$@"; do
  space=$(.venv/Scripts/python.exe -c "import json,sys;print(json.dumps(json.load(open('configs/det_tuning_spaces.json'))[sys.argv[1]]))" $m)
  .venv/Scripts/python.exe tools/tune_det.py --method $m --space "$space" --caches $CACHES --gt-root $GT --benchmark $BENCH \
     --output outputs/det-tune/$SPLIT/$m --limit 48 --workers 10 2>&1 | grep -E "BEST|ok,|Traceback"
  echo "TUNED $SPLIT $m"
done
