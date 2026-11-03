#!/usr/bin/env bash
# Plug-in tuning (tuning split only): ORM/GAW on top of the best Deep OC-SORT and BoT-SORT-ReID configs.
set -u
cd "$(dirname "$0")/.."
SPLIT=$1; CACHES=$2; GT=$3; BENCH=$4
T=outputs/det-tune/$SPLIT
pick() {
  .venv/Scripts/python.exe - "$@" <<'EOF'
import json, sys, os
c = [json.load(open(p)) for p in sys.argv[1:] if os.path.exists(p)]
best = max(c, key=lambda r: r["HOTA"])["params"]
print(json.dumps(best if isinstance(best, dict) else json.loads(best)))
EOF
}
.venv/Scripts/python.exe tools/tune_det.py --method deepocsort-rf \
  --base "$(pick $T/deepocsort/best.json $T/deepocsort-ext/best.json $T/deepocsort-r2/best.json)" \
  --space '{"orm_kappa":[0.0,0.3,0.6,0.9],"gaw_kappa":[0.0,0.25,0.5,1.0],"gaw_margin":[0.2,0.4]}' \
  --caches $CACHES --gt-root $GT --benchmark $BENCH --output $T/deepocsort-rf --workers 8 | grep -E "BEST|ok,"
.venv/Scripts/python.exe tools/tune_det.py --method botsort-reid-rf \
  --base "$(pick $T/botsort-reid/best.json $T/botsort-reid-ext/best.json $T/botsort-reid-r2/best.json)" \
  --space '{"orm_kappa":[0.0,0.3,0.6,0.9]}' \
  --caches $CACHES --gt-root $GT --benchmark $BENCH --output $T/botsort-reid-rf --workers 8 | grep -E "BEST|ok,"
echo PLUGINS_DONE $SPLIT
