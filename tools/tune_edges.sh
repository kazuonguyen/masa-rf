#!/usr/bin/env bash
# Edge extension: grids where the best value sat on the boundary are extended (applies to any method).
set -u
cd "$(dirname "$0")/.."
SPLIT=$1; CACHES=$2; GT=$3; BENCH=$4
.venv/Scripts/python.exe tools/tune_det.py --method botsort-reid --space '{"appearance_thresh": [0.35, 0.4, 0.45, 0.5], "proximity_thresh": [0.4, 0.5, 0.6], "match_thresh": [0.7, 0.8]}' --caches $CACHES --gt-root $GT --benchmark $BENCH --output outputs/det-tune/$SPLIT/botsort-reid-ext --workers 10 | grep -E "BEST|ok,"
.venv/Scripts/python.exe tools/tune_det.py --method deepocsort --space '{"w_association_emb": [1.5, 2.0, 3.0], "alpha_fixed_emb": [0.8, 0.9], "det_thresh": [0.5, 0.6, 0.7], "aw_param": [0.5, 1.0]}' --caches $CACHES --gt-root $GT --benchmark $BENCH --output outputs/det-tune/$SPLIT/deepocsort-ext --workers 10 | grep -E "BEST|ok,"
echo EDGES_DONE
