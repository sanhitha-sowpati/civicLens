#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python3 -m pip install -r requirements.txt
if [ ! -f artifacts/model.joblib ]; then
  python3 scripts/train.py
fi
python3 run_civiclens.py
