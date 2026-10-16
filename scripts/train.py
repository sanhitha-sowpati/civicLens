#!/usr/bin/env python
"""Explicit (re)training.  Never invoked automatically by the service.

    python scripts/train.py                 # full training
    python scripts/train.py --quick         # fewer iterations, for smoke tests only
Outputs: artifacts/ (model + schema + config + metrics) and reports/ (evaluation,
plots, ranked_locations.csv).  The original CSVs in data/ are only read.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from priority_model.training import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--artifacts-dir", default=str(ROOT / "artifacts"))
    ap.add_argument("--reports-dir", default=str(ROOT / "reports"))
    ap.add_argument("--quick", action="store_true", help="reduced iterations (not for production)")
    a = ap.parse_args()
    train(a.data_dir, a.artifacts_dir, a.reports_dir, quick=a.quick)


if __name__ == "__main__":
    main()
