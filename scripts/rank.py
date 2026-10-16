#!/usr/bin/env python
"""Print / export the current ranking using the saved model (no training, no API needed)."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from priority_model.service import PriorityService  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out", default=None, help="optional CSV path for the full ranking")
    a = ap.parse_args()
    svc = PriorityService(ROOT / "artifacts", ROOT / "data", ROOT / "runtime")
    r = svc.get_ranking()
    import pandas as pd
    df = pd.DataFrame(r["rows"])
    print(df.head(a.top).drop(columns=["supporting_evidence"]).to_string(index=False))
    if a.out:
        df.to_csv(a.out, index=False)
        print(f"Wrote {len(df)} rows to {a.out}")


if __name__ == "__main__":
    main()
