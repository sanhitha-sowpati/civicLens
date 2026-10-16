"""Convenience loader: validated + prepared source data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from . import features as F
from .validation import load_and_validate


@dataclass
class Dataset:
    locations: pd.DataFrame
    incidents: pd.DataFrame          # validated, raw columns
    complaints: pd.DataFrame
    incidents_p: pd.DataFrame        # + derived columns (prepare_incidents)
    complaints_p: pd.DataFrame
    data_start: pd.Timestamp         # first calendar day with any record
    data_end: pd.Timestamp           # latest record timestamp
    validation_report: Dict


def load_dataset(data_dir) -> Dataset:
    loc, inc, com, rep = load_and_validate(data_dir)
    start = min(inc["timestamp"].min(), com["created_at"].min()).normalize()
    end = max(inc["timestamp"].max(), com["created_at"].max())
    return Dataset(loc, inc, com, F.prepare_incidents(inc), F.prepare_complaints(com),
                   start, end, rep)
