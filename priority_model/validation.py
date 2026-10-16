"""Load and validate the three source CSVs (never modifies the originals)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from . import config as C

REQUIRED = {
    "locations": ["location_id", "zone", "road_type", "road_condition",
                  "population_density_per_km2", "near_school", "near_hospital",
                  "pedestrian_activity", "latitude", "longitude"],
    "incidents": ["incident_id", "location_id", "timestamp", "incident_type",
                  "severity", "injuries", "reported_by", "resolved",
                  "resolution_hours"],
    "public_complaints": ["complaint_id", "location_id", "created_at", "category",
                          "priority_reported", "status", "days_to_resolution"],
}


class DataValidationError(ValueError):
    """Raised when a source file cannot be used at all (e.g. missing columns)."""


def _read(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise DataValidationError(f"Missing required file: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=True)
    missing = [c for c in REQUIRED[name] if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"{path.name}: missing required columns {missing}. Found: {list(df.columns)}")
    df = df[REQUIRED[name]].copy()
    for c in df.columns:
        df[c] = df[c].str.strip()
    return df


def _norm_choice(series: pd.Series, allowed) -> pd.Series:
    """Case-insensitive match to the canonical spelling; unknown -> NaN."""
    lookup = {a.lower(): a for a in allowed}
    return series.str.lower().map(lookup)


def load_and_validate(data_dir) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Return (locations, incidents, complaints, report).

    Policy (every action is counted in ``report``):
      * duplicate identifiers  -> keep first occurrence
      * unparseable timestamps / unknown location / invalid severity -> row dropped
      * negative or non-numeric injuries -> row dropped; missing injuries -> 0
      * missing `resolved` -> treated as unresolved (conservative for triage)
      * invalid coordinates -> kept, coordinates blanked (model never uses them)
    """
    data_dir = Path(data_dir)
    rep: Dict = {"data_dir": str(data_dir), "files": {}, "actions": {}}

    # ---------------- locations ----------------
    loc = _read(data_dir / "locations.csv", "locations")
    rep["files"]["locations"] = {"rows_raw": int(len(loc))}
    rep["missing_values"] = {"locations": loc.isna().sum().to_dict()}
    dup = int(loc["location_id"].duplicated().sum())
    loc = loc[loc["location_id"].notna()].drop_duplicates("location_id", keep="first")
    for col in ["population_density_per_km2", "near_school", "near_hospital",
                "latitude", "longitude"]:
        loc[col] = pd.to_numeric(loc[col], errors="coerce")
    loc["road_condition"] = _norm_choice(loc["road_condition"], C.ROAD_CONDITION_SCORE)
    loc["pedestrian_activity"] = _norm_choice(loc["pedestrian_activity"], C.PEDESTRIAN_ACTIVITY_SCORE)
    bad_lat = ~loc["latitude"].between(*C.COORD_LAT_RANGE)
    bad_lon = ~loc["longitude"].between(*C.COORD_LON_RANGE)
    bad_coord = bad_lat | bad_lon
    loc.loc[bad_coord, ["latitude", "longitude"]] = np.nan
    rep["actions"]["locations"] = {
        "duplicate_location_ids_dropped": dup,
        "invalid_coordinates_blanked": int(bad_coord.sum()),
        "unknown_road_condition": int(loc["road_condition"].isna().sum()),
        "unknown_pedestrian_activity": int(loc["pedestrian_activity"].isna().sum()),
    }
    loc = loc.reset_index(drop=True)
    known = set(loc["location_id"])

    # ---------------- incidents ----------------
    inc = _read(data_dir / "incidents.csv", "incidents")
    rep["files"]["incidents"] = {"rows_raw": int(len(inc))}
    rep["missing_values"]["incidents"] = inc.isna().sum().to_dict()
    a: Dict[str, int] = {}
    a["duplicate_incident_ids_dropped"] = int(inc["incident_id"].duplicated().sum())
    inc = inc.drop_duplicates("incident_id", keep="first")
    inc["timestamp"] = pd.to_datetime(inc["timestamp"], errors="coerce")
    a["invalid_timestamps_dropped"] = int(inc["timestamp"].isna().sum())
    inc = inc[inc["timestamp"].notna()]
    inc["severity"] = _norm_choice(inc["severity"], C.SEVERITY_SCORE)
    a["invalid_severity_dropped"] = int(inc["severity"].isna().sum())
    inc = inc[inc["severity"].notna()]
    unk = ~inc["location_id"].isin(known)
    a["unknown_location_ids_dropped"] = int(unk.sum())
    inc = inc[~unk]
    inj = pd.to_numeric(inc["injuries"], errors="coerce")
    a["missing_injuries_set_to_zero"] = int(inj.isna().sum())
    inj = inj.fillna(0)
    neg = inj < 0
    a["negative_injuries_dropped"] = int(neg.sum())
    inc = inc[~neg.values].copy()
    inc["injuries"] = inj[~neg].astype(int).values
    res = pd.to_numeric(inc["resolved"], errors="coerce")
    a["missing_resolved_treated_unresolved"] = int(res.isna().sum())
    inc["resolved"] = res.fillna(0).clip(0, 1).astype(int)
    rh = pd.to_numeric(inc["resolution_hours"], errors="coerce")
    a["negative_resolution_hours_blanked"] = int((rh < 0).sum())
    rh[rh < 0] = np.nan
    inc["resolution_hours"] = rh
    inc["incident_type"] = inc["incident_type"].fillna("Unknown")
    inc["reported_by"] = inc["reported_by"].fillna("Unknown")
    rep["actions"]["incidents"] = a
    inc = inc.sort_values("timestamp").reset_index(drop=True)

    # ---------------- complaints ----------------
    com = _read(data_dir / "public_complaints.csv", "public_complaints")
    rep["files"]["public_complaints"] = {"rows_raw": int(len(com))}
    rep["missing_values"]["public_complaints"] = com.isna().sum().to_dict()
    b: Dict[str, int] = {}
    b["duplicate_complaint_ids_dropped"] = int(com["complaint_id"].duplicated().sum())
    com = com.drop_duplicates("complaint_id", keep="first")
    com["created_at"] = pd.to_datetime(com["created_at"], errors="coerce")
    b["invalid_timestamps_dropped"] = int(com["created_at"].isna().sum())
    com = com[com["created_at"].notna()]
    unk = ~com["location_id"].isin(known)
    b["unknown_location_ids_dropped"] = int(unk.sum())
    com = com[~unk].copy()
    com["priority_reported"] = _norm_choice(com["priority_reported"], C.COMPLAINT_PRIORITY_WEIGHT)
    b["unknown_priority_set_to_Normal"] = int(com["priority_reported"].isna().sum())
    com["priority_reported"] = com["priority_reported"].fillna("Normal")
    st_ok = C.OPEN_COMPLAINT_STATUSES | C.CLOSED_COMPLAINT_STATUSES
    st = _norm_choice(com["status"], st_ok)
    b["unknown_status_treated_open"] = int(st.isna().sum())
    com["status"] = st.fillna("Open")
    d = pd.to_numeric(com["days_to_resolution"], errors="coerce")
    b["negative_days_to_resolution_blanked"] = int((d < 0).sum())
    d[d < 0] = np.nan
    com["days_to_resolution"] = d
    com["category"] = com["category"].fillna("Unknown")
    rep["actions"]["public_complaints"] = b
    com = com.sort_values("created_at").reset_index(drop=True)

    for name, df in (("locations", loc), ("incidents", inc), ("public_complaints", com)):
        rep["files"][name]["rows_valid"] = int(len(df))
    rep["time_range"] = {
        "incidents": [str(inc["timestamp"].min()), str(inc["timestamp"].max())],
        "complaints": [str(com["created_at"].min()), str(com["created_at"].max())],
    }
    rep["vocabularies"] = {
        "incident_types": sorted(inc["incident_type"].unique().tolist()),
        "complaint_categories": sorted(com["category"].unique().tolist()),
    }
    return loc, inc, com, rep
