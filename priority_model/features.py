"""Point-in-time feature engineering and the documented proxy priority score.

Every aggregate is computed *as of* a timestamp: only records with
``timestamp <= as_of`` are used, and "unresolved" / "open" status is
reconstructed at ``as_of`` from ``timestamp + resolution_hours`` (incidents) or
``created_at + days_to_resolution`` (complaints), so information that only
became known later is never used.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import config as C

CATEGORICAL_FEATURES: List[str] = ["zone", "road_type"]
NUMERIC_FEATURES: List[str] = [
    # incident history (rates are per 30 days of available history)
    "incident_rate_per_30d", "critical_incident_rate_per_30d",
    "high_incident_rate_per_30d", "severity_weighted_rate_per_30d",
    "mean_severity_score", "injury_rate_per_30d",
    "unresolved_incident_rate_per_30d", "unresolved_incident_share",
    "incident_count_last_30d", "incident_rate_last_90d", "injury_rate_last_90d",
    "days_since_last_incident",
    # complaint history
    "complaint_rate_per_30d", "open_complaint_rate_per_30d", "open_complaint_share",
    "high_priority_complaint_rate_per_30d", "complaint_count_last_30d",
    "complaint_rate_last_90d", "days_since_last_complaint",
    # location characteristics
    "road_condition_score", "pedestrian_activity_score", "near_school",
    "near_hospital", "population_density_band",
]
# Order matters: this is the feature order used for training AND inference.
FEATURE_COLUMNS: List[str] = CATEGORICAL_FEATURES[:1] + CATEGORICAL_FEATURES[1:] + NUMERIC_FEATURES
FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "zone": "Administrative zone (categorical).",
    "road_type": "Road class (categorical).",
    "incident_rate_per_30d": "Incidents to date per 30 days of history.",
    "critical_incident_rate_per_30d": "Critical incidents to date per 30 days of history.",
    "high_incident_rate_per_30d": "High-severity incidents to date per 30 days of history.",
    "severity_weighted_rate_per_30d": "Severity-weighted incident burden per 30 days of history.",
    "mean_severity_score": "Mean severity (Low=1 .. Critical=4); missing if no incidents.",
    "injury_rate_per_30d": "Injuries to date per 30 days of history.",
    "unresolved_incident_rate_per_30d": "Incidents unresolved at prediction time, per 30 days of history.",
    "unresolved_incident_share": "Share of incidents unresolved at prediction time.",
    "incident_count_last_30d": "Incidents in the 30 days before prediction time.",
    "incident_rate_last_90d": "Incidents in the last 90 days, per 30 days.",
    "injury_rate_last_90d": "Injuries in the last 90 days, per 30 days.",
    "days_since_last_incident": "Days since latest incident (capped at 90).",
    "complaint_rate_per_30d": "Complaints to date per 30 days of history.",
    "open_complaint_rate_per_30d": "Complaints open at prediction time, per 30 days of history.",
    "open_complaint_share": "Share of complaints open at prediction time.",
    "high_priority_complaint_rate_per_30d": "Citizen-flagged High-priority complaints per 30 days.",
    "complaint_count_last_30d": "Complaints in the 30 days before prediction time.",
    "complaint_rate_last_90d": "Complaints in the last 90 days, per 30 days.",
    "days_since_last_complaint": "Days since latest complaint (capped at 90).",
    "road_condition_score": "Good=0, Fair=1, Poor=2, Critical=3.",
    "pedestrian_activity_score": "Low=0, Medium=1, High=2.",
    "near_school": "1 if near a school.",
    "near_hospital": "1 if near a hospital.",
    "population_density_band": "Density band 0-4 (edges 5k/10k/15k/20k per km2).",
}

# Raw (non-model) aggregates used for the ranking table and evidence text.
RAW_COLUMNS = ["incident_count", "critical_count", "high_count", "injury_count",
               "unresolved_incident_count", "complaint_count", "open_complaint_count",
               "last_incident_days_ago", "last_complaint_days_ago"]


# --------------------------------------------------------------------------
# Static location transformation
# --------------------------------------------------------------------------
def density_band(values) -> pd.Series:
    v = pd.to_numeric(pd.Series(values), errors="coerce")
    band = np.digitize(v.fillna(-1).values, C.DENSITY_BAND_EDGES)
    out = pd.Series(band, index=v.index, dtype="float64")
    out[v.isna()] = np.nan
    return out


def location_static_features(locations: pd.DataFrame) -> pd.DataFrame:
    """Static model features per location (indexed by location_id)."""
    loc = locations.set_index("location_id")
    out = pd.DataFrame(index=loc.index)
    out["zone"] = loc["zone"].fillna("UNKNOWN").astype(str)
    out["road_type"] = loc["road_type"].fillna("UNKNOWN").astype(str)
    out["road_condition_score"] = loc["road_condition"].map(C.ROAD_CONDITION_SCORE).astype(float)
    out["pedestrian_activity_score"] = loc["pedestrian_activity"].map(C.PEDESTRIAN_ACTIVITY_SCORE).astype(float)
    out["near_school"] = pd.to_numeric(loc["near_school"], errors="coerce")
    out["near_hospital"] = pd.to_numeric(loc["near_hospital"], errors="coerce")
    out["population_density_band"] = density_band(loc["population_density_per_km2"]).values
    return out


# --------------------------------------------------------------------------
# Event preparation
# --------------------------------------------------------------------------
def prepare_incidents(inc: pd.DataFrame) -> pd.DataFrame:
    df = inc.copy()
    df["severity_score"] = df["severity"].map(C.SEVERITY_SCORE).astype(float)
    df["severity_weight"] = df["severity"].map(C.SEVERITY_WEIGHT).astype(float)
    hrs = pd.to_timedelta(df["resolution_hours"], unit="h")
    resolved_at = df["timestamp"] + hrs
    # resolved but duration unknown -> assume it was resolved immediately
    resolved_at = resolved_at.where(resolved_at.notna() | (df["resolved"] != 1), df["timestamp"])
    df["resolved_at"] = resolved_at
    df["is_critical"] = (df["severity"] == "Critical").astype(int)
    df["is_high"] = (df["severity"] == "High").astype(int)
    return df


def prepare_complaints(com: pd.DataFrame) -> pd.DataFrame:
    df = com.copy()
    df["priority_weight"] = df["priority_reported"].map(C.COMPLAINT_PRIORITY_WEIGHT).astype(float)
    df["open_status"] = df["status"].isin(C.OPEN_COMPLAINT_STATUSES)
    df["resolved_at"] = df["created_at"] + pd.to_timedelta(df["days_to_resolution"], unit="D")
    df["is_high_priority"] = (df["priority_reported"] == "High").astype(int)
    return df


# --------------------------------------------------------------------------
# Window aggregation
# --------------------------------------------------------------------------
def aggregate_window(loc_ids, inc: pd.DataFrame, com: pd.DataFrame,
                     start: Optional[pd.Timestamp], end: pd.Timestamp,
                     resolution_time: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Aggregate events in the window (start, end] per location.

    Unresolved/open status is evaluated at ``resolution_time`` (default ``end``).
    """
    resolution_time = end if resolution_time is None else resolution_time
    m = inc["timestamp"] <= end
    if start is not None:
        m &= inc["timestamp"] > start
    s = inc.loc[m]
    unres = ((s["resolved"] == 0) | (s["resolved_at"] > resolution_time)).astype(int)
    s = s.assign(unres=unres, unres_w=unres * s["severity_weight"],
                 has_injury=(s["injuries"] > 0).astype(int))
    g = s.groupby("location_id")
    ia = pd.DataFrame({
        "inc_count": g.size(),
        "inc_critical": g["is_critical"].sum(),
        "inc_high": g["is_high"].sum(),
        "inc_sev_weight": g["severity_weight"].sum(),
        "inc_sev_score_sum": g["severity_score"].sum(),
        "inc_injuries": g["injuries"].sum(),
        "inc_unresolved": g["unres"].sum(),
        "inc_unresolved_weight": g["unres_w"].sum(),
        "last_inc_ts": g["timestamp"].max(),
    })

    m = com["created_at"] <= end
    if start is not None:
        m &= com["created_at"] > start
    c = com.loc[m]
    copen = (c["open_status"] | (c["resolved_at"] > resolution_time)).astype(int)
    c = c.assign(open_flag=copen)
    g = c.groupby("location_id")
    ca = pd.DataFrame({
        "comp_count": g.size(),
        "comp_open": g["open_flag"].sum(),
        "comp_high": g["is_high_priority"].sum(),
        "comp_weight": g["priority_weight"].sum(),
        "last_comp_ts": g["created_at"].max(),
    })
    out = ia.join(ca, how="outer").reindex(pd.Index(loc_ids, name="location_id"))
    ts_cols = ["last_inc_ts", "last_comp_ts"]
    num_cols = [c for c in out.columns if c not in ts_cols]
    out[num_cols] = out[num_cols].fillna(0.0)
    return out


# --------------------------------------------------------------------------
# Proxy priority score
# --------------------------------------------------------------------------
def location_condition_component(static: pd.DataFrame) -> pd.Series:
    w = C.PROXY_CONFIG["location_condition_weights"]
    comp = (w["road_condition"] * static["road_condition_score"] / 3.0
            + w["pedestrian_activity"] * static["pedestrian_activity_score"] / 2.0
            + w["near_school"] * static["near_school"]
            + w["near_hospital"] * static["near_hospital"]
            + w["density_band"] * static["population_density_band"] / 4.0)
    return comp.fillna(0.0).clip(0, 1)


def proxy_priority_score(agg: pd.DataFrame, static: pd.DataFrame) -> pd.Series:
    """Documented proxy priority score on a 0-100 scale (see README)."""
    cfg = C.PROXY_CONFIG
    wts, k = cfg["weights"], cfg["saturation_k"]

    def sat(x, kk):
        return 1.0 - np.exp(-x / kk)

    score = 100.0 * (
        wts["severity_burden"] * sat(agg["inc_sev_weight"], k["severity_burden"])
        + wts["injuries"] * sat(agg["inc_injuries"], k["injuries"])
        + wts["unresolved_burden"] * sat(agg["inc_unresolved_weight"], k["unresolved_burden"])
        + wts["complaint_burden"] * sat(agg["comp_weight"], k["complaint_burden"])
        + wts["open_complaints"] * sat(agg["comp_open"], k["open_complaints"])
        + wts["location_condition"] * location_condition_component(static).reindex(agg.index)
    )
    return score.clip(0, 100).rename("proxy_priority_score")


# --------------------------------------------------------------------------
# Feature table
# --------------------------------------------------------------------------
def build_features(locations: pd.DataFrame, inc: pd.DataFrame, com: pd.DataFrame,
                   as_of: pd.Timestamp, data_start: pd.Timestamp,
                   loc_ids=None) -> pd.DataFrame:
    """Point-in-time features + raw display aggregates for each location.

    ``inc``/``com`` must already be passed through ``prepare_*``.
    Returns a frame indexed by location_id with FEATURE_COLUMNS + RAW_COLUMNS.
    """
    sc = C.SNAPSHOT_CONFIG
    static = location_static_features(locations)
    if loc_ids is not None:
        static = static.loc[list(loc_ids)]
    ids = static.index
    if loc_ids is not None:  # single/partial-location update: only scan those records
        inc = inc[inc["location_id"].isin(ids)]
        com = com[com["location_id"].isin(ids)]

    hist_days = max((as_of - data_start).total_seconds() / 86400.0, 1.0)
    months = max(hist_days / 30.0, 1.0 / 30.0)
    d90 = min(sc["long_recent_window_days"], hist_days)
    m90 = d90 / 30.0
    cap = sc["days_since_cap"]

    allw = aggregate_window(ids, inc, com, None, as_of)
    w30 = aggregate_window(ids, inc, com, as_of - pd.Timedelta(days=sc["recent_window_days"]), as_of)
    w90 = aggregate_window(ids, inc, com, as_of - pd.Timedelta(days=sc["long_recent_window_days"]), as_of)

    f = pd.DataFrame(index=ids)
    f["zone"] = static["zone"]
    f["road_type"] = static["road_type"]
    f["incident_rate_per_30d"] = allw["inc_count"] / months
    f["critical_incident_rate_per_30d"] = allw["inc_critical"] / months
    f["high_incident_rate_per_30d"] = allw["inc_high"] / months
    f["severity_weighted_rate_per_30d"] = allw["inc_sev_weight"] / months
    f["mean_severity_score"] = (allw["inc_sev_score_sum"] / allw["inc_count"].replace(0, np.nan))
    f["injury_rate_per_30d"] = allw["inc_injuries"] / months
    f["unresolved_incident_rate_per_30d"] = allw["inc_unresolved"] / months
    f["unresolved_incident_share"] = allw["inc_unresolved"] / allw["inc_count"].replace(0, np.nan)
    f["unresolved_incident_share"] = f["unresolved_incident_share"].fillna(0.0)
    f["incident_count_last_30d"] = w30["inc_count"]
    f["incident_rate_last_90d"] = w90["inc_count"] / m90
    f["injury_rate_last_90d"] = w90["inc_injuries"] / m90
    f["days_since_last_incident"] = ((as_of - allw["last_inc_ts"]).dt.total_seconds() / 86400.0).clip(upper=cap).fillna(cap)
    f["complaint_rate_per_30d"] = allw["comp_count"] / months
    f["open_complaint_rate_per_30d"] = allw["comp_open"] / months
    f["open_complaint_share"] = (allw["comp_open"] / allw["comp_count"].replace(0, np.nan)).fillna(0.0)
    f["high_priority_complaint_rate_per_30d"] = allw["comp_high"] / months
    f["complaint_count_last_30d"] = w30["comp_count"]
    f["complaint_rate_last_90d"] = w90["comp_count"] / m90
    f["days_since_last_complaint"] = ((as_of - allw["last_comp_ts"]).dt.total_seconds() / 86400.0).clip(upper=cap).fillna(cap)
    for col in ["road_condition_score", "pedestrian_activity_score", "near_school",
                "near_hospital", "population_density_band"]:
        f[col] = static[col]
    f = f[FEATURE_COLUMNS].astype({c: "float64" for c in NUMERIC_FEATURES})

    raw = pd.DataFrame(index=ids)
    raw["incident_count"] = allw["inc_count"].astype(int)
    raw["critical_count"] = allw["inc_critical"].astype(int)
    raw["high_count"] = allw["inc_high"].astype(int)
    raw["injury_count"] = allw["inc_injuries"].astype(int)
    raw["unresolved_incident_count"] = allw["inc_unresolved"].astype(int)
    raw["complaint_count"] = allw["comp_count"].astype(int)
    raw["open_complaint_count"] = allw["comp_open"].astype(int)
    raw["last_incident_days_ago"] = (as_of - allw["last_inc_ts"]).dt.total_seconds() / 86400.0
    raw["last_complaint_days_ago"] = (as_of - allw["last_comp_ts"]).dt.total_seconds() / 86400.0
    return f.join(raw)
