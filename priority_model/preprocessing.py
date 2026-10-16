"""Feature schema + record preprocessing shared by training and inference."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config as C
from . import features as F


class InvalidRecordError(ValueError):
    """Raised when inference input is unusable. ``errors`` lists every problem."""

    def __init__(self, errors: Sequence[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


# Raw location-style fields accepted in addition to the model features; they are
# converted to the model's numeric features (and are otherwise ignored).
DERIVABLE = {"road_condition", "pedestrian_activity", "population_density_per_km2"}
PASSTHROUGH = {"location_id", "latitude", "longitude", "name", "notes"}

RANGES = {  # inclusive (min, max); None = unbounded
    "road_condition_score": (0, 3), "pedestrian_activity_score": (0, 2),
    "near_school": (0, 1), "near_hospital": (0, 1),
    "population_density_band": (0, 4), "unresolved_incident_share": (0, 1),
    "open_complaint_share": (0, 1), "mean_severity_score": (1, 4),
    "days_since_last_incident": (0, C.SNAPSHOT_CONFIG["days_since_cap"]),
    "days_since_last_complaint": (0, C.SNAPSHOT_CONFIG["days_since_cap"]),
}
MAX_MISSING_FRACTION = 0.5


def build_schema(train_features: pd.DataFrame) -> Dict[str, Any]:
    """Feature schema saved next to the model (order == training column order)."""
    feats = []
    for name in F.FEATURE_COLUMNS:
        if name in F.CATEGORICAL_FEATURES:
            feats.append({"name": name, "dtype": "categorical",
                          "allowed_values": sorted(train_features[name].astype(str).unique().tolist()),
                          "description": F.FEATURE_DESCRIPTIONS[name]})
        else:
            col = train_features[name]
            lo, hi = RANGES.get(name, (0, None))
            feats.append({"name": name, "dtype": "numeric",
                          "valid_min": lo, "valid_max": hi,
                          "train_min": float(col.min()), "train_max": float(col.max()),
                          "missing_allowed": True,
                          "description": F.FEATURE_DESCRIPTIONS[name]})
    return {"feature_order": list(F.FEATURE_COLUMNS),
            "categorical_features": list(F.CATEGORICAL_FEATURES),
            "numeric_features": list(F.NUMERIC_FEATURES),
            "features": feats}


class FeaturePreprocessor:
    """Validates records and returns a model-ready frame in training column order."""

    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
        self.order: List[str] = list(schema["feature_order"])
        self.cat: List[str] = list(schema["categorical_features"])
        self.num: List[str] = list(schema["numeric_features"])
        self.allowed = {f["name"]: set(f["allowed_values"])
                        for f in schema["features"] if f["dtype"] == "categorical"}

    @staticmethod
    def _derive(rec: Dict[str, Any]) -> Dict[str, Any]:
        rec = dict(rec)
        if rec.get("road_condition_score") is None and rec.get("road_condition") is not None:
            key = str(rec["road_condition"]).strip().title()
            if key in C.ROAD_CONDITION_SCORE:
                rec["road_condition_score"] = C.ROAD_CONDITION_SCORE[key]
        if rec.get("pedestrian_activity_score") is None and rec.get("pedestrian_activity") is not None:
            key = str(rec["pedestrian_activity"]).strip().title()
            if key in C.PEDESTRIAN_ACTIVITY_SCORE:
                rec["pedestrian_activity_score"] = C.PEDESTRIAN_ACTIVITY_SCORE[key]
        if rec.get("population_density_band") is None and rec.get("population_density_per_km2") is not None:
            try:
                rec["population_density_band"] = int(np.digitize(float(rec["population_density_per_km2"]),
                                                                  C.DENSITY_BAND_EDGES))
            except (TypeError, ValueError):
                pass
        return rec

    def transform(self, records: Iterable[Dict[str, Any]]) -> Tuple[pd.DataFrame, List[List[str]]]:
        records = list(records)
        if not records:
            raise InvalidRecordError(["No records supplied."])
        errors: List[str] = []
        warnings: List[List[str]] = []
        rows: List[Dict[str, Any]] = []
        known = set(self.order) | DERIVABLE | PASSTHROUGH
        for i, raw in enumerate(records):
            if not isinstance(raw, dict):
                errors.append(f"record {i}: must be an object/dict, got {type(raw).__name__}")
                rows.append({}); warnings.append([]); continue
            unknown = sorted(set(raw) - known)
            if unknown:
                errors.append(f"record {i}: unknown field(s) {unknown}. Valid model features: {self.order}")
            rec = self._derive(raw)
            w: List[str] = []
            row: Dict[str, Any] = {}
            missing = 0
            for name in self.cat:
                v = rec.get(name)
                if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "":
                    row[name] = "UNKNOWN"; missing += 1
                    w.append(f"'{name}' missing; using 'UNKNOWN'.")
                else:
                    v = str(v).strip()
                    if v not in self.allowed[name]:
                        w.append(f"'{name}'={v!r} not seen in training; treated as 'UNKNOWN'.")
                        v = "UNKNOWN"
                    row[name] = v
            for name in self.num:
                v = rec.get(name)
                if v is None or (isinstance(v, str) and v.strip() == ""):
                    row[name] = np.nan; missing += 1
                    w.append(f"'{name}' missing; passed as NaN.")
                    continue
                try:
                    x = float(v)
                except (TypeError, ValueError):
                    errors.append(f"record {i}: '{name}'={v!r} is not numeric")
                    row[name] = np.nan; continue
                if np.isnan(x):
                    row[name] = np.nan; missing += 1
                    w.append(f"'{name}' is NaN; passed as NaN."); continue
                if np.isinf(x):
                    errors.append(f"record {i}: '{name}' is infinite"); row[name] = np.nan; continue
                lo, hi = RANGES.get(name, (0, None))
                if x < lo or (hi is not None and x > hi):
                    errors.append(f"record {i}: '{name}'={x} outside valid range [{lo}, {hi}]")
                row[name] = x
            if missing / len(self.order) > MAX_MISSING_FRACTION:
                errors.append(f"record {i}: {missing}/{len(self.order)} features missing "
                              f"(limit {int(MAX_MISSING_FRACTION*100)}%); cannot predict reliably")
            rows.append(row); warnings.append(w)
        if errors:
            raise InvalidRecordError(errors)
        X = pd.DataFrame(rows)[self.order]
        X = X.astype({c: "float64" for c in self.num})
        for c in self.cat:
            X[c] = X[c].astype(str)
        return X, warnings
