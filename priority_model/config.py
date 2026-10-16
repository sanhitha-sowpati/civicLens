"""Central configuration: vocabularies, proxy-score definition, training settings.

Everything that defines *what the model means* lives here so it can be reviewed
in one place and is also written into the saved artifacts.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# Vocabularies (verified against the supplied CSVs)
# --------------------------------------------------------------------------
SEVERITY_ORDER = ["Low", "Medium", "High", "Critical"]
SEVERITY_SCORE = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
# Non-linear weights: one Critical incident counts as much as eight Low ones.
SEVERITY_WEIGHT = {"Low": 1.0, "Medium": 2.0, "High": 4.0, "Critical": 8.0}

ROAD_CONDITION_SCORE = {"Good": 0, "Fair": 1, "Poor": 2, "Critical": 3}
PEDESTRIAN_ACTIVITY_SCORE = {"Low": 0, "Medium": 1, "High": 2}

COMPLAINT_PRIORITY_WEIGHT = {"Low": 0.5, "Normal": 1.0, "High": 2.0}
OPEN_COMPLAINT_STATUSES = {"Open", "Assigned", "In progress"}
CLOSED_COMPLAINT_STATUSES = {"Resolved", "Closed"}

# Population density is converted to a coarse band so it cannot act as a
# near-unique fingerprint of a location (a hidden location-ID shortcut).
DENSITY_BAND_EDGES = [5000, 10000, 15000, 20000]  # -> bands 0..4

COORD_LAT_RANGE = (-90.0, 90.0)
COORD_LON_RANGE = (-180.0, 180.0)

# --------------------------------------------------------------------------
# Proxy priority score (0-100).  NOT an official municipal decision.
# --------------------------------------------------------------------------
# score = 100 * sum_i weight_i * component_i,  component_i in [0, 1]
# Event components use a saturating transform  s(x, k) = 1 - exp(-x / k)  so a
# single extreme location cannot dominate the scale.  The k constants are
# judgement calls for a 60-day window, chosen so a typical location lands
# mid-scale; they were set by looking at TRAINING-period magnitudes only.
PROXY_CONFIG = {
    "window_days": 60,
    "weights": {
        "severity_burden": 0.25,
        "injuries": 0.20,
        "unresolved_burden": 0.15,
        "complaint_burden": 0.10,
        "open_complaints": 0.10,
        "location_condition": 0.20,
    },
    "saturation_k": {
        "severity_burden": 6.0,
        "injuries": 1.5,
        "unresolved_burden": 3.0,
        "complaint_burden": 2.5,
        "open_complaints": 1.5,
    },
    "location_condition_weights": {
        "road_condition": 0.40,
        "pedestrian_activity": 0.20,
        "near_school": 0.15,
        "near_hospital": 0.10,
        "density_band": 0.15,
    },
}

# --------------------------------------------------------------------------
# Snapshot / split design
# --------------------------------------------------------------------------
SNAPSHOT_CONFIG = {
    "horizon_days": 60,
    "step_days": 15,
    "min_history_days": 30,
    "val_start_fraction": 0.50,
    "test_start_fraction": 0.80,
    "recent_window_days": 30,
    "long_recent_window_days": 90,
    "days_since_cap": 90,
}

# Priority levels: quantiles of PREDICTED scores on the validation split.
LEVEL_QUANTILES = {"Medium": 0.50, "High": 0.80, "Critical": 0.95}
LEVELS_DESC = ["Critical", "High", "Medium", "Low"]

CATBOOST_PARAMS = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "iterations": 2000,
    "learning_rate": 0.03,
    "depth": 4,
    "l2_leaf_reg": 10.0,
    "random_seed": 42,
    "allow_writing_files": False,
}
EARLY_STOPPING_ROUNDS = 100
RANDOM_SEED = 42
