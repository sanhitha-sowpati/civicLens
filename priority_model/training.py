"""Training pipeline: snapshots -> proxy labels -> CatBoost -> evaluation -> artifacts.

Design (see README for the full rationale)
------------------------------------------
* No genuine target exists in the data (no municipal decision / repair-urgency
  field), so the label is a documented PROXY priority score (features.proxy_priority_score).
* To avoid the proxy trivially being a function of the model inputs, training
  uses point-in-time snapshots: for each cutoff date, features use only records
  up to the cutoff and the label is the proxy score of the 60 days AFTER it.
* Train / validation / test are separated by cutoff date, with a purge gap so
  that label windows never overlap across splits.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from . import config as C
from . import features as F
from . import preprocessing as P
from . import ranking as R
from .dataset import Dataset, load_dataset

ARTIFACT_FILES = {
    "model": "model.joblib", "native": "model.cbm", "schema": "feature_schema.json",
    "preprocessing": "preprocessing_config.json", "metrics": "training_metrics.json",
    "metadata": "model_metadata.json",
}


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------
def make_cutoffs(data_start: pd.Timestamp, data_end: pd.Timestamp, sc=None) -> List[pd.Timestamp]:
    sc = sc or C.SNAPSHOT_CONFIG
    first = data_start + pd.Timedelta(days=sc["min_history_days"])
    last = data_end - pd.Timedelta(days=sc["horizon_days"])
    if last < first:
        raise ValueError("Not enough history for the configured horizon/min_history_days.")
    return list(pd.date_range(first, last, freq=f"{sc['step_days']}D"))


def build_snapshots(ds: Dataset, cutoffs: List[pd.Timestamp], sc=None) -> pd.DataFrame:
    """One row per (cutoff, location): features, forward proxy label, persistence baseline."""
    sc = sc or C.SNAPSHOT_CONFIG
    H = pd.Timedelta(days=sc["horizon_days"])
    static = F.location_static_features(ds.locations)
    ids = static.index
    frames = []
    for c in cutoffs:
        feats = F.build_features(ds.locations, ds.incidents_p, ds.complaints_p, c, ds.data_start)
        fwd = F.aggregate_window(ids, ds.incidents_p, ds.complaints_p, c, c + H, resolution_time=c + H)
        past = F.aggregate_window(ids, ds.incidents_p, ds.complaints_p, c - H, c, resolution_time=c)
        snap = feats[F.FEATURE_COLUMNS].copy()
        snap["label"] = F.proxy_priority_score(fwd, static)
        snap["persistence"] = F.proxy_priority_score(past, static)
        snap["cutoff"] = c
        snap["location_id"] = snap.index
        frames.append(snap.reset_index(drop=True))
    return pd.concat(frames, ignore_index=True)


def split_cutoffs(cutoffs: List[pd.Timestamp], sc=None) -> Dict[str, List[pd.Timestamp]]:
    """Time-ordered split with a purge gap (label window length) between splits."""
    sc = sc or C.SNAPSHOT_CONFIG
    H = pd.Timedelta(days=sc["horizon_days"])
    n = len(cutoffs)
    val_start = cutoffs[int(n * sc["val_start_fraction"])]
    test_start = cutoffs[min(int(round(n * sc["test_start_fraction"])), n - 1)]
    train = [c for c in cutoffs if c + H <= val_start]
    val = [c for c in cutoffs if val_start <= c and c + H <= test_start]
    test = [c for c in cutoffs if c >= test_start]
    if not (train and val and test):
        raise ValueError(f"Split produced an empty partition (train={len(train)}, val={len(val)}, test={len(test)}).")
    return {"train": train, "val": val, "test": test}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {"mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)) if np.var(y_true) > 0 else float("nan")}


def ranking_metrics(df: pd.DataFrame, pred_col: str, top_frac: float = 0.05) -> Dict[str, float]:
    """Per-cutoff Spearman and top-k overlap, averaged across cutoffs."""
    sp, hit = [], []
    for _, g in df.groupby("cutoff"):
        if g[pred_col].nunique() > 1:
            sp.append(g[pred_col].corr(g["label"], method="spearman"))
        k = max(1, int(np.ceil(top_frac * len(g))))
        a = set(g.nlargest(k, pred_col)["location_id"])
        b = set(g.nlargest(k, "label")["location_id"])
        hit.append(len(a & b) / k)
    return {"spearman": float(np.nanmean(sp)) if sp else float("nan"),
            f"top{int(top_frac*100)}pct_overlap": float(np.mean(hit)),
            f"top{int(top_frac*100)}pct_overlap_by_chance": top_frac}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _fit_catboost(Xtr, ytr, Xva, yva, params, early_stopping_rounds):
    from catboost import CatBoostRegressor  # imported lazily so other modules work without it
    model = CatBoostRegressor(**params)
    model.fit(Xtr, ytr, cat_features=F.CATEGORICAL_FEATURES, eval_set=(Xva, yva),
              early_stopping_rounds=early_stopping_rounds, use_best_model=True, verbose=False)
    return model


def _fingerprint(data_dir: Path, params: dict) -> str:
    h = hashlib.sha256()
    for name in ("locations.csv", "incidents.csv", "public_complaints.csv"):
        h.update((Path(data_dir) / name).read_bytes())
    h.update(json.dumps(params, sort_keys=True).encode())
    h.update(json.dumps(C.PROXY_CONFIG, sort_keys=True).encode())
    return h.hexdigest()[:10]


def train(data_dir, artifacts_dir, reports_dir, quick: bool = False,
          params: Optional[dict] = None, verbose: bool = True) -> Dict:
    log = print if verbose else (lambda *a, **k: None)
    data_dir, artifacts_dir, reports_dir = Path(data_dir), Path(artifacts_dir), Path(reports_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    params = dict(C.CATBOOST_PARAMS, **(params or {}))
    if quick:
        params["iterations"] = min(params["iterations"], 300)
        params["learning_rate"] = max(params["learning_rate"], 0.08)

    log("[1/7] Loading and validating data ...")
    ds = load_dataset(data_dir)
    (reports_dir / "data_validation_report.json").write_text(json.dumps(ds.validation_report, indent=2, default=str))

    log("[2/7] Building point-in-time snapshots and proxy labels ...")
    cutoffs = make_cutoffs(ds.data_start, ds.data_end)
    snaps = build_snapshots(ds, cutoffs)
    split = split_cutoffs(cutoffs)
    parts = {k: snaps[snaps["cutoff"].isin(v)].reset_index(drop=True) for k, v in split.items()}
    for k, v in split.items():
        log(f"      {k:5s}: {len(v)} cutoffs, {len(parts[k])} rows ({v[0].date()} .. {v[-1].date()})")

    Xtr, ytr = parts["train"][F.FEATURE_COLUMNS], parts["train"]["label"]
    Xva, yva = parts["val"][F.FEATURE_COLUMNS], parts["val"]["label"]
    Xte, yte = parts["test"][F.FEATURE_COLUMNS], parts["test"]["label"]

    log("[3/7] Training CatBoostRegressor (early stopping on the validation split) ...")
    model = _fit_catboost(Xtr, ytr, Xva, yva, params, C.EARLY_STOPPING_ROUNDS)
    best_iter = int(model.get_best_iteration())

    log("[4/7] Evaluating on validation and test splits ...")
    pred = {k: R.clip_scores(model.predict(parts[k][F.FEATURE_COLUMNS])) for k in ("train", "val", "test")}
    train_mean = float(ytr.mean())
    metrics: Dict = {"target": "proxy_priority_score (0-100), NOT verified municipal decisions"}
    for k in ("train", "val", "test"):
        y = parts[k]["label"]
        metrics[k] = {
            "n_rows": int(len(y)),
            "catboost": regression_metrics(y, pred[k]),
            "baseline_train_mean": regression_metrics(y, np.full(len(y), train_mean)),
            "baseline_persistence_prev_60d": regression_metrics(y, parts[k]["persistence"]),
        }
    for k in ("val", "test"):
        d = parts[k].assign(pred=pred[k])
        metrics[k]["catboost"].update(ranking_metrics(d, "pred"))
        metrics[k]["baseline_persistence_prev_60d"].update(ranking_metrics(d, "persistence"))
    metrics["label_stats_train"] = {"mean": train_mean, "std": float(ytr.std()),
                                    "min": float(ytr.min()), "max": float(ytr.max())}
    # share of label variance that is purely static location condition (upper bound on
    # what location characteristics alone could explain)
    static = F.location_static_features(ds.locations)
    cond = F.location_condition_component(static) * 100 * C.PROXY_CONFIG["weights"]["location_condition"]
    metrics["static_condition_component_std"] = float(cond.std())
    metrics["best_iteration"] = best_iter

    log("[5/7] Level thresholds, final ranking, importance ...")
    thresholds = R.level_thresholds_from_scores(pred["val"])
    as_of = ds.data_end
    feats_now = F.build_features(ds.locations, ds.incidents_p, ds.complaints_p, as_of, ds.data_start)
    scores_now = R.clip_scores(model.predict(feats_now[F.FEATURE_COLUMNS]))
    static_now = F.location_static_features(ds.locations)
    H = pd.Timedelta(days=C.SNAPSHOT_CONFIG["horizon_days"])
    recent = F.proxy_priority_score(
        F.aggregate_window(static_now.index, ds.incidents_p, ds.complaints_p, as_of - H, as_of), static_now)
    ranked = R.build_ranking(ds.locations, feats_now, scores_now, thresholds, ds.incidents_p, recent, as_of)
    ranked.to_csv(reports_dir / "ranked_locations.csv", index=False)

    imp = pd.DataFrame({"feature": F.FEATURE_COLUMNS,
                        "importance": np.asarray(model.get_feature_importance(), float)}
                       ).sort_values("importance", ascending=False)
    imp.to_csv(reports_dir / "feature_importance.csv", index=False)

    log("[6/7] Writing plots and evaluation report ...")
    _plots(parts["test"]["label"], pred["test"], imp, ranked, reports_dir)

    log("[7/7] Saving model artifacts ...")
    version = f"{datetime.now(timezone.utc):%Y%m%d%H%M%S}-{_fingerprint(data_dir, params)}"
    schema = P.build_schema(parts["train"])
    schema["model_version"] = version
    pre_cfg = {
        "model_version": version,
        "target_description": "Proxy priority score (0-100) of the 60 days after the prediction time.",
        "label_is_proxy": True,
        "proxy_config": C.PROXY_CONFIG,
        "snapshot_config": C.SNAPSHOT_CONFIG,
        "severity_weight": C.SEVERITY_WEIGHT,
        "road_condition_score": C.ROAD_CONDITION_SCORE,
        "pedestrian_activity_score": C.PEDESTRIAN_ACTIVITY_SCORE,
        "density_band_edges": C.DENSITY_BAND_EDGES,
        "level_quantiles": C.LEVEL_QUANTILES,
        "level_thresholds": thresholds,
        "score_clip_range": [0, 100],
        "data_start": str(ds.data_start),
        "default_as_of": str(as_of),
        "excluded_from_features": ["location_id", "latitude", "longitude",
                                   "population_density_per_km2 (replaced by coarse band)"],
    }
    split_info = {k: [str(c) for c in v] for k, v in split.items()}
    meta = {
        "model_version": version, "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_class": "catboost.CatBoostRegressor", "catboost_params": params,
        "best_iteration": best_iter, "early_stopping_rounds": C.EARLY_STOPPING_ROUNDS,
        "target": "proxy_priority_score", "label_is_proxy": True,
        "split_cutoffs": split_info, "n_features": len(F.FEATURE_COLUMNS),
        "feature_order": F.FEATURE_COLUMNS,
        "data_fingerprint_sha256_10": _fingerprint(data_dir, params),
        "quick_mode": bool(quick),
        "environment": {"python": platform.python_version(), "pandas": pd.__version__,
                        "numpy": np.__version__, "catboost": _ver("catboost"),
                        "scikit_learn": _ver("sklearn")},
        "disclaimer": ("Trained on proxy labels derived from synthetic data; it does not learn "
                       "actual municipal priorities."),
    }
    joblib.dump(model, artifacts_dir / ARTIFACT_FILES["model"])
    try:
        model.save_model(str(artifacts_dir / ARTIFACT_FILES["native"]), format="cbm")
    except Exception as exc:  # native copy is a convenience; joblib is the primary artifact
        log(f"      (native .cbm export skipped: {exc})")
    (artifacts_dir / ARTIFACT_FILES["schema"]).write_text(json.dumps(schema, indent=2))
    (artifacts_dir / ARTIFACT_FILES["preprocessing"]).write_text(json.dumps(pre_cfg, indent=2))
    (artifacts_dir / ARTIFACT_FILES["metrics"]).write_text(json.dumps(metrics, indent=2, default=float))
    (artifacts_dir / ARTIFACT_FILES["metadata"]).write_text(json.dumps(meta, indent=2))

    (reports_dir / "evaluation_report.md").write_text(
        _report_md(metrics, meta, thresholds, imp, ranked, split, ds, pre_cfg))
    log(f"Done. Model version {version}")
    return {"version": version, "metrics": metrics, "thresholds": thresholds}


def _ver(mod: str) -> str:
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "unknown")
    except Exception:
        return "not installed"


# ---------------------------------------------------------------------------
# Plots + report
# ---------------------------------------------------------------------------
def _plots(y_true, y_pred, imp, ranked, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].scatter(y_pred, y_true, s=6, alpha=0.25)
    lo, hi = 0, max(100.0, float(max(y_true.max(), y_pred.max())))
    ax[0].plot([lo, hi], [lo, hi], "r--", lw=1, label="perfect prediction")
    ax[0].set_xlabel("Predicted proxy priority score"); ax[0].set_ylabel("Actual proxy priority score")
    ax[0].set_title("Test split: actual vs predicted"); ax[0].legend()
    bins = pd.qcut(pd.Series(y_pred), 10, duplicates="drop")
    b = pd.DataFrame({"p": y_pred, "a": y_true}).groupby(bins, observed=True).mean()
    ax[1].plot(b["p"], b["a"], "o-", label="binned mean actual")
    ax[1].plot([b["p"].min(), b["p"].max()], [b["p"].min(), b["p"].max()], "r--", lw=1, label="ideal")
    ax[1].set_xlabel("Mean predicted (decile)"); ax[1].set_ylabel("Mean actual")
    ax[1].set_title("Calibration by predicted decile"); ax[1].legend()
    fig.tight_layout(); fig.savefig(out / "actual_vs_predicted.png", dpi=130); plt.close(fig)

    top = imp.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.barh(top["feature"], top["importance"]); ax.set_xlabel("CatBoost feature importance")
    ax.set_title("Top feature importances"); fig.tight_layout()
    fig.savefig(out / "feature_importance.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ranked["predicted_priority_score"], bins=30)
    ax.set_xlabel("Predicted priority score"); ax.set_ylabel("Locations")
    ax.set_title("Distribution of predicted scores (all locations)")
    fig.tight_layout(); fig.savefig(out / "score_distribution.png", dpi=130); plt.close(fig)


def _fmt(m): return f"{m['mae']:.2f} | {m['rmse']:.2f} | {m['r2']:.3f}"


def _verdict(metrics) -> List[str]:
    t = metrics["test"]
    m, base, pers = t["catboost"], t["baseline_train_mean"], t["baseline_persistence_prev_60d"]
    lines = []
    r2 = m["r2"]
    if r2 < 0.02:
        strength = "no meaningful predictive skill beyond simply predicting the average"
    elif r2 < 0.15:
        strength = "weak predictive skill"
    elif r2 < 0.40:
        strength = "moderate predictive skill"
    else:
        strength = "strong predictive skill"
    lines.append(f"On the held-out test period the model shows **{strength}** (R² = {r2:.3f}).")
    gain = (base["mae"] - m["mae"]) / base["mae"] * 100
    lines.append(f"Its MAE is {m['mae']:.2f} versus {base['mae']:.2f} for the train-mean baseline "
                 f"({gain:+.1f}% MAE improvement; negative = worse) and {pers['mae']:.2f} for the persistence baseline "
                 f"(previous 60-day proxy score).")
    top_key = [k for k in m if k.startswith("top") and k.endswith("_overlap")][0]
    lines.append(f"Top-5% overlap with the truly highest-scoring locations: model {m[top_key]*100:.1f}% "
                 f"(chance level 5%), persistence {pers[top_key]*100:.1f}%.")
    if m["mae"] >= pers["mae"]:
        lines.append("The model does **not** beat the persistence baseline on MAE, so a plain "
                     "formula over recent records is at least as good a forecast on this data.")
    return lines


def _report_md(metrics, meta, thresholds, imp, ranked, split, ds, pre_cfg) -> str:
    t, v, tr = metrics["test"], metrics["val"], metrics["train"]
    tbl = ["| Split | Model | MAE \\| RMSE \\| R² |", "|---|---|---|"]
    for name, blk in (("train", tr), ("validation", v), ("test", t)):
        tbl.append(f"| {name} | CatBoost | {_fmt(blk['catboost'])} |")
        tbl.append(f"| {name} | Train-mean baseline | {_fmt(blk['baseline_train_mean'])} |")
        tbl.append(f"| {name} | Persistence (prev. 60d proxy) | {_fmt(blk['baseline_persistence_prev_60d'])} |")
    top = ranked.head(15)[["rank", "location_id", "predicted_priority_score", "priority_level",
                           "incident_count", "complaint_count", "unresolved_incident_count",
                           "road_condition"]]
    top_md = top.to_markdown(index=False) if _has_tabulate() else "```\n" + top.to_string(index=False) + "\n```"
    imp_md = "\n".join(f"- `{r.feature}`: {r.importance:.2f}" for r in imp.head(10).itertuples())
    lv = ranked["priority_level"].value_counts().reindex(R_LEVELS, fill_value=0).to_dict()
    sp = "\n".join(f"- {k}: {[str(c.date()) for c in (c_[0], c_[-1])][0]} to {[str(c.date()) for c in (c_[0], c_[-1])][1]} "
                   f"({len(c_)} cutoffs)" for k, c_ in split.items())
    return f"""# Evaluation report - Where Should We Fix First?

Model version: `{meta['model_version']}`  |  Trained (UTC): {meta['trained_at_utc']}
{"**QUICK MODE - reduced iterations, not for production.**" if meta['quick_mode'] else ""}

## 1. What this model is (and is not)

* The supplied data contains **no genuine target** - no municipal decision, repair-urgency or
  official priority field. (`priority_reported` in the complaints file is a citizen-chosen flag,
  not an official decision.)
* The model is therefore trained on a **documented proxy priority score** (0-100) built from
  incident severity, injuries, unresolved incidents, complaints, open complaints and static
  location conditions (formula in the README and `preprocessing_config.json`).
* **This model does not learn real municipal priorities.** It forecasts the proxy score of the
  next {pre_cfg['snapshot_config']['horizon_days']} days from records available up to the
  prediction time.
* The data is synthetic (see `data/DATASET_README.txt`); results must not be used for real decisions.

## 2. Evaluation design

Point-in-time snapshots every {pre_cfg['snapshot_config']['step_days']} days. Features use only
records up to the cutoff (unresolved/open status reconstructed *at* the cutoff); the label is the
proxy score of the following {pre_cfg['snapshot_config']['horizon_days']} days. Splits are
time-ordered with a purge gap so label windows never overlap across splits:

{sp}

The test split was used only for the numbers below - not for label construction, tuning,
early stopping, or the priority-level thresholds.

## 3. Metrics

{chr(10).join(tbl)}

Rank-oriented metrics (per cutoff, averaged): test Spearman = {t['catboost']['spearman']:.3f}
(persistence {t['baseline_persistence_prev_60d']['spearman']:.3f}); validation Spearman =
{v['catboost']['spearman']:.3f}.

### Honest reading

{chr(10).join('* ' + l for l in _verdict(metrics))}

Context: the label's static location-condition component alone has a standard deviation of
{metrics['static_condition_component_std']:.1f} points, versus {metrics['label_stats_train']['std']:.1f}
for the full label. Event-driven parts of the label are largely random from one window to the
next in this dataset, which caps how well *any* model can forecast it. Do not read the model's
scores as stronger evidence than these metrics support.

## 4. Plots

* `actual_vs_predicted.png` - scatter and decile calibration on the test split
* `feature_importance.png` / `feature_importance.csv`
* `score_distribution.png`

Top features:

{imp_md}

## 5. Priority levels

Thresholds are quantiles of **predicted validation scores** (fixed at training time, so levels
do not shuffle when a new report arrives): Medium >= {thresholds['Medium']:.2f} (50th pct),
High >= {thresholds['High']:.2f} (80th pct), Critical >= {thresholds['Critical']:.2f} (95th pct),
otherwise Low. Levels are therefore *relative* triage bands. Current distribution: {lv}.
Displayed scores are clipped to 0-100.

## 6. Top-ranked locations (as of {pre_cfg['default_as_of']})

{top_md}

Full table: `ranked_locations.csv`. `supporting_evidence` is computed from the underlying
records; `recent_proxy_score_60d` is the plain formula applied to the last 60 days of records.

## 7. Limitations

* Labels are a proxy defined by the project authors, not ground truth. Weights and saturation
  constants are judgement calls (see README) and could reasonably differ.
* Synthetic data: incident counts show essentially no persistence per location, so forecasting
  skill beyond static conditions is expected to be low.
* About one year of history limits time-aware evaluation (few cutoffs, correlated windows).
* Rates are normalised per 30 days of history so snapshots at different dates are comparable;
  behaviour on much longer histories than seen in training is untested.
* Latitude/longitude, location IDs and exact population density are deliberately excluded to
  prevent location-identity shortcuts; spatial effects are therefore not modelled.
* The model is not retrained when reports arrive; retrain explicitly with `scripts/train.py`.
"""


R_LEVELS = ["Critical", "High", "Medium", "Low"]


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except Exception:
        return False
