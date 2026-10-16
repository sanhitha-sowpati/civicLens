"""Framework-independent prediction service.

The website / FastAPI layer only talks to :class:`PriorityService`; nothing outside
this module needs to know about CatBoost.  Retraining is a separate, explicit
operation (``scripts/train.py``) followed by :meth:`PriorityService.reload_model`.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from . import config as C
from . import features as F
from . import preprocessing as P
from . import ranking as R
from .dataset import Dataset, load_dataset
from .store import ReportStore
from .training import ARTIFACT_FILES

KNOWN_ONLY_KEYS = {"location_id"}


class ModelNotReadyError(RuntimeError):
    """Artifacts are missing or inconsistent (run scripts/train.py)."""


class ReportValidationError(ValueError):
    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


class ReportNotFoundError(KeyError):
    pass


class ReportStateError(RuntimeError):
    pass


def _json_safe(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (pd.Timestamp, datetime)):
        return str(v)
    return v


def _records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return [{k: _json_safe(v) for k, v in r.items()} for r in df.to_dict("records")]


class ModelBundle:
    """Loaded, cross-checked artifacts."""

    def __init__(self, artifacts_dir):
        d = Path(artifacts_dir)
        missing = [f for k, f in ARTIFACT_FILES.items() if k != "native" and not (d / f).exists()]
        if missing:
            raise ModelNotReadyError(f"Missing artifact files in {d}: {missing}. Run scripts/train.py first.")
        self.model = joblib.load(d / ARTIFACT_FILES["model"])
        self.schema = json.loads((d / ARTIFACT_FILES["schema"]).read_text())
        self.config = json.loads((d / ARTIFACT_FILES["preprocessing"]).read_text())
        self.metrics = json.loads((d / ARTIFACT_FILES["metrics"]).read_text())
        self.metadata = json.loads((d / ARTIFACT_FILES["metadata"]).read_text())
        versions = {self.schema.get("model_version"), self.config.get("model_version"),
                    self.metadata.get("model_version")}
        if len(versions) != 1:
            raise ModelNotReadyError(f"Artifact versions disagree: {versions}")
        self.version = versions.pop()
        if self.schema["feature_order"] != F.FEATURE_COLUMNS:
            raise ModelNotReadyError("Saved feature schema differs from the code's feature definition; retrain.")
        names = getattr(self.model, "feature_names_", None)
        if names is not None and list(names) != self.schema["feature_order"]:
            raise ModelNotReadyError("Model feature names differ from feature_schema.json; retrain.")
        self.pre = P.FeaturePreprocessor(self.schema)
        self.thresholds = self.config["level_thresholds"]

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        return R.clip_scores(self.model.predict(X[self.schema["feature_order"]]))


class PriorityService:
    def __init__(self, artifacts_dir, data_dir, runtime_dir,
                 clock: Optional[Callable[[], datetime]] = None):
        self.artifacts_dir, self.data_dir, self.runtime_dir = Path(artifacts_dir), Path(data_dir), Path(runtime_dir)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self._clock = clock or (lambda: datetime.now(timezone.utc).replace(tzinfo=None))
        self._lock = threading.RLock()
        self.bundle = ModelBundle(self.artifacts_dir)
        self.ds: Dataset = load_dataset(self.data_dir)
        self.store = ReportStore(self.runtime_dir / "reports.db")
        self.last_recalculated: Optional[str] = None
        self._rebuild_all()

    # ------------------------------------------------------------------ state
    def _combined_incidents(self) -> pd.DataFrame:
        ver = self.store.verified_incidents()
        if len(ver) == 0:
            return self.ds.incidents_p
        return pd.concat([self.ds.incidents_p, F.prepare_incidents(ver)], ignore_index=True)

    def _compute_as_of(self, inc_p: pd.DataFrame) -> pd.Timestamp:
        return max(self.ds.data_end, inc_p["timestamp"].max())

    def _rank_from_features(self) -> None:
        self._scores = pd.Series(self.bundle.predict_scores(self._feat), index=self._feat.index)
        static = F.location_static_features(self.ds.locations)
        H = pd.Timedelta(days=C.SNAPSHOT_CONFIG["horizon_days"])
        recent = F.proxy_priority_score(
            F.aggregate_window(static.index, self._inc_p, self.ds.complaints_p, self.as_of - H, self.as_of), static)
        self._ranking = R.build_ranking(self.ds.locations, self._feat, self._scores.values,
                                        self.bundle.thresholds, self._inc_p, recent, self.as_of)

    def _rebuild_all(self) -> None:
        with self._lock:
            self._inc_p = self._combined_incidents()
            self.as_of = self._compute_as_of(self._inc_p)
            self._feat = F.build_features(self.ds.locations, self._inc_p, self.ds.complaints_p,
                                          self.as_of, self.ds.data_start)
            self._rank_from_features()

    def reload_model(self) -> Dict[str, Any]:
        """Pick up a newly trained model (after an explicit retrain) without restarting."""
        with self._lock:
            self.bundle = ModelBundle(self.artifacts_dir)
            self._rebuild_all()
            return self.status()

    # ----------------------------------------------------------------- status
    def status(self) -> Dict[str, Any]:
        b = self.bundle
        test = b.metrics.get("test", {})
        return {
            "model_loaded": True, "model_version": b.version,
            "model_class": b.metadata.get("model_class"),
            "trained_at_utc": b.metadata.get("trained_at_utc"),
            "target": "proxy_priority_score (0-100)", "label_is_proxy": True,
            "disclaimer": b.metadata.get("disclaimer"),
            "n_features": len(b.schema["feature_order"]),
            "feature_order": b.schema["feature_order"],
            "level_thresholds": b.thresholds,
            "test_metrics": test.get("catboost"),
            "test_baseline_train_mean": test.get("baseline_train_mean"),
            "n_locations": int(len(self._feat)),
            "as_of": str(self.as_of), "last_recalculated": self.last_recalculated,
            "reports": self.store.counts(),
            "quick_mode": b.metadata.get("quick_mode", False),
        }

    # ---------------------------------------------------------------- predict
    def predict(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict for known locations ({"location_id": ...}) or new aggregated records."""
        if not isinstance(records, list) or not records:
            raise P.InvalidRecordError(["'records' must be a non-empty list."])
        with self._lock:
            errors: List[str] = []
            frames: List[Optional[pd.DataFrame]] = [None] * len(records)
            warns: List[List[str]] = [[] for _ in records]
            feature_recs, feature_idx = [], []
            for i, r in enumerate(records):
                if isinstance(r, dict) and set(r) and set(r) <= KNOWN_ONLY_KEYS:
                    lid = r.get("location_id")
                    if lid not in self._feat.index:
                        errors.append(f"record {i}: unknown location_id {lid!r}")
                    else:
                        frames[i] = self._feat.loc[[lid], F.FEATURE_COLUMNS]
                else:
                    feature_recs.append(r); feature_idx.append(i)
            if feature_recs:
                try:
                    X, w = self.bundle.pre.transform(feature_recs)
                    for j, i in enumerate(feature_idx):
                        frames[i] = X.iloc[[j]]; warns[i] = w[j]
                except P.InvalidRecordError as exc:
                    # re-map record indexes (relative to feature_recs) to the original list
                    mapped = []
                    for e in exc.errors:
                        if e.startswith("record "):
                            n, rest = e[7:].split(":", 1)
                            if n.isdigit():
                                e = f"record {feature_idx[int(n)]}:{rest}"
                        mapped.append(e)
                    errors.extend(mapped)
            if errors:
                raise P.InvalidRecordError(errors)
            X_all = pd.concat(frames, ignore_index=True)[F.FEATURE_COLUMNS]
            scores = self.bundle.predict_scores(X_all)
            levels = R.priority_level(scores, self.bundle.thresholds)
            out = []
            for i, (s, lv) in enumerate(zip(scores, levels)):
                rec = records[i] if isinstance(records[i], dict) else {}
                out.append({"index": i, "location_id": rec.get("location_id"),
                            "predicted_priority_score": round(float(s), 2),
                            "priority_level": str(lv), "warnings": warns[i]})
            return {"model_version": self.bundle.version, "as_of": str(self.as_of), "predictions": out}

    # ---------------------------------------------------------------- ranking
    def get_ranking(self, top: Optional[int] = None, level: Optional[str] = None,
                    location_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            df = self._ranking
            if level:
                lv = level.strip().title()
                if lv not in C.LEVELS_DESC:
                    raise ValueError(f"level must be one of {C.LEVELS_DESC}")
                df = df[df["priority_level"] == lv]
            if location_id:
                df = df[df["location_id"] == location_id]
            total = int(len(df))
            if top is not None:
                df = df.head(max(int(top), 0))
            return {"model_version": self.bundle.version, "as_of": str(self.as_of),
                    "label_is_proxy": True, "total": total, "returned": int(len(df)),
                    "level_thresholds": self.bundle.thresholds, "rows": _records(df)}

    def recalculate(self) -> Dict[str, Any]:
        """Re-run inference for every location from the current data + verified reports (no retraining)."""
        with self._lock:
            old = self._ranking.set_index("location_id")[["rank", "predicted_priority_score", "priority_level"]]
            self._rebuild_all()
            new = self._ranking.set_index("location_id")
            self.last_recalculated = self._clock().isoformat()
            self._ranking.to_csv(self.runtime_dir / "ranked_locations.csv", index=False)
            diff = new[["rank", "predicted_priority_score", "priority_level"]].join(old, rsuffix="_before")
            moved = diff[(diff["rank"] != diff["rank_before"])]
            lvl = diff[diff["priority_level"] != diff["priority_level_before"]]
            return {"model_version": self.bundle.version, "as_of": str(self.as_of),
                    "recalculated_at": self.last_recalculated, "n_locations": int(len(new)),
                    "n_rank_changes": int(len(moved)), "n_level_changes": int(len(lvl)),
                    "max_abs_score_change": float((diff["predicted_priority_score"]
                                                   - diff["predicted_priority_score_before"]).abs().max()),
                    "output_csv": str(self.runtime_dir / "ranked_locations.csv")}

    # ---------------------------------------------------------------- reports
    def _validate_report(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        errs: List[str] = []
        if not isinstance(payload, dict):
            raise ReportValidationError(["Report must be an object."])
        allowed = {"location_id", "incident_type", "severity", "injuries", "timestamp",
                   "description", "reporter_id"}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            errs.append(f"unknown field(s): {unknown}")
        lid = payload.get("location_id")
        if not lid or lid not in self._feat.index:
            errs.append(f"location_id {lid!r} is missing or unknown")
        types = self.ds.validation_report["vocabularies"]["incident_types"]
        itype = payload.get("incident_type")
        if itype not in types:
            errs.append(f"incident_type {itype!r} invalid; allowed: {types}")
        sev = str(payload.get("severity", "")).strip().title()
        if sev not in C.SEVERITY_SCORE:
            errs.append(f"severity {payload.get('severity')!r} invalid; allowed: {C.SEVERITY_ORDER}")
        inj = payload.get("injuries", 0)
        try:
            inj_i = int(inj)
            if inj_i != float(inj) or inj_i < 0 or inj_i > 100:
                raise ValueError
        except (TypeError, ValueError):
            errs.append(f"injuries {inj!r} must be an integer between 0 and 100")
            inj_i = 0
        now = self._clock()
        ts = payload.get("timestamp")
        if ts in (None, ""):
            ts_p = pd.Timestamp(now)
        else:
            ts_p = pd.to_datetime(ts, errors="coerce")
            if pd.isna(ts_p):
                errs.append(f"timestamp {ts!r} is not a valid datetime (use ISO format)")
            else:
                if ts_p.tzinfo is not None:
                    ts_p = ts_p.tz_convert("UTC").tz_localize(None)
                if ts_p > pd.Timestamp(now) + pd.Timedelta(days=1):
                    errs.append("timestamp is in the future")
        desc = payload.get("description")
        if desc is not None and len(str(desc)) > 1000:
            errs.append("description longer than 1000 characters")
        if errs:
            raise ReportValidationError(errs)
        return {"location_id": lid, "incident_type": itype, "severity": sev, "injuries": inj_i,
                "timestamp": str(ts_p), "description": desc, "reporter_id": payload.get("reporter_id")}

    def submit_report(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and persist a citizen report as *pending*. It does NOT affect the ranking
        until a fixer verifies it."""
        with self._lock:
            rec = self._validate_report(payload)
            saved = self.store.add(rec, submitted_at=self._clock().isoformat())
            return {"report": saved, "ranking_changed": False,
                    "message": "Report saved as pending. It will count as an incident only after fixer verification."}

    def get_report(self, report_id: str) -> Dict[str, Any]:
        rep = self.store.get(report_id)
        if rep is None:
            raise ReportNotFoundError(report_id)
        return rep

    def reject_report(self, report_id: str, reviewed_by: str, reason: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            rep = self.get_report(report_id)
            if rep["status"] != "pending":
                raise ReportStateError(f"Report {report_id} is already {rep['status']}.")
            self.store.set_review(report_id, "rejected", reviewed_by or "unknown",
                                  self._clock().isoformat(), reason=reason)
            return {"report": self.store.get(report_id), "ranking_changed": False}

    def resolve_report(self, report_id: str, reviewed_by: str) -> Dict[str, Any]:
        """Mark a verified report resolved while keeping it in incident history."""
        with self._lock:
            rep = self.get_report(report_id)
            if rep["status"] != "verified":
                raise ReportStateError(f"Only verified reports can be resolved; {report_id} is {rep['status']}.")
            lid = rep["location_id"]
            before = self._snapshot(lid)
            self.store.set_review(report_id, "resolved", reviewed_by or "unknown",
                                  self._clock().isoformat(),
                                  final_severity=rep.get("final_severity") or rep["severity"],
                                  final_injuries=rep.get("final_injuries") if rep.get("final_injuries") is not None else int(rep["injuries"]))
            self._inc_p = self._combined_incidents()
            self.as_of = self._compute_as_of(self._inc_p)
            self._feat = F.build_features(self.ds.locations, self._inc_p, self.ds.complaints_p,
                                          self.as_of, self.ds.data_start)
            self._rank_from_features()
            after = self._snapshot(lid)
            self._ranking.to_csv(self.runtime_dir / "ranked_locations.csv", index=False)
            return {
                "report": self.store.get(report_id),
                "affected_location": lid,
                "before": before,
                "after": after,
                "score_change": round(after["predicted_priority_score"] - before["predicted_priority_score"], 2),
                "rank_change": before["rank"] - after["rank"],
                "model_version": self.bundle.version,
                "retrained": False,
            }

    def _snapshot(self, lid: str) -> Dict[str, Any]:
        row = self._ranking[self._ranking["location_id"] == lid].iloc[0]
        return {"rank": int(row["rank"]), "predicted_priority_score": float(row["predicted_priority_score"]),
                "priority_level": str(row["priority_level"]), "incident_count": int(row["incident_count"]),
                "unresolved_incident_count": int(row["unresolved_incident_count"]),
                "recent_proxy_score_60d": float(row["recent_proxy_score_60d"])}

    def verify_report(self, report_id: str, reviewed_by: str, severity: Optional[str] = None,
                      injuries: Optional[int] = None, top_n: int = 10) -> Dict[str, Any]:
        """Fixer verification: the report becomes a confirmed incident, the location's aggregate
        features are updated, its priority recalculated with the existing model (no retraining)
        and the ranking refreshed."""
        with self._lock:
            rep = self.get_report(report_id)
            if rep["status"] != "pending":
                raise ReportStateError(f"Report {report_id} is already {rep['status']}.")
            fsev = None
            if severity is not None:
                fsev = str(severity).strip().title()
                if fsev not in C.SEVERITY_SCORE:
                    raise ReportValidationError([f"severity {severity!r} invalid; allowed: {C.SEVERITY_ORDER}"])
            if injuries is not None and (int(injuries) < 0 or int(injuries) > 100):
                raise ReportValidationError(["injuries must be between 0 and 100"])
            lid = rep["location_id"]
            before = self._snapshot(lid)
            old_feat = self._feat.loc[lid, F.FEATURE_COLUMNS].copy()
            old_ranks = self._ranking.set_index("location_id")["rank"]
            old_levels = self._ranking.set_index("location_id")["priority_level"]

            self.store.set_review(report_id, "verified", reviewed_by or "unknown", self._clock().isoformat(),
                                  final_severity=fsev, final_injuries=None if injuries is None else int(injuries))
            self._inc_p = self._combined_incidents()
            new_as_of = self._compute_as_of(self._inc_p)
            advanced = new_as_of > self.as_of
            self.as_of = new_as_of
            if advanced:  # report newer than anything seen: every "days since"/recent window shifts
                self._feat = F.build_features(self.ds.locations, self._inc_p, self.ds.complaints_p,
                                              self.as_of, self.ds.data_start)
            else:         # only the affected location's aggregates change
                row = F.build_features(self.ds.locations, self._inc_p, self.ds.complaints_p,
                                       self.as_of, self.ds.data_start, loc_ids=[lid])
                for col in row.columns:
                    self._feat.at[lid, col] = row.at[lid, col]
            self._rank_from_features()

            after = self._snapshot(lid)
            new_feat = self._feat.loc[lid, F.FEATURE_COLUMNS]
            changed = {}
            for name in F.FEATURE_COLUMNS:
                a, b = old_feat[name], new_feat[name]
                if (isinstance(a, str) or isinstance(b, str)):
                    if a != b:
                        changed[name] = [a, b]
                elif not (pd.isna(a) and pd.isna(b)) and (pd.isna(a) != pd.isna(b) or abs(float(a) - float(b)) > 1e-9):
                    changed[name] = [_json_safe(a), _json_safe(b)]
            new_ranks = self._ranking.set_index("location_id")["rank"]
            new_levels = self._ranking.set_index("location_id")["priority_level"]
            level_changes = [{"location_id": k, "from": old_levels[k], "to": new_levels[k]}
                             for k in new_levels.index if new_levels[k] != old_levels[k]]
            shifted = int((new_ranks != old_ranks.reindex(new_ranks.index)).sum())
            self._ranking.to_csv(self.runtime_dir / "ranked_locations.csv", index=False)
            return {
                "report": self.store.get(report_id),
                "affected_location": lid,
                "before": before, "after": after,
                "score_change": round(after["predicted_priority_score"] - before["predicted_priority_score"], 2),
                "rank_change": before["rank"] - after["rank"],  # positive = moved up (more urgent)
                "level_changed": before["priority_level"] != after["priority_level"],
                "features_changed": changed,
                "locations_with_rank_shift": shifted,
                "level_changes": level_changes,
                "as_of_advanced": bool(advanced), "as_of": str(self.as_of),
                "model_version": self.bundle.version,
                "retrained": False,
                "ranking_top": _records(self._ranking.head(top_n)),
                "affected_location_row": _records(self._ranking[self._ranking["location_id"] == lid])[0],
            }
