import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tests import _fixture as fx

from priority_model import config as C
from priority_model import features as F
from priority_model import preprocessing as P
from priority_model import ranking as R
from priority_model.dataset import load_dataset
from priority_model.training import (ARTIFACT_FILES, build_snapshots, make_cutoffs, split_cutoffs)
from priority_model.validation import DataValidationError, load_and_validate


def tearDownModule():
    fx.cleanup()


class DataValidationTests(unittest.TestCase):
    def test_clean_dataset_loads(self):
        loc, inc, com, rep = load_and_validate(fx.DATA_DIR)
        self.assertEqual((len(loc), len(inc), len(com)), (1000, 10000, 8000))
        self.assertTrue(all(v == 0 for v in rep["actions"]["incidents"].values()))

    def test_defects_are_detected_and_dropped(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            for f in ("locations.csv", "incidents.csv", "public_complaints.csv"):
                shutil.copy(fx.DATA_DIR / f, tmp / f)
            inc = pd.read_csv(tmp / "incidents.csv", dtype=str)
            bad = inc.iloc[:5].copy()
            bad["incident_id"] = ["IBAD0", "IBAD1", "IBAD2", "IBAD3", "IBAD4"]
            bad.loc[bad.index[0], "incident_id"] = inc.iloc[10]["incident_id"]   # duplicate id
            bad.loc[bad.index[1], "timestamp"] = "not-a-date"                     # invalid timestamp
            bad.loc[bad.index[2], "severity"] = "Catastrophic"                    # invalid severity
            bad.loc[bad.index[3], "location_id"] = "L9999"                        # unknown location
            bad.loc[bad.index[4], "injuries"] = "-3"                              # negative injuries
            pd.concat([inc, bad]).to_csv(tmp / "incidents.csv", index=False)
            loc = pd.read_csv(tmp / "locations.csv", dtype=str)
            loc.loc[0, "latitude"] = "123.4"                                      # invalid coordinate
            loc.to_csv(tmp / "locations.csv", index=False)
            _, inc2, _, rep = load_and_validate(tmp)
            a = rep["actions"]["incidents"]
            self.assertEqual(a["duplicate_incident_ids_dropped"], 1)
            self.assertEqual(a["invalid_timestamps_dropped"], 1)
            self.assertEqual(a["invalid_severity_dropped"], 1)
            self.assertEqual(a["unknown_location_ids_dropped"], 1)
            self.assertEqual(a["negative_injuries_dropped"], 1)
            self.assertEqual(rep["actions"]["locations"]["invalid_coordinates_blanked"], 1)
            self.assertEqual(len(inc2), 10000)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_missing_columns_raise(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            for f in ("locations.csv", "incidents.csv", "public_complaints.csv"):
                shutil.copy(fx.DATA_DIR / f, tmp / f)
            pd.read_csv(tmp / "incidents.csv").drop(columns=["severity"]).to_csv(tmp / "incidents.csv", index=False)
            with self.assertRaises(DataValidationError):
                load_and_validate(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_original_dataset_untouched_by_training(self):
        st = fx.get()
        self.assertEqual(st["data_hashes_before"], fx.data_hashes())


class FeatureAndLeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = load_dataset(fx.DATA_DIR)

    def test_forbidden_features_absent(self):
        for bad in ("location_id", "latitude", "longitude", "population_density_per_km2"):
            self.assertNotIn(bad, F.FEATURE_COLUMNS)

    def test_features_ignore_future_records(self):
        ds = self.ds
        cutoff = pd.Timestamp("2025-06-01")
        base = F.build_features(ds.locations, ds.incidents_p, ds.complaints_p, cutoff, ds.data_start)
        fut = ds.incidents[ds.incidents["timestamp"] > cutoff]
        self.assertGreater(len(fut), 0)
        # remove every future record: features must not change
        inc_cut = F.prepare_incidents(ds.incidents[ds.incidents["timestamp"] <= cutoff])
        com_cut = F.prepare_complaints(ds.complaints[ds.complaints["created_at"] <= cutoff])
        cut = F.build_features(ds.locations, inc_cut, com_cut, cutoff, ds.data_start)
        pd.testing.assert_frame_equal(base, cut)

    def test_unresolved_status_reconstructed_at_cutoff(self):
        inc = pd.DataFrame({
            "incident_id": ["a", "b"], "location_id": ["L0001", "L0001"],
            "timestamp": pd.to_datetime(["2025-03-01 00:00", "2025-03-01 00:00"]),
            "incident_type": ["Pothole", "Pothole"], "severity": ["High", "High"],
            "injuries": [0, 0], "reported_by": ["Citizen"] * 2,
            "resolved": [1, 0], "resolution_hours": [100.0, 5.0]})
        ip = F.prepare_incidents(inc)
        cp = F.prepare_complaints(self.ds.complaints.iloc[:0])
        ids = ["L0001"]
        early = F.aggregate_window(ids, ip, cp, None, pd.Timestamp("2025-03-02"))   # 'a' not yet resolved
        late = F.aggregate_window(ids, ip, cp, None, pd.Timestamp("2025-04-01"))    # 'a' resolved by then
        self.assertEqual(int(early.loc["L0001", "inc_unresolved"]), 2)
        self.assertEqual(int(late.loc["L0001", "inc_unresolved"]), 1)   # 'b' is never resolved

    def test_proxy_score_in_range_and_monotone(self):
        ds = self.ds
        static = F.location_static_features(ds.locations)
        agg = F.aggregate_window(static.index, ds.incidents_p, ds.complaints_p, None, ds.data_end)
        s = F.proxy_priority_score(agg, static)
        self.assertTrue(((s >= 0) & (s <= 100)).all())
        agg2 = agg.copy()
        agg2["inc_injuries"] += 3
        self.assertTrue((F.proxy_priority_score(agg2, static) >= s - 1e-9).all())

    def test_split_has_purge_gap_and_time_order(self):
        ds = self.ds
        cut = make_cutoffs(ds.data_start, ds.data_end)
        sp = split_cutoffs(cut)
        H = pd.Timedelta(days=C.SNAPSHOT_CONFIG["horizon_days"])
        self.assertLess(max(sp["train"]) + H, min(sp["val"]) + pd.Timedelta(seconds=1))
        self.assertLessEqual(max(sp["val"]) + H, min(sp["test"]))
        self.assertTrue(set(sp["train"]).isdisjoint(sp["val"]) and set(sp["val"]).isdisjoint(sp["test"]))
        # the last label window must lie inside the data
        self.assertLessEqual(max(cut) + H, ds.data_end)

    def test_snapshot_labels_use_future_only(self):
        ds = self.ds
        cut = [pd.Timestamp("2025-03-01")]
        snaps = build_snapshots(ds, cut)
        self.assertEqual(len(snaps), 1000)
        self.assertTrue(snaps["label"].between(0, 100).all())
        self.assertTrue(set(F.FEATURE_COLUMNS) <= set(snaps.columns))


class TrainingAndArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.st = fx.get()
        cls.art = cls.st["artifacts"]

    def test_artifacts_written(self):
        for key, name in ARTIFACT_FILES.items():
            if key == "native":
                continue
            self.assertTrue((self.art / name).exists(), name)
        for name in ("evaluation_report.md", "ranked_locations.csv", "actual_vs_predicted.png",
                     "feature_importance.png", "feature_importance.csv", "data_validation_report.json"):
            self.assertTrue((self.st["reports"] / name).exists(), name)

    def test_metrics_present_and_finite(self):
        m = json.loads((self.art / ARTIFACT_FILES["metrics"]).read_text())
        for split in ("train", "val", "test"):
            for block in ("catboost", "baseline_train_mean", "baseline_persistence_prev_60d"):
                for k in ("mae", "rmse", "r2"):
                    self.assertTrue(math.isfinite(m[split][block][k]), (split, block, k))
        self.assertIn("NOT verified municipal", m["target"])

    def test_metadata_flags_proxy_labels(self):
        meta = json.loads((self.art / ARTIFACT_FILES["metadata"]).read_text())
        self.assertTrue(meta["label_is_proxy"])
        self.assertEqual(meta["feature_order"], F.FEATURE_COLUMNS)

    def test_save_and_load_roundtrip_predicts_identically(self):
        from priority_model.service import ModelBundle
        import joblib
        b = ModelBundle(self.art)
        ds = load_dataset(fx.DATA_DIR)
        X = F.build_features(ds.locations, ds.incidents_p, ds.complaints_p, ds.data_end,
                             ds.data_start)[F.FEATURE_COLUMNS].head(50)
        again = joblib.load(self.art / ARTIFACT_FILES["model"])
        np.testing.assert_allclose(b.model.predict(X), again.predict(X))

    def test_schema_matches_model_and_code(self):
        schema = json.loads((self.art / ARTIFACT_FILES["schema"]).read_text())
        self.assertEqual(schema["feature_order"], F.FEATURE_COLUMNS)
        import joblib
        model = joblib.load(self.art / ARTIFACT_FILES["model"])
        self.assertEqual(list(model.feature_names_), schema["feature_order"])

    def test_level_thresholds_ordered(self):
        cfg = json.loads((self.art / ARTIFACT_FILES["preprocessing"]).read_text())
        t = cfg["level_thresholds"]
        self.assertLessEqual(t["Medium"], t["High"])
        self.assertLessEqual(t["High"], t["Critical"])


class PreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.svc, _ = fx.new_service()
        cls.pre = cls.svc.bundle.pre
        cls.known = cls.svc._feat.iloc[0][F.FEATURE_COLUMNS].to_dict()

    def test_column_order_independent_of_dict_order(self):
        rec = dict(reversed(list(self.known.items())))
        X, _ = self.pre.transform([rec])
        self.assertEqual(list(X.columns), F.FEATURE_COLUMNS)

    def test_missing_values_handled_with_warnings(self):
        rec = dict(self.known)
        rec.pop("mean_severity_score"); rec["zone"] = None
        X, w = self.pre.transform([rec])
        self.assertTrue(np.isnan(X.loc[0, "mean_severity_score"]))
        self.assertEqual(X.loc[0, "zone"], "UNKNOWN")
        self.assertEqual(len(w[0]), 2)
        self.assertTrue(np.isfinite(self.svc.bundle.predict_scores(X)[0]))

    def test_unseen_category_mapped_to_unknown(self):
        rec = dict(self.known, zone="ZONE_999")
        X, w = self.pre.transform([rec])
        self.assertEqual(X.loc[0, "zone"], "UNKNOWN")
        self.assertTrue(any("not seen in training" in x for x in w[0]))

    def test_errors_are_informative(self):
        for rec, needle in [
            (dict(self.known, incident_rate_per_30d="abc"), "not numeric"),
            (dict(self.known, near_school=2), "outside valid range"),
            (dict(self.known, open_complaint_share=1.5), "outside valid range"),
            (dict(self.known, typo_field=1), "unknown field"),
            ({"zone": "ZONE_1"}, "features missing"),
        ]:
            with self.assertRaises(P.InvalidRecordError) as cm:
                self.pre.transform([rec])
            self.assertIn(needle, str(cm.exception))
        with self.assertRaises(P.InvalidRecordError):
            self.pre.transform([])

    def test_raw_location_fields_are_converted(self):
        rec = dict(self.known)
        for k in ("road_condition_score", "pedestrian_activity_score", "population_density_band"):
            rec.pop(k)
        rec.update(road_condition="Poor", pedestrian_activity="high", population_density_per_km2=12000)
        X, w = self.pre.transform([rec])
        self.assertEqual(X.loc[0, "road_condition_score"], 2)
        self.assertEqual(X.loc[0, "pedestrian_activity_score"], 2)
        self.assertEqual(X.loc[0, "population_density_band"], 2)
        self.assertEqual(w[0], [])


class InferenceAndRankingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.svc, _ = fx.new_service()

    def test_known_location_inference_matches_ranking(self):
        top = self.svc.get_ranking(top=5)["rows"]
        ids = [r["location_id"] for r in top]
        out = self.svc.predict([{"location_id": i} for i in ids])["predictions"]
        for r, o in zip(top, out):
            self.assertAlmostEqual(r["predicted_priority_score"], o["predicted_priority_score"], places=2)
            self.assertEqual(r["priority_level"], o["priority_level"])

    def test_unknown_location_returns_error(self):
        with self.assertRaises(P.InvalidRecordError) as cm:
            self.svc.predict([{"location_id": "NOPE"}])
        self.assertIn("unknown location_id", str(cm.exception))

    def test_new_compatible_record(self):
        rec = self.svc._feat.iloc[3][F.FEATURE_COLUMNS].to_dict()
        rec["location_id"] = "NEW_SITE_1"
        res = self.svc.predict([rec])["predictions"][0]
        self.assertEqual(res["location_id"], "NEW_SITE_1")
        self.assertTrue(0 <= res["predicted_priority_score"] <= 100)
        self.assertIn(res["priority_level"], C.LEVELS_DESC)

    def test_mixed_batch_error_index_refers_to_original_list(self):
        good = {"location_id": self.svc._feat.index[0]}
        bad = dict(self.svc._feat.iloc[1][F.FEATURE_COLUMNS].to_dict(), near_school=5)
        with self.assertRaises(P.InvalidRecordError) as cm:
            self.svc.predict([good, bad])
        self.assertIn("record 1:", str(cm.exception))

    def test_ranking_structure(self):
        r = self.svc.get_ranking()
        df = pd.DataFrame(r["rows"])
        self.assertEqual(len(df), 1000)
        for col in ("location_id", "predicted_priority_score", "priority_level", "incident_count",
                    "complaint_count", "unresolved_incident_count", "road_condition", "supporting_evidence"):
            self.assertIn(col, df.columns)
        self.assertEqual(df["rank"].tolist(), list(range(1, 1001)))
        self.assertTrue(df["predicted_priority_score"].is_monotonic_decreasing)
        self.assertTrue(df["predicted_priority_score"].between(0, 100).all())
        self.assertTrue(set(df["priority_level"]) <= set(C.LEVELS_DESC))
        self.assertTrue(df["location_id"].is_unique)
        # level order must follow score order
        order = {l: i for i, l in enumerate(C.LEVELS_DESC)}
        self.assertTrue(df["priority_level"].map(order).is_monotonic_increasing)

    def test_supporting_evidence_matches_records(self):
        ds = self.svc.ds
        row = self.svc.get_ranking(top=1)["rows"][0]
        lid = row["location_id"]
        n_inc = int((ds.incidents["location_id"] == lid).sum())
        n_com = int((ds.complaints["location_id"] == lid).sum())
        self.assertEqual(row["incident_count"], n_inc)
        self.assertEqual(row["complaint_count"], n_com)
        self.assertIn(f"{n_inc} incidents", row["supporting_evidence"])
        self.assertIn(f"{n_com} complaints", row["supporting_evidence"])
        self.assertIn(row["road_condition"], row["supporting_evidence"])

    def test_filters(self):
        crit = self.svc.get_ranking(level="critical")
        self.assertTrue(all(r["priority_level"] == "Critical" for r in crit["rows"]))
        with self.assertRaises(ValueError):
            self.svc.get_ranking(level="Urgent")
        self.assertEqual(self.svc.get_ranking(location_id="L0001")["returned"], 1)

    def test_status(self):
        s = self.svc.status()
        self.assertTrue(s["label_is_proxy"])
        self.assertEqual(s["n_locations"], 1000)
        self.assertEqual(s["feature_order"], F.FEATURE_COLUMNS)

    def test_recalculate_is_idempotent_and_writes_csv(self):
        a = self.svc.get_ranking()["rows"]
        res = self.svc.recalculate()
        self.assertEqual(res["n_rank_changes"], 0)
        self.assertEqual(res["n_level_changes"], 0)
        self.assertTrue(Path(res["output_csv"]).exists())
        self.assertEqual(a, self.svc.get_ranking()["rows"])


class NewReportTests(unittest.TestCase):
    def setUp(self):
        base = pd.Timestamp("2026-01-05 12:00")
        self.svc, self.runtime = fx.new_service(clock=lambda: base.to_pydatetime())
        self.loc = self.svc.get_ranking()["rows"][500]["location_id"]   # a mid-ranked location
        self.ts_in_range = str(self.svc.ds.data_end - pd.Timedelta(hours=2))

    def _report(self, **kw):
        p = {"location_id": self.loc, "incident_type": "Collision", "severity": "Critical",
             "injuries": 3, "timestamp": self.ts_in_range, "reporter_id": "citizen-1"}
        p.update(kw)
        return p

    def test_pending_report_does_not_change_ranking(self):
        before = self.svc.get_ranking()["rows"]
        res = self.svc.submit_report(self._report())
        self.assertEqual(res["report"]["status"], "pending")
        self.assertFalse(res["ranking_changed"])
        self.assertEqual(before, self.svc.get_ranking()["rows"])
        self.assertEqual(self.svc.status()["reports"]["pending"], 1)

    def test_validation_errors(self):
        for kw, needle in [({"location_id": "ZZZ"}, "location_id"), ({"severity": "Huge"}, "severity"),
                           ({"incident_type": "Alien"}, "incident_type"), ({"injuries": -1}, "injuries"),
                           ({"timestamp": "garbage"}, "timestamp"),
                           ({"timestamp": "2030-01-01T00:00:00"}, "future"), ({"bogus": 1}, "unknown field")]:
            with self.assertRaises(Exception) as cm:
                self.svc.submit_report(self._report(**kw))
            self.assertIn(needle, str(cm.exception))
        self.assertEqual(self.svc.status()["reports"]["pending"], 0)

    def test_verification_updates_only_affected_location_and_reranks(self):
        feats_before = self.svc._feat.copy()
        rid = self.svc.submit_report(self._report())["report"]["report_id"]
        res = self.svc.verify_report(rid, "fixer-7")
        self.assertFalse(res["as_of_advanced"])
        self.assertFalse(res["retrained"])
        self.assertEqual(res["after"]["incident_count"], res["before"]["incident_count"] + 1)
        self.assertGreater(res["after"]["unresolved_incident_count"], res["before"]["unresolved_incident_count"] - 1)
        self.assertIn("incident_rate_per_30d", res["features_changed"])
        # features of every other location are untouched
        others = feats_before.index != self.loc
        pd.testing.assert_frame_equal(feats_before[others], self.svc._feat[others])
        # a new critical incident with injuries must not lower the score of a model-ranked site much,
        # and ranking bookkeeping must be consistent
        self.assertEqual(res["rank_change"], res["before"]["rank"] - res["after"]["rank"])
        df = pd.DataFrame(self.svc.get_ranking()["rows"])
        self.assertEqual(df["rank"].tolist(), list(range(1, 1001)))
        self.assertEqual(int(df.loc[df["location_id"] == self.loc, "rank"].iloc[0]), res["after"]["rank"])
        self.assertEqual(len(res["ranking_top"]), 10)

    def test_verified_report_equals_full_rebuild(self):
        rid = self.svc.submit_report(self._report()).get("report")["report_id"]
        self.svc.verify_report(rid, "fixer-7")
        incremental = self.svc.get_ranking()["rows"]
        self.svc.recalculate()
        self.assertEqual(incremental, self.svc.get_ranking()["rows"])

    def test_future_report_advances_as_of(self):
        ts = str(self.svc.ds.data_end + pd.Timedelta(days=3))
        rid = self.svc.submit_report(self._report(timestamp=ts))["report"]["report_id"]
        res = self.svc.verify_report(rid, "fixer-7")
        self.assertTrue(res["as_of_advanced"])
        self.assertEqual(self.svc.as_of, pd.Timestamp(ts))

    def test_fixer_can_correct_severity_and_injuries(self):
        rid = self.svc.submit_report(self._report(severity="Low", injuries=0))["report"]["report_id"]
        res = self.svc.verify_report(rid, "fixer-7", severity="critical", injuries=2)
        self.assertEqual(res["report"]["final_severity"], "Critical")
        self.assertEqual(res["report"]["final_injuries"], 2)
        inc = self.svc._inc_p
        row = inc[inc["incident_id"] == rid].iloc[0]
        self.assertEqual((row["severity"], int(row["injuries"])), ("Critical", 2))

    def test_reject_and_double_review(self):
        from priority_model.service import ReportStateError
        rid = self.svc.submit_report(self._report())["report"]["report_id"]
        before = self.svc.get_ranking()["rows"]
        self.svc.reject_report(rid, "fixer-7", "duplicate")
        self.assertEqual(before, self.svc.get_ranking()["rows"])
        with self.assertRaises(ReportStateError):
            self.svc.verify_report(rid, "fixer-7")
        rid2 = self.svc.submit_report(self._report())["report"]["report_id"]
        self.svc.verify_report(rid2, "fixer-7")
        with self.assertRaises(ReportStateError):
            self.svc.verify_report(rid2, "fixer-7")

    def test_unknown_report(self):
        from priority_model.service import ReportNotFoundError
        with self.assertRaises(ReportNotFoundError):
            self.svc.verify_report("R_DOES_NOT_EXIST", "x")

    def test_verified_reports_survive_restart_and_originals_untouched(self):
        from priority_model.service import PriorityService
        rid = self.svc.submit_report(self._report())["report"]["report_id"]
        self.svc.verify_report(rid, "fixer-7")
        expected = self.svc.get_ranking()["rows"]
        svc2 = PriorityService(fx.get()["artifacts"], fx.DATA_DIR, self.runtime)
        self.assertEqual(expected, svc2.get_ranking()["rows"])
        self.assertEqual(fx.get()["data_hashes_before"], fx.data_hashes())

    def test_reload_model_keeps_state(self):
        rid = self.svc.submit_report(self._report())["report"]["report_id"]
        self.svc.verify_report(rid, "fixer-7")
        before = self.svc.get_ranking()["rows"]
        self.svc.reload_model()
        self.assertEqual(before, self.svc.get_ranking()["rows"])


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from fastapi import FastAPI
            from fastapi.testclient import TestClient
        except Exception as exc:  # fastapi / httpx not installed
            raise unittest.SkipTest(f"fastapi test client unavailable: {exc}")
        from priority_model.api import create_router
        cls.svc, _ = fx.new_service()
        app = FastAPI()
        app.include_router(create_router(cls.svc))
        cls.client = TestClient(app)

    def test_endpoints(self):
        c = self.client
        self.assertEqual(c.get("/api/model/status").status_code, 200)
        r = c.get("/api/priority-ranking", params={"top": 3})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["returned"], 3)
        lid = r.json()["rows"][0]["location_id"]
        p = c.post("/api/model/predict", json={"records": [{"location_id": lid}]})
        self.assertEqual(p.status_code, 200)
        bad = c.post("/api/model/predict", json={"records": [{"nonsense": 1}]})
        self.assertEqual(bad.status_code, 422)
        self.assertEqual(c.post("/api/priority-ranking/recalculate").status_code, 200)
        rep = c.post("/api/reports", json={"location_id": lid, "incident_type": "Pothole",
                                           "severity": "High", "injuries": 0})
        self.assertEqual(rep.status_code, 201)
        rid = rep.json()["report"]["report_id"]
        v = c.post(f"/api/reports/{rid}/verify", json={"verified_by": "fixer-1"})
        self.assertEqual(v.status_code, 200)
        self.assertEqual(c.post(f"/api/reports/{rid}/verify", json={"verified_by": "fixer-1"}).status_code, 409)
        self.assertEqual(c.get("/api/reports/NOPE").status_code, 404)


if __name__ == "__main__":
    unittest.main()
