"""SQLite persistence for citizen reports (the original CSVs are never modified)."""
from __future__ import annotations

import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    report_id TEXT PRIMARY KEY,
    location_id TEXT NOT NULL,
    incident_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    injuries INTEGER NOT NULL DEFAULT 0,
    timestamp TEXT NOT NULL,
    description TEXT,
    reporter_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending',      -- pending | verified | rejected
    submitted_at TEXT NOT NULL,
    reviewed_by TEXT,
    reviewed_at TEXT,
    final_severity TEXT,
    final_injuries INTEGER,
    rejection_reason TEXT
);
"""


class ReportStore:
    def __init__(self, path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(SCHEMA)
            self._db.commit()

    def add(self, rec: Dict[str, Any], submitted_at: str) -> Dict[str, Any]:
        report_id = "R" + uuid.uuid4().hex[:10].upper()
        with self._lock:
            self._db.execute(
                "INSERT INTO reports (report_id, location_id, incident_type, severity, injuries,"
                " timestamp, description, reporter_id, status, submitted_at)"
                " VALUES (?,?,?,?,?,?,?,?, 'pending', ?)",
                (report_id, rec["location_id"], rec["incident_type"], rec["severity"],
                 int(rec["injuries"]), rec["timestamp"], rec.get("description"),
                 rec.get("reporter_id"), submitted_at))
            self._db.commit()
        return self.get(report_id)

    def get(self, report_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._db.execute("SELECT * FROM reports WHERE report_id=?", (report_id,)).fetchone()
        return dict(row) if row else None

    def list(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        q, args = "SELECT * FROM reports", ()
        if status:
            q, args = q + " WHERE status=?", (status,)
        with self._lock:
            return [dict(r) for r in self._db.execute(q + " ORDER BY submitted_at", args).fetchall()]

    def set_review(self, report_id: str, status: str, reviewer: str, reviewed_at: str,
                   final_severity: Optional[str] = None, final_injuries: Optional[int] = None,
                   reason: Optional[str] = None) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE reports SET status=?, reviewed_by=?, reviewed_at=?, final_severity=?,"
                " final_injuries=?, rejection_reason=? WHERE report_id=?",
                (status, reviewer, reviewed_at, final_severity, final_injuries, reason, report_id))
            self._db.commit()

    def counts(self) -> Dict[str, int]:
        with self._lock:
            rows = self._db.execute("SELECT status, COUNT(*) n FROM reports GROUP BY status").fetchall()
        out = {"pending": 0, "verified": 0, "rejected": 0, "resolved": 0}
        out.update({r["status"]: r["n"] for r in rows})
        return out

    def verified_incidents(self) -> pd.DataFrame:
        """Verified reports as rows shaped like incidents.csv (only these count as incidents)."""
        cols = ["incident_id", "location_id", "timestamp", "incident_type", "severity",
                "injuries", "reported_by", "resolved", "resolution_hours"]
        rows = [r for r in self.list() if r["status"] in {"verified", "resolved"}]
        if not rows:
            return pd.DataFrame({c: pd.Series(dtype="object") for c in cols}).astype(
                {"injuries": int, "resolved": int, "resolution_hours": float}
            ).assign(timestamp=pd.Series(dtype="datetime64[ns]"))
        df = pd.DataFrame({
            "incident_id": [r["report_id"] for r in rows],
            "location_id": [r["location_id"] for r in rows],
            "timestamp": pd.to_datetime([r["timestamp"] for r in rows]),
            "incident_type": [r["incident_type"] for r in rows],
            "severity": [r["final_severity"] or r["severity"] for r in rows],
            "injuries": [int(r["final_injuries"] if r["final_injuries"] is not None else r["injuries"]) for r in rows],
            "reported_by": "Citizen",
            "resolved": [1 if r["status"] == "resolved" else 0 for r in rows],
            "resolution_hours": [
                max(0.0, (pd.to_datetime(r["reviewed_at"]) - pd.to_datetime(r["timestamp"])).total_seconds() / 3600)
                if r["status"] == "resolved" and r.get("reviewed_at") else np.nan
                for r in rows
            ],
        })
        return df[cols]

    def close(self) -> None:
        with self._lock:
            self._db.close()
