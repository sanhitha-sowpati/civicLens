"""FastAPI integration.  Mount the router in the existing website backend:

    from priority_model.api import create_router
    app.include_router(create_router(service))

or run standalone:  uvicorn priority_model.api:create_app --factory   (paths from env vars, see create_app).
The frontend only sees plain JSON; it never needs to know about CatBoost.
"""
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .preprocessing import InvalidRecordError
from .service import (ModelNotReadyError, PriorityService, ReportNotFoundError,
                      ReportStateError, ReportValidationError)


class PredictRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(
        ..., description=("Either {'location_id': 'L0001'} to score a known location, or a dict of "
                          "aggregated model features / raw location fields for a new record."))


class ReportRequest(BaseModel):
    location_id: str
    incident_type: str
    severity: str
    injuries: int = 0
    timestamp: Optional[str] = None
    description: Optional[str] = None
    reporter_id: Optional[str] = None


class VerifyRequest(BaseModel):
    verified_by: str
    severity: Optional[str] = Field(None, description="Optional fixer correction of the severity.")
    injuries: Optional[int] = Field(None, description="Optional fixer correction of the injury count.")


class RejectRequest(BaseModel):
    rejected_by: str
    reason: Optional[str] = None


def create_router(service: PriorityService) -> APIRouter:
    router = APIRouter()

    def _422(exc):
        return HTTPException(status_code=422, detail={"errors": exc.errors})

    @router.get("/api/health")
    def health():
        return {"status": "ok", "service": "CivicLens API", "model_loaded": True}

    @router.get("/api/model/status")
    def model_status():
        return service.status()

    @router.post("/api/model/predict")
    def model_predict(req: PredictRequest):
        try:
            return service.predict(req.records)
        except InvalidRecordError as exc:
            raise _422(exc)

    @router.get("/api/priority-ranking")
    def priority_ranking(top: Optional[int] = Query(None, ge=0, le=5000),
                         level: Optional[str] = None, location_id: Optional[str] = None):
        try:
            return service.get_ranking(top=top, level=level, location_id=location_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail={"errors": [str(exc)]})

    @router.post("/api/priority-ranking/recalculate")
    def recalculate():
        """Re-runs inference for all locations. Does NOT retrain the model."""
        return service.recalculate()

    @router.post("/api/reports", status_code=201)
    def submit_report(req: ReportRequest):
        try:
            return service.submit_report(req.model_dump() if hasattr(req, "model_dump") else req.dict())
        except ReportValidationError as exc:
            raise _422(exc)

    @router.get("/api/reports")
    def list_reports(status: Optional[str] = None):
        """List persisted citizen reports; pending reports do not affect model features."""
        if status:
            st = status.strip().lower()
            if st not in {"pending", "verified", "rejected", "resolved"}:
                raise HTTPException(status_code=422, detail="status must be pending, verified, rejected, or resolved")
            return {"reports": service.store.list(st), "counts": service.store.counts()}
        return {"reports": service.store.list(), "counts": service.store.counts()}

    @router.get("/api/locations")
    def list_locations():
        """Location coordinates plus the authoritative model ranking."""
        loc = service.ds.locations.copy()
        ranking = service.get_ranking()["rows"]
        rank_df = __import__("pandas").DataFrame(ranking)
        merged = loc.merge(rank_df, on="location_id", how="left")
        return {"total": int(len(merged)), "locations": merged.astype(object).where(merged.notna(), None).to_dict("records")}

    @router.get("/api/summary")
    def summary():
        ranking = service.get_ranking()["rows"]
        reports = service.store.counts()
        inc = service._inc_p
        return {
            "total_locations": int(len(service.ds.locations)),
            "total_incidents": int(len(inc)),
            "total_complaints": int(len(service.ds.complaints)),
            "critical_locations": int(sum(1 for row in ranking if row["priority_level"] == "Critical")),
            "unresolved_incidents": int(inc["resolved"].eq(0).sum()),
            "citizen_reports": int(sum(reports.values())),
            "pending_reports": int(reports.get("pending", 0)),
            "verified_reports": int(reports.get("verified", 0)),
            "rejected_reports": int(reports.get("rejected", 0)),
            "model_version": service.bundle.version,
            "model_loaded": True,
            "label_is_proxy": True
        }

    @router.get("/api/incidents")
    def list_incidents(limit: int = Query(100, ge=1, le=1000)):
        base = service._inc_p.copy()
        base = base.sort_values("timestamp", ascending=False).head(limit)
        columns = [c for c in ["incident_id", "location_id", "timestamp", "incident_type",
                                "severity", "injuries", "resolved", "resolution_hours"] if c in base.columns]
        return {"total": int(len(service._inc_p)), "incidents": base[columns].astype(object).where(base[columns].notna(), None).to_dict("records")}

    @router.post("/api/reports/{report_id}/resolve")
    def resolve_report(report_id: str, req: VerifyRequest):
        try:
            return service.resolve_report(report_id, req.verified_by)
        except ReportNotFoundError:
            raise HTTPException(status_code=404, detail="Report not found")
        except ReportStateError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @router.get("/api/reports/{report_id}")
    def get_report(report_id: str):
        try:
            return service.get_report(report_id)
        except ReportNotFoundError:
            raise HTTPException(status_code=404, detail="Report not found")

    @router.post("/api/reports/{report_id}/verify")
    def verify_report(report_id: str, req: VerifyRequest):
        try:
            return service.verify_report(report_id, req.verified_by, req.severity, req.injuries)
        except ReportNotFoundError:
            raise HTTPException(status_code=404, detail="Report not found")
        except ReportStateError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except ReportValidationError as exc:
            raise _422(exc)

    @router.post("/api/reports/{report_id}/reject")
    def reject_report(report_id: str, req: RejectRequest):
        try:
            return service.reject_report(report_id, req.rejected_by, req.reason)
        except ReportNotFoundError:
            raise HTTPException(status_code=404, detail="Report not found")
        except ReportStateError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    return router


def create_app() -> FastAPI:
    """Standalone app. Paths come from env vars (defaults are relative to the project root)."""
    root = os.environ.get("PRIORITY_PROJECT_ROOT", os.getcwd())
    app = FastAPI(title="Where Should We Fix First? - priority model service")
    try:
        service = PriorityService(
            artifacts_dir=os.environ.get("PRIORITY_ARTIFACTS_DIR", os.path.join(root, "artifacts")),
            data_dir=os.environ.get("PRIORITY_DATA_DIR", os.path.join(root, "data")),
            runtime_dir=os.environ.get("PRIORITY_RUNTIME_DIR", os.path.join(root, "runtime")))
    except ModelNotReadyError as exc:
        @app.get("/api/model/status")
        def not_ready():
            raise HTTPException(status_code=503, detail=str(exc))
        return app
    app.include_router(create_router(service))
    return app

