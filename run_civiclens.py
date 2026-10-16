"""Run the integrated CivicLens web app: python run_civiclens.py"""
import os
from pathlib import Path
from fastapi.responses import FileResponse
from priority_model.api import create_app

PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("PRIORITY_PROJECT_ROOT", str(PROJECT_DIR))
os.environ.setdefault("PRIORITY_ARTIFACTS_DIR", str(PROJECT_DIR / "artifacts"))
os.environ.setdefault("PRIORITY_DATA_DIR", str(PROJECT_DIR / "data"))
os.environ.setdefault("PRIORITY_RUNTIME_DIR", str(PROJECT_DIR / "runtime"))

app = create_app()
WEB_DIR = PROJECT_DIR / "web"

@app.get("/", include_in_schema=False)
def homepage():
    return FileResponse(WEB_DIR / "index.html")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return {"ok": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
