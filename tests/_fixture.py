"""Shared, lazily-built fixture: trains a quick model once into a temp dir."""
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
_STATE = {}


def data_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DATA_DIR.glob("*.csv"))}


def get():
    """Returns dict(tmp, artifacts, reports, runtime, train_result, data_hashes_before)."""
    if not _STATE:
        from priority_model.training import train
        tmp = Path(tempfile.mkdtemp(prefix="priority_tests_"))
        before = data_hashes()
        result = train(DATA_DIR, tmp / "artifacts", tmp / "reports", quick=True, verbose=False)
        _STATE.update(tmp=tmp, artifacts=tmp / "artifacts", reports=tmp / "reports",
                      train_result=result, data_hashes_before=before)
    return _STATE


def new_service(clock=None):
    from priority_model.service import PriorityService
    st = get()
    runtime = Path(tempfile.mkdtemp(prefix="runtime_", dir=st["tmp"]))
    return PriorityService(st["artifacts"], DATA_DIR, runtime, clock=clock), runtime


def cleanup():
    if _STATE:
        shutil.rmtree(_STATE["tmp"], ignore_errors=True)
        _STATE.clear()
