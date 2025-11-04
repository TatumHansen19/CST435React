from fastapi import APIRouter
router = APIRouter(prefix="/api/p1", tags=["p1"])

@router.get("/health")
def health():
    return {"status": "ok", "project": "p1"}
