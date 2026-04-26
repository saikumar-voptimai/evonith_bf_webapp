from fastapi import APIRouter

from app.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse()
