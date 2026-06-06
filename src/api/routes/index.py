from fastapi import APIRouter

from .health import router as health_router
from .exam import router as exam_router
from .queue import router as queue_router

api_router = APIRouter()
api_router.include_router(health_router, prefix="/health", tags=["health"])
api_router.include_router(exam_router, prefix="/exams", tags=["exams"])
api_router.include_router(queue_router, prefix="/queue", tags=["queue"])