"""API routes package."""

from fastapi import APIRouter
from app.api.routes.documents import router as documents_router
from app.api.routes.health import router as health_router
from app.api.routes.query import router as query_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health_router)
api_router.include_router(query_router)
api_router.include_router(documents_router)

__all__ = ["api_router"]
