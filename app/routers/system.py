"""System routes: /health, /v1/models."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.schemas import HealthResponse, ModelListResponse, ModelInfo
from app.config import settings

from loguru import logger

router = APIRouter()

# Engine instance will be set by main.py at startup
_engine = None


def set_engine(engine):
    """Set the ASR engine instance."""
    global _engine
    _engine = engine


def get_engine():
    """Get the ASR engine instance."""
    return _engine


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint.

    Returns service status, engine type, device, and loaded models.
    """
    engine = get_engine()

    models_loaded = []
    models_available = []
    engine_name = settings.engine
    device = settings.device

    if engine and engine.is_loaded():
        models_loaded = [engine.model_name]
        models_available = engine.available_models
        engine_name = engine.engine_name

    return HealthResponse(
        status="ok" if engine and engine.is_loaded() else "loading",
        engine=engine_name,
        device=device,
        models_loaded=models_loaded,
        models_available=models_available,
    )


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    """List available models (OpenAI-compatible).

    Returns model list in OpenAI API format.
    """
    engine = get_engine()

    models = []
    if engine:
        for name in engine.available_models:
            models.append(
                ModelInfo(
                    id=name,
                    ready=engine.is_loaded() and name == engine.model_name,
                )
            )
    else:
        # Return default models when engine not loaded
        default_models = ["sensevoice", "paraformer", "paraformer-en", "fun-asr-nano"]
        for name in default_models:
            models.append(ModelInfo(id=name, ready=False))

    return ModelListResponse(data=models)
