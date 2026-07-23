"""FunASR Unified API Server - Main entry point."""

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from app.config import parse_args, settings as global_settings
from app.engines.base import ASREngine
from app.routers import offline, realtime, system

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Global engine instance
_engine: ASREngine | None = None


def create_engine() -> ASREngine:
    """Create ASR engine based on configuration."""
    if global_settings.engine == "vllm":
        from app.engines.vllm_engine import VLLMEngine
        return VLLMEngine(global_settings)
    else:
        from app.engines.automodel_engine import AutoModelEngine
        return AutoModelEngine(global_settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load model on startup."""
    global _engine

    logger.info("=" * 60)
    logger.info("FunASR Unified API Server")
    logger.info("=" * 60)
    logger.info(f"  Engine: {global_settings.engine}")
    logger.info(f"  Model:  {global_settings.model}")
    logger.info(f"  Device: {global_settings.device}")
    logger.info(f"  SPK:    {global_settings.enable_spk}")
    logger.info("=" * 60)

    # Create and load engine
    _engine = create_engine()

    # Set engine in routers
    offline.set_engine(_engine)
    realtime.set_engine(_engine)
    system.set_engine(_engine)

    # Load model
    logger.info("Loading ASR model...")
    await _engine.load()
    logger.info("ASR model loaded successfully!")

    yield

    # Cleanup
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="FunASR Unified API",
    description="Unified speech recognition API with offline and realtime support",
    version="0.1.0",
    lifespan=lifespan,
)

# Register routers
app.include_router(offline.router, tags=["Offline Transcription"])
app.include_router(realtime.router, tags=["Realtime Streaming"])
app.include_router(system.router, tags=["System"])


def main():
    """Main entry point for the server."""
    # Parse command line arguments and update global settings
    parsed_settings = parse_args()

    # Update global settings
    global_settings.host = parsed_settings.host
    global_settings.port = parsed_settings.port
    global_settings.engine = parsed_settings.engine
    global_settings.model = parsed_settings.model
    global_settings.device = parsed_settings.device
    global_settings.gpu_memory_utilization = parsed_settings.gpu_memory_utilization
    global_settings.tensor_parallel_size = parsed_settings.tensor_parallel_size
    global_settings.max_model_len = parsed_settings.max_model_len
    global_settings.enable_spk = parsed_settings.enable_spk
    global_settings.language = parsed_settings.language
    global_settings.hotwords = parsed_settings.hotwords
    global_settings.partial_window_sec = parsed_settings.partial_window_sec
    global_settings.decode_interval = parsed_settings.decode_interval
    global_settings.chunk_ms = parsed_settings.chunk_ms
    global_settings.rollback_chars = parsed_settings.rollback_chars

    logger.info(f"Starting server on http://{global_settings.host}:{global_settings.port}")
    logger.info(f"  Docs: http://{global_settings.host}:{global_settings.port}/docs")

    uvicorn.run(
        app,
        host=global_settings.host,
        port=global_settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
