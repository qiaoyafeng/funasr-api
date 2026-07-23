"""Configuration management for FunASR API service."""

import argparse
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Settings:
    """Application settings with CLI args and environment variable support."""

    host: str = "0.0.0.0"
    port: int = 8000
    engine: str = "auto"  # "vllm" or "auto"
    model: str = "sensevoice"
    device: str = "cuda"
    gpu_memory_utilization: float = 0.8
    tensor_parallel_size: int = 1
    max_model_len: Optional[int] = None
    enable_spk: bool = False
    language: Optional[str] = None
    hotwords: str = ""
    # Realtime specific
    partial_window_sec: float = 15.0
    decode_interval: float = 0.48
    chunk_ms: int = 720
    rollback_chars: int = 8

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables."""
        return cls(
            host=os.getenv("FUNASR_HOST", "0.0.0.0"),
            port=int(os.getenv("FUNASR_PORT", "8000")),
            engine=os.getenv("FUNASR_ENGINE", "auto"),
            model=os.getenv("FUNASR_MODEL", "sensevoice"),
            device=os.getenv("FUNASR_DEVICE", "cuda"),
            gpu_memory_utilization=float(os.getenv("FUNASR_GPU_MEMORY_UTILIZATION", "0.8")),
            tensor_parallel_size=int(os.getenv("FUNASR_TENSOR_PARALLEL_SIZE", "1")),
            max_model_len=int(os.getenv("FUNASR_MAX_MODEL_LEN", "0")) or None,
            enable_spk=os.getenv("FUNASR_ENABLE_SPK", "false").lower() == "true",
            language=os.getenv("FUNASR_LANGUAGE") or None,
            hotwords=os.getenv("FUNASR_HOTWORDS", ""),
            partial_window_sec=float(os.getenv("FUNASR_PARTIAL_WINDOW_SEC", "15.0")),
            decode_interval=float(os.getenv("FUNASR_DECODE_INTERVAL", "0.48")),
            chunk_ms=int(os.getenv("FUNASR_CHUNK_MS", "720")),
            rollback_chars=int(os.getenv("FUNASR_ROLLBACK_CHARS", "8")),
        )


def parse_args() -> Settings:
    """Parse command line arguments with environment variable fallback."""
    env_settings = Settings.from_env()

    parser = argparse.ArgumentParser(description="FunASR Unified API Server")
    parser.add_argument("--host", default=env_settings.host, help="Bind address")
    parser.add_argument("--port", type=int, default=env_settings.port, help="Bind port")
    parser.add_argument(
        "--engine",
        default=env_settings.engine,
        choices=["vllm", "auto"],
        help="Inference engine: vllm (GPU high-perf) or auto (PyTorch AutoModel)",
    )
    parser.add_argument("--model", default=env_settings.model, help="Model name to load")
    parser.add_argument(
        "--device", default=env_settings.device, help="Device: cuda, cpu, mps"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=env_settings.gpu_memory_utilization,
        help="vLLM GPU memory utilization (0-1)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=env_settings.tensor_parallel_size,
        help="vLLM tensor parallel size (multi-GPU)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=env_settings.max_model_len,
        help="vLLM max model length",
    )
    parser.add_argument(
        "--enable-spk",
        action="store_true",
        default=env_settings.enable_spk,
        help="Enable speaker diarization",
    )
    parser.add_argument("--language", default=env_settings.language, help="Default language")
    parser.add_argument("--hotwords", default=env_settings.hotwords, help="Default hotwords, comma separated")
    parser.add_argument(
        "--partial-window-sec",
        type=float,
        default=env_settings.partial_window_sec,
        help="Realtime partial preview window in seconds",
    )
    parser.add_argument(
        "--decode-interval",
        type=float,
        default=env_settings.decode_interval,
        help="Realtime decode interval in seconds",
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=env_settings.chunk_ms,
        help="Streaming chunk size in milliseconds",
    )
    parser.add_argument(
        "--rollback-chars",
        type=int,
        default=env_settings.rollback_chars,
        help="Streaming rollback characters for unfixed region",
    )

    args = parser.parse_args()

    return Settings(
        host=args.host,
        port=args.port,
        engine=args.engine,
        model=args.model,
        device=args.device,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        enable_spk=args.enable_spk,
        language=args.language,
        hotwords=args.hotwords,
        partial_window_sec=args.partial_window_sec,
        decode_interval=args.decode_interval,
        chunk_ms=args.chunk_ms,
        rollback_chars=args.rollback_chars,
    )


# Global settings instance
settings: Settings = Settings()
