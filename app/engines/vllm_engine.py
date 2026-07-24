"""vLLM engine for FunASR (Fun-ASR-Nano high-performance inference)."""

import asyncio
import time
from typing import AsyncGenerator, Optional

import numpy as np

from app.engines.base import ASREngine
from app.schemas import (
    TranscriptionResult,
    Segment,
    WordTimestamp,
    StreamResult,
    StreamSentence,
)
from app.config import Settings

from loguru import logger


class VLLMEngine(ASREngine):
    """ASR engine using vLLM for high-performance inference.

    Supports Fun-ASR-Nano model with:
    - PagedAttention + Continuous Batching
    - Tensor Parallel (multi-GPU)
    - CTC word-level timestamps
    - Speaker diarization (optional)
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._model = None
        self._streaming_engine = None
        self._loaded = False
        self._model_name = settings.model

    async def load(self) -> None:
        """Load the vLLM model."""
        try:
            from funasr.auto.auto_model_vllm import AutoModelVLLM
        except ImportError:
            raise ImportError(
                "vLLM engine requires vllm package. "
                "Install with: uv sync --extra vllm"
            )

        logger.info(
            f"Loading vLLM model '{self._model_name}' "
            f"(tensor_parallel={self._settings.tensor_parallel_size}, "
            f"gpu_mem={self._settings.gpu_memory_utilization})..."
        )
        t0 = time.time()

        # vLLM model config
        model_kwargs = {
            "model": "FunAudioLLM/Fun-ASR-Nano-2512",
            "hub": "ms",
            "tensor_parallel_size": self._settings.tensor_parallel_size,
            "gpu_memory_utilization": self._settings.gpu_memory_utilization,
        }
        if self._settings.max_model_len:
            model_kwargs["max_model_len"] = self._settings.max_model_len

        # Load in thread pool (vLLM init takes 60-90s)
        self._model = await asyncio.to_thread(AutoModelVLLM, **model_kwargs)

        elapsed = time.time() - t0
        logger.info(f"vLLM model loaded in {elapsed:.1f}s")
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        hotwords: Optional[str] = None,
        spk: bool = False,
        timestamp: bool = True,
    ) -> TranscriptionResult:
        """Transcribe audio file using vLLM engine."""
        if not self._loaded:
            raise RuntimeError("vLLM model not loaded")

        t0 = time.time()

        # Build generate kwargs - safe defaults for API stability
        generate_kwargs = {
            "inputs": audio_path,
            "language": language or "auto",
            "temperature": 0.0,
            "repetition_penalty": 1.0,  # Must be 1.0 for EmbedsPrompt
            "max_new_tokens": 200,
        }
        if hotwords:
            generate_kwargs["hotwords"] = hotwords.split(",")

        # Run inference in thread pool
        results = await asyncio.to_thread(self._model.generate, **generate_kwargs)

        processing_time = time.time() - t0

        # Parse results
        if not results:
            return TranscriptionResult(text="", duration=0.0, processing_time=processing_time)

        # Combine all segment results
        all_text_parts = []
        segments = []

        for res in results:
            text = res.get("text", "")
            all_text_parts.append(text)

            # Build segment with timestamps
            words = []
            if timestamp and "timestamp" in res:
                for ts in res["timestamp"]:
                    if len(ts) >= 3:
                        words.append(
                            WordTimestamp(
                                word=ts[0] if isinstance(ts[0], str) else "",
                                start=ts[1] / 1000.0 if len(ts) > 1 else 0.0,
                                end=ts[2] / 1000.0 if len(ts) > 2 else 0.0,
                            )
                        )

            segments.append(
                Segment(
                    text=text,
                    start=res.get("start", 0) / 1000.0,
                    end=res.get("end", 0) / 1000.0,
                    speaker=f"SPK{res.get('spk', 0)}" if spk and "spk" in res else None,
                    words=words,
                )
            )

        full_text = "".join(all_text_parts)

        # Get audio duration
        from app.audio_utils import get_audio_duration
        duration = get_audio_duration(audio_path)

        rtf = processing_time / duration if duration > 0 else 0.0

        return TranscriptionResult(
            text=full_text,
            segments=segments,
            duration=round(duration, 3),
            processing_time=round(processing_time, 3),
            rtf=round(rtf, 4),
        )

    async def transcribe_stream(
        self,
        audio_chunks: AsyncGenerator[bytes, None],
        language: Optional[str] = None,
        hotwords: Optional[str] = None,
    ) -> AsyncGenerator[StreamResult, None]:
        """Realtime streaming using FunASRNanoStreamingVLLM."""
        try:
            from funasr.models.fun_asr_nano.inference_vllm_streaming import (
                FunASRNanoStreamingVLLM,
            )
        except ImportError:
            raise ImportError(
                "Streaming vLLM requires funasr with vLLM support. "
                "Install with: uv sync --extra vllm"
            )

        # Initialize streaming engine
        if self._streaming_engine is None:
            logger.info("Loading vLLM streaming engine...")
            engine_kwargs = {
                "model": "FunAudioLLM/Fun-ASR-Nano-2512",
                "chunk_ms": self._settings.chunk_ms,
                "rollback_chars": self._settings.rollback_chars,
            }
            self._streaming_engine = await asyncio.to_thread(
                FunASRNanoStreamingVLLM.from_pretrained, **engine_kwargs
            )
            logger.info("vLLM streaming engine loaded")

        # Collect all audio first (streaming VLLM needs cumulative encoding)
        audio_buffer = bytearray()
        sentences: list[StreamSentence] = []

        async for chunk in audio_chunks:
            audio_buffer.extend(chunk)

            # Calculate current duration
            current_samples = len(audio_buffer) // 2
            current_duration_ms = current_samples / 16  # 16kHz -> ms

            # Only decode after minimum audio length (1.5s)
            if current_duration_ms < 1500:
                continue

            # Convert to numpy for streaming engine
            pcm = np.frombuffer(bytes(audio_buffer), dtype=np.int16)
            audio_float = pcm.astype(np.float32) / 32768.0

            # Run streaming generate
            try:
                gen_kwargs = {}
                if language:
                    gen_kwargs["language"] = language

                # Get latest result from streaming generator
                last_result = None
                for result in self._streaming_engine.streaming_generate(
                    audio_float, **gen_kwargs
                ):
                    last_result = result

                if last_result:
                    if last_result.get("is_final"):
                        text = last_result.get("text", "")
                        if text:
                            sentences.append(
                                StreamSentence(
                                    text=text,
                                    start=0,
                                    end=int(current_duration_ms),
                                )
                            )
                    else:
                        fixed_text = last_result.get("fixed_text", "")
                        yield StreamResult(
                            sentences=sentences,
                            partial=fixed_text,
                            partial_start_ms=0,
                            is_final=False,
                        )
            except Exception as e:
                logger.warning(f"Streaming decode error: {e}")

        # Final result
        yield StreamResult(sentences=sentences, partial="", is_final=True)

    @property
    def engine_name(self) -> str:
        return "vllm"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def available_models(self) -> list[str]:
        return ["fun-asr-nano"]
