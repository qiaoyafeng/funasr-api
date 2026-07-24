"""PyTorch AutoModel engine for FunASR (SenseVoice/Paraformer)."""

import asyncio
import re
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

# Model configurations for AutoModel
MODEL_CONFIGS = {
    "sensevoice": {
        "model": "iic/SenseVoiceSmall",
        "vad_model": "fsmn-vad",
        "vad_kwargs": {"max_single_segment_time": 30000},
    },
    "paraformer": {
        "model": "paraformer-zh",
        "vad_model": "fsmn-vad",
        "punc_model": "ct-punc",
    },
    "paraformer-en": {
        "model": "paraformer-en",
        "vad_model": "fsmn-vad",
    },
    "fun-asr-nano": {
        "model": "FunAudioLLM/Fun-ASR-Nano-2512",
        "hub": "hf",
        "trust_remote_code": True,
        "vad_model": "fsmn-vad",
        "vad_kwargs": {"max_single_segment_time": 30000},
    },
}

# Online model for streaming
ONLINE_MODEL_CONFIG = {
    "model": "paraformer-zh-streaming",
    "model_revision": "v2.0.4",
}


def clean_text(text: str) -> str:
    """Remove SenseVoice special tags from output."""
    return re.sub(r"<\|[^|]*\|>", "", text).strip()


class AutoModelEngine(ASREngine):
    """ASR engine using FunASR AutoModel (PyTorch backend).

    Supports SenseVoice, Paraformer, and Fun-ASR-Nano models.
    Works on both CPU and GPU.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._model = None
        self._online_model = None
        self._loaded = False
        self._model_name = settings.model

    async def load(self) -> None:
        """Load the ASR model."""
        from funasr import AutoModel

        model_name = self._model_name
        if model_name not in MODEL_CONFIGS:
            available = list(MODEL_CONFIGS.keys())
            raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

        cfg = MODEL_CONFIGS[model_name].copy()
        cfg["device"] = self._settings.device
        cfg["disable_update"] = True

        # Add speaker model if enabled
        if self._settings.enable_spk:
            cfg["spk_model"] = "iic/speech_eres2netv2_sv_zh-cn_16k-common"

        logger.info(f"Loading model '{model_name}' on {self._settings.device}...")
        t0 = time.time()

        # Run model loading in thread pool to avoid blocking
        self._model = await asyncio.to_thread(AutoModel, **cfg)

        elapsed = time.time() - t0
        logger.info(f"Model '{model_name}' loaded in {elapsed:.1f}s")
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
        """Transcribe audio file using AutoModel."""
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        t0 = time.time()

        # Build generate kwargs
        generate_kwargs = {"input": audio_path, "batch_size": 1}
        if language:
            generate_kwargs["language"] = language
        if hotwords:
            generate_kwargs["hotword"] = hotwords

        # Run inference in thread pool
        result = await asyncio.to_thread(self._model.generate, **generate_kwargs)

        processing_time = time.time() - t0

        # Parse results
        if not result:
            return TranscriptionResult(text="", duration=0.0, processing_time=processing_time)

        res = result[0]
        text = clean_text(res.get("text", ""))

        # Build segments
        segments = []
        if "sentence_info" in res:
            for seg in res["sentence_info"]:
                words = []
                if timestamp and "words" in seg:
                    for w in seg["words"]:
                        words.append(
                            WordTimestamp(
                                word=w.get("word", ""),
                                start=w.get("start", 0) / 1000.0,
                                end=w.get("end", 0) / 1000.0,
                            )
                        )
                segments.append(
                    Segment(
                        text=clean_text(seg.get("text", "")),
                        start=seg.get("start", 0) / 1000.0,
                        end=seg.get("end", 0) / 1000.0,
                        speaker=f"SPK{seg.get('spk', 0)}" if spk else None,
                        words=words,
                    )
                )

        # Get audio duration
        duration = res.get("audio_duration", 0.0)
        if duration == 0.0:
            from app.audio_utils import get_audio_duration
            duration = get_audio_duration(audio_path)

        rtf = processing_time / duration if duration > 0 else 0.0

        return TranscriptionResult(
            text=text,
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
        """Realtime streaming transcription using online model."""
        from funasr import AutoModel

        # Load online model if not loaded
        if self._online_model is None:
            logger.info("Loading online streaming model...")
            cfg = ONLINE_MODEL_CONFIG.copy()
            cfg["device"] = self._settings.device
            cfg["disable_update"] = True
            self._online_model = await asyncio.to_thread(AutoModel, **cfg)
            logger.info("Online streaming model loaded")

        # Accumulate audio and process
        audio_buffer = bytearray()
        sentences: list[StreamSentence] = []
        chunk_size = 1600 * 2  # 100ms at 16kHz, 2 bytes per sample
        decode_interval_samples = int(self._settings.decode_interval * 16000)
        samples_since_last_decode = 0

        async for chunk in audio_chunks:
            audio_buffer.extend(chunk)
            samples_since_last_decode += len(chunk) // 2

            # Decode periodically
            if samples_since_last_decode >= decode_interval_samples:
                samples_since_last_decode = 0

                # Convert buffer to numpy
                pcm = np.frombuffer(bytes(audio_buffer), dtype=np.int16)
                audio_float = pcm.astype(np.float32) / 32768.0

                # Run online model
                try:
                    result = await asyncio.to_thread(
                        self._online_model.generate,
                        input=audio_float,
                        is_final=False,
                        chunk_size=[5, 10, 5],
                    )
                    if result and result[0].get("text"):
                        partial_text = clean_text(result[0]["text"])
                        yield StreamResult(
                            sentences=sentences,
                            partial=partial_text,
                            partial_start_ms=0,
                            is_final=False,
                        )
                except Exception as e:
                    logger.warning(f"Streaming decode error: {e}")

        # Final decode
        if audio_buffer:
            pcm = np.frombuffer(bytes(audio_buffer), dtype=np.int16)
            audio_float = pcm.astype(np.float32) / 32768.0

            try:
                result = await asyncio.to_thread(
                    self._online_model.generate,
                    input=audio_float,
                    is_final=True,
                    chunk_size=[5, 10, 5],
                )
                if result and result[0].get("text"):
                    final_text = clean_text(result[0]["text"])
                    duration_ms = len(audio_buffer) // 2 // 16  # samples / 16 = ms
                    sentences.append(
                        StreamSentence(text=final_text, start=0, end=duration_ms)
                    )
            except Exception as e:
                logger.warning(f"Final decode error: {e}")

        yield StreamResult(sentences=sentences, partial="", is_final=True)

    @property
    def engine_name(self) -> str:
        return "auto"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def available_models(self) -> list[str]:
        return list(MODEL_CONFIGS.keys())
