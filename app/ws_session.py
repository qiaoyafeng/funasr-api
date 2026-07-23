"""WebSocket session management for realtime streaming ASR."""

import asyncio
import logging
import time
from typing import Optional

import numpy as np

from app.schemas import StreamResult, StreamSentence
from app.config import Settings

logger = logging.getLogger(__name__)


class RealtimeASRSession:
    """Manages a single realtime ASR WebSocket session.

    Handles audio buffering, VAD-based sentence detection,
    and periodic partial result generation.
    """

    def __init__(self, engine, settings: Settings):
        self._engine = engine
        self._settings = settings
        self._audio_buffer = bytearray()
        self._sentences: list[StreamSentence] = []
        self._language: Optional[str] = settings.language
        self._hotwords: Optional[str] = settings.hotwords or None
        self._started = False
        self._last_decode_time = 0.0
        self._current_sentence_start_ms = 0

    @property
    def started(self) -> bool:
        return self._started

    def start(self):
        """Initialize session."""
        self._started = True
        self._audio_buffer.clear()
        self._sentences.clear()
        self._last_decode_time = time.time()

    def stop(self):
        """End session."""
        self._started = False

    def set_language(self, language: str):
        """Set language for this session."""
        self._language = language

    def set_hotwords(self, hotwords: str):
        """Set hotwords for this session."""
        self._hotwords = hotwords

    def add_audio(self, pcm_bytes: bytes):
        """Add audio chunk to buffer."""
        self._audio_buffer.extend(pcm_bytes)

    @property
    def audio_duration_ms(self) -> int:
        """Get current audio duration in milliseconds."""
        return len(self._audio_buffer) // 2 // 16  # bytes / 2 / 16 = ms

    def should_decode(self) -> bool:
        """Check if enough time has passed for next partial decode."""
        now = time.time()
        if now - self._last_decode_time >= self._settings.decode_interval:
            self._last_decode_time = now
            return True
        return False

    async def get_partial_result(self) -> Optional[StreamResult]:
        """Get partial transcription result for current audio."""
        if len(self._audio_buffer) < 1600 * 2 * 2:  # Min 2 seconds
            return None

        try:
            # Convert buffer to numpy
            pcm = np.frombuffer(bytes(self._audio_buffer), dtype=np.int16)
            audio_float = pcm.astype(np.float32) / 32768.0

            # Apply partial window limit
            max_samples = int(self._settings.partial_window_sec * 16000)
            if len(audio_float) > max_samples:
                audio_float = audio_float[-max_samples:]

            # Use engine's streaming capability
            # For simplicity, we do a single-shot decode of current buffer
            result = await self._decode_audio(audio_float, is_final=False)

            if result:
                return StreamResult(
                    sentences=self._sentences.copy(),
                    partial=result,
                    partial_start_ms=self._current_sentence_start_ms,
                    is_final=False,
                )
        except Exception as e:
            logger.warning(f"Partial decode error: {e}")

        return None

    async def get_final_result(self) -> StreamResult:
        """Get final transcription result."""
        if not self._audio_buffer:
            return StreamResult(sentences=self._sentences, partial="", is_final=True)

        try:
            pcm = np.frombuffer(bytes(self._audio_buffer), dtype=np.int16)
            audio_float = pcm.astype(np.float32) / 32768.0

            result = await self._decode_audio(audio_float, is_final=True)

            if result:
                duration_ms = self.audio_duration_ms
                self._sentences.append(
                    StreamSentence(
                        text=result,
                        start=self._current_sentence_start_ms,
                        end=duration_ms,
                    )
                )
        except Exception as e:
            logger.warning(f"Final decode error: {e}")

        return StreamResult(
            sentences=self._sentences,
            partial="",
            is_final=True,
        )

    async def _decode_audio(self, audio: np.ndarray, is_final: bool) -> Optional[str]:
        """Decode audio using the engine."""
        import asyncio
        import tempfile
        import soundfile as sf
        from app.audio_utils import cleanup_temp_file

        # Save to temp file for engine
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, 16000)
            tmp_path = tmp.name

        try:
            result = await self._engine.transcribe(
                audio_path=tmp_path,
                language=self._language,
                hotwords=self._hotwords,
                spk=False,
                timestamp=False,
            )
            return result.text if result.text else None
        finally:
            cleanup_temp_file(tmp_path)

    def reset_for_new_sentence(self):
        """Reset buffer for new sentence after VAD endpoint."""
        self._current_sentence_start_ms = self.audio_duration_ms
        # Keep buffer for context but mark sentence boundary
