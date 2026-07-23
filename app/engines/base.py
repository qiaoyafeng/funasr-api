"""Abstract base class for ASR engines."""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional

from app.schemas import TranscriptionResult, StreamResult


class ASREngine(ABC):
    """Abstract base class for ASR inference engines.

    All engines must implement transcribe (offline) and optionally
    transcribe_stream (realtime) methods.
    """

    @abstractmethod
    async def load(self) -> None:
        """Load the model into memory. Called at startup."""
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        ...

    @abstractmethod
    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        hotwords: Optional[str] = None,
        spk: bool = False,
        timestamp: bool = True,
    ) -> TranscriptionResult:
        """Transcribe an audio file (offline mode).

        Args:
            audio_path: Path to the audio file.
            language: Language hint (e.g., "中文", "English"). None for auto.
            hotwords: Comma-separated hotwords.
            spk: Whether to enable speaker diarization.
            timestamp: Whether to output word-level timestamps.

        Returns:
            TranscriptionResult with text, segments, and timing info.
        """
        ...

    async def transcribe_stream(
        self,
        audio_chunks: AsyncGenerator[bytes, None],
        language: Optional[str] = None,
        hotwords: Optional[str] = None,
    ) -> AsyncGenerator[StreamResult, None]:
        """Transcribe audio stream in realtime.

        Args:
            audio_chunks: Async generator yielding PCM16 16kHz mono chunks.
            language: Language hint.
            hotwords: Comma-separated hotwords.

        Yields:
            StreamResult with partial and final results.
        """
        raise NotImplementedError("Streaming not supported by this engine")
        yield  # Make it a generator

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Return the engine name."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the loaded model name."""
        ...

    @property
    def available_models(self) -> list[str]:
        """Return list of available model names."""
        return [self.model_name]
