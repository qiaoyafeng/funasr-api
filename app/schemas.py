"""Pydantic request/response models for FunASR API."""

from typing import Optional
from pydantic import BaseModel, Field


# --- Offline Transcription Response Models ---


class WordTimestamp(BaseModel):
    """Word-level timestamp."""

    word: str
    start: float
    end: float


class Segment(BaseModel):
    """A transcription segment with timing and optional speaker."""

    text: str
    start: float
    end: float
    speaker: Optional[str] = None
    words: list[WordTimestamp] = Field(default_factory=list)


class TranscriptionResult(BaseModel):
    """Full transcription result from POST /asr."""

    text: str
    segments: list[Segment] = Field(default_factory=list)
    duration: float = 0.0
    processing_time: float = 0.0
    rtf: float = 0.0


# --- OpenAI-compatible Response Models ---


class OpenAISegment(BaseModel):
    """OpenAI-compatible segment."""

    id: int = 0
    start: float = 0.0
    end: float = 0.0
    text: str = ""
    words: list[WordTimestamp] = Field(default_factory=list)


class OpenAITranscriptionVerbose(BaseModel):
    """OpenAI verbose_json response format."""

    task: str = "transcribe"
    language: str = "auto"
    duration: float = 0.0
    text: str = ""
    segments: list[OpenAISegment] = Field(default_factory=list)


class OpenAITranscriptionJson(BaseModel):
    """OpenAI json response format."""

    text: str = ""


# --- Streaming Response Models ---


class StreamSentence(BaseModel):
    """A confirmed sentence in streaming output."""

    text: str
    start: int  # milliseconds
    end: int  # milliseconds
    spk: Optional[str] = None


class StreamResult(BaseModel):
    """Streaming recognition result."""

    sentences: list[StreamSentence] = Field(default_factory=list)
    partial: str = ""
    partial_start_ms: int = 0
    is_final: bool = False


# --- System Models ---


class ModelInfo(BaseModel):
    """Model information for /v1/models."""

    id: str
    object: str = "model"
    created: int = 1700000000
    owned_by: str = "funasr"
    ready: bool = False


class ModelListResponse(BaseModel):
    """OpenAI-compatible model list response."""

    object: str = "list"
    data: list[ModelInfo] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    engine: str = "auto"
    device: str = "cuda"
    models_loaded: list[str] = Field(default_factory=list)
    models_available: list[str] = Field(default_factory=list)
