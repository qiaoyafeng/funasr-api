"""Offline transcription routes: POST /asr, POST /v1/audio/transcriptions, WS /ws."""

import json
import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse

from app.schemas import (
    TranscriptionResult,
    OpenAITranscriptionVerbose,
    OpenAITranscriptionJson,
    OpenAISegment,
)
from app.audio_utils import save_upload_file, cleanup_temp_file, get_file_extension
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Engine instance will be set by main.py at startup
_engine = None


def set_engine(engine):
    """Set the ASR engine instance."""
    global _engine
    _engine = engine


def get_engine():
    """Get the ASR engine instance."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="ASR engine not initialized")
    return _engine


@router.post("/asr", response_model=TranscriptionResult)
async def transcribe_asr(
    file: UploadFile = File(...),
    language: Optional[str] = Form(default=None),
    hotwords: Optional[str] = Form(default=None),
    spk: bool = Form(default=False),
    timestamp: bool = Form(default=True),
):
    """FunASR native transcription endpoint.

    Full-featured interface supporting speaker diarization, timestamps, and hotwords.

    - **file**: Audio file (wav/mp3/flac/m4a/ogg)
    - **language**: Language hint ("中文", "English", etc.). None for auto-detect.
    - **hotwords**: Comma-separated hotwords for better recognition.
    - **spk**: Enable speaker diarization.
    - **timestamp**: Output word-level timestamps.
    """
    engine = get_engine()

    # Save uploaded file
    suffix = get_file_extension(file.filename)
    content = await file.read()
    tmp_path = save_upload_file(content, suffix)

    try:
        result = await engine.transcribe(
            audio_path=tmp_path,
            language=language or settings.language,
            hotwords=hotwords or settings.hotwords or None,
            spk=spk,
            timestamp=timestamp,
        )
        return result
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_temp_file(tmp_path)


@router.post("/v1/audio/transcriptions")
async def transcribe_openai(
    file: UploadFile = File(...),
    model: str = Form(default="sensevoice"),
    language: Optional[str] = Form(default=None),
    response_format: Optional[str] = Form(default="json"),
    timestamp_granularities: Optional[str] = Form(default="word"),
    spk: bool = Form(default=False),
):
    """OpenAI Whisper-compatible transcription endpoint.

    Drop-in replacement for OpenAI's /v1/audio/transcriptions.
    Works with any OpenAI SDK client.

    - **file**: Audio file (wav/mp3/flac/m4a/ogg)
    - **model**: Model name (sensevoice, paraformer, fun-asr-nano)
    - **language**: Language hint
    - **response_format**: "json", "text", or "verbose_json"
    - **timestamp_granularities**: "word" or "segment"
    - **spk**: Enable speaker diarization (FunASR extension)
    """
    engine = get_engine()

    # Save uploaded file
    suffix = get_file_extension(file.filename)
    content = await file.read()
    tmp_path = save_upload_file(content, suffix)

    try:
        result = await engine.transcribe(
            audio_path=tmp_path,
            language=language,
            hotwords=None,
            spk=spk,
            timestamp=(timestamp_granularities == "word"),
        )

        # Format response based on response_format
        if response_format == "text":
            return PlainTextResponse(content=result.text)

        elif response_format == "verbose_json":
            # Build OpenAI-compatible verbose response
            segments = []
            for i, seg in enumerate(result.segments):
                openai_seg = OpenAISegment(
                    id=i,
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    words=seg.words if timestamp_granularities == "word" else [],
                )
                segments.append(openai_seg)

            verbose = OpenAITranscriptionVerbose(
                task="transcribe",
                language=language or "auto",
                duration=result.duration,
                text=result.text,
                segments=segments,
            )
            return JSONResponse(content=verbose.model_dump())

        else:  # json (default)
            return JSONResponse(content={"text": result.text})

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_temp_file(tmp_path)


@router.websocket("/ws")
async def websocket_offline(websocket: WebSocket):
    """Offline WebSocket transcription endpoint.

    Protocol:
    - Client sends "START" to begin session
    - Client optionally sends "LANGUAGE:<lang>" and "HOTWORDS:<words>"
    - Client sends binary PCM16 16kHz mono audio data
    - Client sends "STOP" to request transcription
    - Server returns JSON result with sentences

    Example:
        await ws.send("START")
        await ws.recv()  # {"event": "started"}
        await ws.send("LANGUAGE:中文")
        await ws.recv()  # {"event": "language_set", "language": "中文"}
        await ws.send(pcm_bytes)
        await ws.send("STOP")
        result = await ws.recv()  # {"sentences": [...], "is_final": true}
    """
    await websocket.accept()
    engine = get_engine()

    audio_buffer = bytearray()
    language = settings.language
    hotwords = settings.hotwords or None
    started = False

    try:
        while True:
            message = await websocket.receive()

            if "text" in message:
                text = message["text"]

                if text == "START":
                    started = True
                    audio_buffer.clear()
                    await websocket.send_json({"event": "started"})

                elif text.startswith("LANGUAGE:"):
                    language = text[9:]
                    await websocket.send_json({"event": "language_set", "language": language})

                elif text.startswith("HOTWORDS:"):
                    hotwords = text[9:]
                    await websocket.send_json({"event": "hotwords_set", "hotwords": hotwords})

                elif text == "STOP":
                    if not started:
                        await websocket.send_json({"error": "Session not started"})
                        continue

                    # Process accumulated audio
                    if audio_buffer:
                        import numpy as np
                        import tempfile
                        import soundfile as sf

                        # Convert PCM to wav file
                        pcm = np.frombuffer(bytes(audio_buffer), dtype=np.int16)
                        audio_float = pcm.astype(np.float32) / 32768.0

                        # Save to temp wav file
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                            sf.write(tmp.name, audio_float, 16000)
                            tmp_path = tmp.name

                        try:
                            result = await engine.transcribe(
                                audio_path=tmp_path,
                                language=language,
                                hotwords=hotwords,
                                spk=settings.enable_spk,
                                timestamp=True,
                            )

                            # Build response
                            sentences = []
                            for seg in result.segments:
                                sentences.append({
                                    "text": seg.text,
                                    "start": int(seg.start * 1000),
                                    "end": int(seg.end * 1000),
                                    "spk": seg.speaker,
                                })

                            await websocket.send_json({
                                "sentences": sentences,
                                "is_final": True,
                                "duration_ms": int(result.duration * 1000),
                            })
                        finally:
                            cleanup_temp_file(tmp_path)
                    else:
                        await websocket.send_json({
                            "sentences": [],
                            "is_final": True,
                            "duration_ms": 0,
                        })

                    await websocket.send_json({"event": "stopped"})
                    started = False
                    audio_buffer.clear()

            elif "bytes" in message:
                # Binary audio data
                if started:
                    audio_buffer.extend(message["bytes"])

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
