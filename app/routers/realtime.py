"""Realtime streaming WebSocket route: /ws/realtime."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.ws_session import RealtimeASRSession
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
    if _engine is None:
        raise RuntimeError("ASR engine not initialized")
    return _engine


@router.websocket("/ws/realtime")
async def websocket_realtime(websocket: WebSocket):
    """Realtime streaming ASR WebSocket endpoint.

    Protocol (compatible with FunASR serve_realtime_ws.py):

    Client -> Server:
    - "START": Initialize session
    - "LANGUAGE:<lang>": Set language (optional)
    - "HOTWORDS:<words>": Set hotwords (optional)
    - binary: PCM16 16kHz mono audio chunks (~100ms each)
    - "STOP": End session, get final result

    Server -> Client:
    - {"event": "started"}: Session initialized
    - {"sentences": [...], "partial": "...", "is_final": false}: Partial result
    - {"sentences": [...], "is_final": true}: Final result
    - {"event": "stopped"}: Session ended

    Example client:
        await ws.send("START")
        await ws.recv()  # {"event": "started"}

        # Stream audio chunks
        for chunk in audio_chunks:
            await ws.send(chunk)  # PCM16 bytes
            result = await ws.recv()  # Partial results

        await ws.send("STOP")
        final = await ws.recv()  # {"is_final": true, ...}
    """
    await websocket.accept()

    try:
        engine = get_engine()
    except RuntimeError as e:
        await websocket.send_json({"error": str(e)})
        await websocket.close()
        return

    session = RealtimeASRSession(engine, settings)

    try:
        while True:
            message = await websocket.receive()

            if "text" in message:
                text = message["text"]

                if text == "START":
                    session.start()
                    await websocket.send_json({"event": "started"})

                elif text.startswith("LANGUAGE:"):
                    session.set_language(text[9:])
                    await websocket.send_json({
                        "event": "language_set",
                        "language": text[9:],
                    })

                elif text.startswith("HOTWORDS:"):
                    session.set_hotwords(text[9:])
                    await websocket.send_json({
                        "event": "hotwords_set",
                        "hotwords": text[9:],
                    })

                elif text == "STOP":
                    if not session.started:
                        await websocket.send_json({"error": "Session not started"})
                        continue

                    # Get final result
                    final_result = await session.get_final_result()
                    await websocket.send_json(final_result.model_dump())
                    await websocket.send_json({"event": "stopped"})

                    session.stop()

            elif "bytes" in message:
                # Binary audio data
                if not session.started:
                    continue

                session.add_audio(message["bytes"])

                # Check if we should send partial result
                if session.should_decode():
                    partial_result = await session.get_partial_result()
                    if partial_result:
                        await websocket.send_json(partial_result.model_dump())

    except WebSocketDisconnect:
        logger.info("Realtime WebSocket client disconnected")
    except Exception as e:
        logger.error(f"Realtime WebSocket error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
