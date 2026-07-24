"""Audio processing utilities for FunASR API."""

import os
import tempfile
from typing import Optional

import numpy as np
import soundfile as sf

from loguru import logger

# Supported audio extensions
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm", ".pcm"}

# Target sample rate for ASR
TARGET_SR = 16000


def save_upload_file(content: bytes, suffix: str = ".wav") -> str:
    """Save uploaded file content to a temporary file.

    Args:
        content: Raw file bytes.
        suffix: File extension.

    Returns:
        Path to the temporary file.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(content)
    tmp.close()
    return tmp.name


def cleanup_temp_file(path: str) -> None:
    """Remove a temporary file safely."""
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except OSError as e:
        logger.warning(f"Failed to cleanup temp file {path}: {e}")


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds.

    Args:
        audio_path: Path to audio file.

    Returns:
        Duration in seconds.
    """
    try:
        info = sf.info(audio_path)
        return info.duration
    except Exception:
        return 0.0


def load_audio_pcm16(audio_path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    """Load audio file and convert to PCM16 mono at target sample rate.

    Args:
        audio_path: Path to audio file.
        target_sr: Target sample rate.

    Returns:
        PCM16 numpy array (int16).
    """
    audio, sr = sf.read(audio_path, dtype="float32")

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        try:
            import torchaudio
            import torch

            waveform = torch.from_numpy(audio).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            audio = resampler(waveform).squeeze(0).numpy()
        except ImportError:
            # Fallback: simple linear interpolation
            duration = len(audio) / sr
            target_len = int(duration * target_sr)
            indices = np.linspace(0, len(audio) - 1, target_len)
            audio = np.interp(indices, np.arange(len(audio)), audio)

    # Convert to PCM16
    pcm = (audio * 32768).astype(np.int16)
    return pcm


def pcm_bytes_to_array(pcm_bytes: bytes) -> np.ndarray:
    """Convert raw PCM16 bytes to numpy array.

    Args:
        pcm_bytes: Raw PCM16 16kHz mono bytes.

    Returns:
        Float32 numpy array normalized to [-1, 1].
    """
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def array_to_pcm_bytes(audio: np.ndarray) -> bytes:
    """Convert float32 numpy array to PCM16 bytes.

    Args:
        audio: Float32 array normalized to [-1, 1].

    Returns:
        PCM16 bytes.
    """
    pcm = (audio * 32768).astype(np.int16)
    return pcm.tobytes()


def get_file_extension(filename: Optional[str]) -> str:
    """Extract file extension from filename.

    Args:
        filename: Original filename or None.

    Returns:
        File extension (e.g., ".wav").
    """
    if filename:
        ext = os.path.splitext(filename)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            return ext
    return ".wav"
