"""Local OpenAI Whisper transcription service.

Audio is always processed locally on CPU or CUDA. Audio files are never sent
to Gemini or any external transcription service.

Long-recording considerations:
    Whisper processes audio in 30-second sliding windows using FFmpeg decoding
    and does not silently truncate long recordings. However, loading very long
    recordings into float32/mel spectrograms increases system RAM (and VRAM on
    CUDA) and scales inference latency linearly with duration. For calls
    exceeding 30 minutes on CPU, consider using the ``tiny`` or ``base`` model.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.config import SUPPORTED_WHISPER_DEVICES, SUPPORTED_WHISPER_MODELS
from app.exceptions import AudioValidationError, TranscriptionError

logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_EXTENSIONS: frozenset[str] = frozenset({".wav", ".mp3", ".m4a", ".flac"})
LONG_RECORDING_THRESHOLD_SECONDS: float = 1800.0  # 30 minutes


class TranscriptionResult(BaseModel):
    """Structured output returned by local Whisper transcription."""

    model_config = ConfigDict(extra="forbid")

    audio_filename: str = Field(..., min_length=1)
    transcript: str
    detected_language: str = Field(default="unknown")
    model_name: str = Field(..., min_length=1)
    device_used: str = Field(..., min_length=1)
    fp16_used: bool
    processing_time_seconds: float = Field(..., ge=0.0)
    audio_duration_seconds: float | None = Field(default=None, ge=0.0)
    segments_count: int = Field(default=0, ge=0)
    long_recording_warning: str | None = None


class WhisperTranscriber:
    """Reusable, thread-safe local Whisper transcriber with model caching.

    Parameters
    ----------
    model_name:
        Whisper model size (``"tiny"``, ``"base"``, or ``"small"``).
    device:
        Execution device preference (``"auto"``, ``"cpu"``, or ``"cuda"``).
    model_loader:
        Optional dependency-injected callable ``(model_name, device) -> model``
        for unit testing without downloading model weights.
    cuda_available_fn:
        Optional dependency-injected callable returning whether CUDA is ready.
    duration_inspector:
        Optional callable ``(Path) -> float | None`` for inspecting duration.
    """

    _shared_model_cache: dict[tuple[str, str], Any] = {}
    _cache_lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        model_name: str = "base",
        device: str = "auto",
        *,
        model_loader: Callable[[str, str], Any] | None = None,
        cuda_available_fn: Callable[[], bool] | None = None,
        duration_inspector: Callable[[Path], float | None] | None = None,
        check_ffmpeg: bool = True,
    ) -> None:
        normalized_model = model_name.strip().lower()
        if normalized_model not in SUPPORTED_WHISPER_MODELS:
            raise TranscriptionError(
                f"Unsupported Whisper model '{model_name}'. "
                f"Supported models: {', '.join(SUPPORTED_WHISPER_MODELS)}."
            )

        normalized_device = device.strip().lower()
        if normalized_device not in SUPPORTED_WHISPER_DEVICES:
            raise TranscriptionError(
                f"Unsupported Whisper device '{device}'. "
                f"Supported devices: {', '.join(SUPPORTED_WHISPER_DEVICES)}."
            )

        self.model_name = normalized_model
        self.requested_device = normalized_device
        self._model_loader = model_loader
        self._cuda_available_fn = cuda_available_fn
        self._duration_inspector = duration_inspector
        self._check_ffmpeg = check_ffmpeg

    @classmethod
    def clear_model_cache(cls) -> None:
        """Clear all cached Whisper models from memory."""
        with cls._cache_lock:
            cls._shared_model_cache.clear()

    def _is_cuda_available(self) -> bool:
        if self._cuda_available_fn is not None:
            return bool(self._cuda_available_fn())
        try:
            import torch

            return bool(torch.cuda.is_available())
        except ImportError as exc:
            raise TranscriptionError(
                "PyTorch is required for local Whisper inference. "
                "Install dependencies from requirements.txt."
            ) from exc

    def resolve_device(self) -> tuple[str, bool]:
        """Determine execution device (``cpu`` or ``cuda``) and FP16 flag.

        FP16 is enabled only when running on CUDA; CPU inference always uses FP32.
        """
        cuda_ready = self._is_cuda_available()
        if self.requested_device == "auto":
            resolved = "cuda" if cuda_ready else "cpu"
        elif self.requested_device == "cuda":
            if not cuda_ready:
                raise TranscriptionError(
                    "CUDA execution was explicitly requested, but no CUDA GPU is available. "
                    "Use '--device cpu' or '--device auto'."
                )
            resolved = "cuda"
        else:
            resolved = "cpu"

        use_fp16 = resolved == "cuda"
        return resolved, use_fp16

    def validate_audio_file(self, audio_path: str | Path) -> Path:
        """Validate that the audio file exists, is non-empty, and has a supported format."""
        path = Path(audio_path)
        if not path.exists():
            raise AudioValidationError(f"Audio file does not exist: {path.name}")
        if not path.is_file():
            raise AudioValidationError(f"Audio path is not a regular file: {path.name}")
        if path.stat().st_size == 0:
            raise AudioValidationError(f"Audio file is empty (0 bytes): {path.name}")

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
            raise AudioValidationError(
                f"Unsupported audio format '{suffix or 'none'}'. Supported formats: {supported}."
            )

        if self._check_ffmpeg and shutil.which("ffmpeg") is None:
            raise TranscriptionError(
                "FFmpeg was not found on the system PATH. "
                "Install FFmpeg before running local Whisper transcription."
            )

        return path

    def inspect_audio_duration(self, audio_path: Path) -> float | None:
        """Inspect audio duration in seconds without loading the entire file when possible.

        Raises AudioValidationError if the file appears corrupted or unreadable.
        """
        if self._duration_inspector is not None:
            return self._duration_inspector(audio_path)

        suffix = audio_path.suffix.lower()
        if suffix in {".wav", ".flac"}:
            try:
                import soundfile as sf

                info = sf.info(str(audio_path))
                if info.frames <= 0 or info.samplerate <= 0:
                    raise AudioValidationError(
                        f"Audio file '{audio_path.name}' contains no playable frames."
                    )
                return float(info.duration)
            except AudioValidationError:
                raise
            except Exception as exc:
                raise AudioValidationError(
                    f"Audio file '{audio_path.name}' appears corrupted or unreadable: {exc}"
                ) from exc

        try:
            import librosa

            duration = float(librosa.get_duration(path=str(audio_path)))
            if duration <= 0:
                raise AudioValidationError(
                    f"Audio file '{audio_path.name}' has zero duration."
                )
            return duration
        except AudioValidationError:
            raise
        except ImportError:
            logger.warning(
                "librosa is unavailable; skipping pre-flight duration inspection for %s",
                audio_path.name,
            )
            return None
        except Exception as exc:
            raise AudioValidationError(
                f"Failed to read audio metadata for '{audio_path.name}': {exc}"
            ) from exc

    def _load_model(self, device: str) -> Any:
        """Load and cache the Whisper model for ``(self.model_name, device)``."""
        cache_key = (self.model_name, device)
        with self._cache_lock:
            if cache_key in self._shared_model_cache:
                logger.debug("Using cached Whisper model '%s' on %s", self.model_name, device)
                return self._shared_model_cache[cache_key]

        logger.info("Loading local Whisper model '%s' on device '%s'", self.model_name, device)
        if self._model_loader is not None:
            model = self._model_loader(self.model_name, device)
        else:
            try:
                import whisper
            except ImportError as exc:
                raise TranscriptionError(
                    "openai-whisper is not installed. Install dependencies from requirements.txt."
                ) from exc
            try:
                model = whisper.load_model(self.model_name, device=device)
            except Exception as exc:
                raise TranscriptionError(
                    f"Failed to load Whisper model '{self.model_name}' on {device}: {exc}"
                ) from exc

        with self._cache_lock:
            self._shared_model_cache[cache_key] = model
        return model

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe an audio file locally using OpenAI Whisper.

        Parameters
        ----------
        audio_path:
            Path to a WAV, MP3, M4A, or FLAC recording.
        language:
            Optional ISO language code (``None`` enables automatic detection).
        """
        validated_path = self.validate_audio_file(audio_path)
        duration_seconds = self.inspect_audio_duration(validated_path)
        device, use_fp16 = self.resolve_device()

        long_warning: str | None = None
        if duration_seconds is not None and duration_seconds > LONG_RECORDING_THRESHOLD_SECONDS:
            minutes = duration_seconds / 60.0
            long_warning = (
                f"Recording duration is {minutes:.1f} minutes. Whisper processes long audio "
                "without truncation via 30s sliding windows; expect higher RAM/VRAM usage and "
                "longer processing time."
            )
            logger.warning(long_warning)

        model = self._load_model(device)
        start_time = time.perf_counter()

        transcribe_kwargs: dict[str, Any] = {
            "fp16": use_fp16,
            "task": "transcribe",
        }
        if language:
            transcribe_kwargs["language"] = language

        try:
            raw_output = model.transcribe(str(validated_path), **transcribe_kwargs)
        except Exception as exc:
            raise TranscriptionError(
                f"Whisper failed while transcribing '{validated_path.name}': {exc}"
            ) from exc

        elapsed = max(0.0, time.perf_counter() - start_time)

        if not isinstance(raw_output, dict):
            raise TranscriptionError(
                "Unexpected Whisper output format; expected a dictionary containing 'text'."
            )

        transcript_text = str(raw_output.get("text") or "").strip()
        detected_language = str(raw_output.get("language") or "unknown").strip() or "unknown"
        segments = raw_output.get("segments")
        segments_count = len(segments) if isinstance(segments, list) else 0

        if duration_seconds is None and isinstance(segments, list) and segments:
            last_end = segments[-1].get("end") if isinstance(segments[-1], dict) else None
            if isinstance(last_end, (int, float)) and last_end >= 0:
                duration_seconds = float(last_end)

        logger.info(
            "Local Whisper transcription finished: file=%s language=%s model=%s device=%s time=%.2fs",
            validated_path.name,
            detected_language,
            self.model_name,
            device,
            elapsed,
        )

        return TranscriptionResult(
            audio_filename=validated_path.name,
            transcript=transcript_text,
            detected_language=detected_language,
            model_name=self.model_name,
            device_used=device,
            fp16_used=use_fp16,
            processing_time_seconds=round(elapsed, 4),
            audio_duration_seconds=(
                round(duration_seconds, 3) if duration_seconds is not None else None
            ),
            segments_count=segments_count,
            long_recording_warning=long_warning,
        )
