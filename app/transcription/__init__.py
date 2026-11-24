"""Local audio transcription package powered by OpenAI Whisper."""

from app.transcription.whisper_transcriber import (
    SUPPORTED_AUDIO_EXTENSIONS,
    TranscriptionResult,
    WhisperTranscriber,
)

__all__ = [
    "SUPPORTED_AUDIO_EXTENSIONS",
    "TranscriptionResult",
    "WhisperTranscriber",
]
