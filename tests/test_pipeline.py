"""Tests for end-to-end SpeechAnalyticsPipeline orchestration and WhisperTranscriber."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.analytics.analyzer import TranscriptAnalyzer
from app.analytics.schemas import ActionItem, PersonEntity, Sentiment, SpeechAnalytics
from app.config import AppConfig
from app.exceptions import (
    AudioValidationError,
    GeminiClientError,
    TranscriptionError,
)
from app.pipeline import SpeechAnalyticsPipeline
from app.transcription.whisper_transcriber import TranscriptionResult, WhisperTranscriber
from app.validation.evidence_checker import EvidenceChecker


def _sample_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        gemini_api_key="dummy-key-for-mocked-tests",
        gemini_model="gemini-4-hydrogen",
        whisper_model="base",
        whisper_device="cpu",
        output_dir=tmp_path / "results",
    )


def _sample_audio_file(tmp_path: Path, name: str = "call.wav") -> Path:
    audio_path = tmp_path / name
    audio_path.write_bytes(b"RIFFdemo-audio-bytes")
    return audio_path


def test_successful_pipeline_with_exports(tmp_path: Path) -> None:
    audio_path = _sample_audio_file(tmp_path)
    transcript = "Priya: I will send the compliance report tomorrow."

    mock_transcriber = MagicMock(spec=WhisperTranscriber)
    mock_transcriber.model_name = "base"
    mock_transcriber.requested_device = "cpu"
    mock_transcriber.transcribe.return_value = TranscriptionResult(
        audio_filename="call.wav",
        transcript=transcript,
        detected_language="en",
        model_name="base",
        device_used="cpu",
        fp16_used=False,
        processing_time_seconds=0.42,
        audio_duration_seconds=12.5,
        segments_count=1,
    )

    mock_gemini_client = MagicMock()
    mock_gemini_client.analyze_transcript.return_value = SpeechAnalytics(
        summary="Priya will send the compliance report tomorrow.",
        sentiment=Sentiment(label="neutral", confidence=0.8, explanation="Factual call."),
        topics=[],
        entities=[
            PersonEntity(
                name="Priya",
                evidence_quote="Priya: I will send the compliance report tomorrow.",
            )
        ],
        action_items=[
            ActionItem(
                task="Send the compliance report",
                owner="Priya",
                deadline="tomorrow",
                evidence_quote="Priya: I will send the compliance report tomorrow.",
            )
        ],
    )

    analyzer = TranscriptAnalyzer(
        gemini_client=mock_gemini_client,
        evidence_checker=EvidenceChecker(),
    )
    pipeline = SpeechAnalyticsPipeline(
        config=_sample_config(tmp_path),
        transcriber=mock_transcriber,
        analyzer=analyzer,
    )

    result = pipeline.run(audio_path, save_outputs=True, output_dir=tmp_path / "out")

    assert result.processing_status == "completed"
    assert result.transcription is not None
    assert result.analytics is not None
    assert result.evidence_validation is not None
    assert result.evidence_validation.verified_entities == 1
    assert result.evidence_validation.verified_action_items == 1
    assert (tmp_path / "out" / "speech_analytics_report.csv").exists()
    assert (tmp_path / "out" / "action_items.csv").exists()
    assert (tmp_path / "out" / "call_analytics.json").exists()


def test_transcription_failure_never_calls_gemini(tmp_path: Path) -> None:
    audio_path = _sample_audio_file(tmp_path)
    mock_transcriber = MagicMock(spec=WhisperTranscriber)
    mock_transcriber.model_name = "base"
    mock_transcriber.requested_device = "cpu"
    mock_transcriber.transcribe.side_effect = TranscriptionError("Corrupted audio stream")

    mock_analyzer = MagicMock(spec=TranscriptAnalyzer)
    pipeline = SpeechAnalyticsPipeline(
        config=_sample_config(tmp_path),
        transcriber=mock_transcriber,
        analyzer=mock_analyzer,
    )

    result = pipeline.run(audio_path)

    assert result.processing_status == "failed"
    assert result.execution_metadata.error_stage == "transcription"
    assert result.transcription is None
    assert result.analytics is None
    mock_analyzer.analyze.assert_not_called()


def test_empty_transcript_fails_and_skips_gemini(tmp_path: Path) -> None:
    audio_path = _sample_audio_file(tmp_path)
    mock_transcriber = MagicMock(spec=WhisperTranscriber)
    mock_transcriber.model_name = "base"
    mock_transcriber.requested_device = "cpu"
    mock_transcriber.transcribe.return_value = TranscriptionResult(
        audio_filename="call.wav",
        transcript="   ",
        detected_language="en",
        model_name="base",
        device_used="cpu",
        fp16_used=False,
        processing_time_seconds=0.1,
        audio_duration_seconds=2.0,
    )

    mock_analyzer = MagicMock(spec=TranscriptAnalyzer)
    pipeline = SpeechAnalyticsPipeline(
        config=_sample_config(tmp_path),
        transcriber=mock_transcriber,
        analyzer=mock_analyzer,
    )

    result = pipeline.run(audio_path)

    assert result.processing_status == "failed"
    assert result.execution_metadata.error_stage == "transcript_validation"
    assert result.analytics is None
    mock_analyzer.analyze.assert_not_called()


def test_gemini_failure_preserves_transcript_as_partial_failure(tmp_path: Path) -> None:
    audio_path = _sample_audio_file(tmp_path)
    mock_transcriber = MagicMock(spec=WhisperTranscriber)
    mock_transcriber.model_name = "base"
    mock_transcriber.requested_device = "cpu"
    mock_transcriber.transcribe.return_value = TranscriptionResult(
        audio_filename="call.wav",
        transcript="Valid local transcript.",
        detected_language="en",
        model_name="base",
        device_used="cpu",
        fp16_used=False,
        processing_time_seconds=0.2,
        audio_duration_seconds=5.0,
    )

    mock_gemini_client = MagicMock()
    mock_gemini_client.analyze_transcript.side_effect = GeminiClientError("API outage")
    analyzer = TranscriptAnalyzer(gemini_client=mock_gemini_client)

    pipeline = SpeechAnalyticsPipeline(
        config=_sample_config(tmp_path),
        transcriber=mock_transcriber,
        analyzer=analyzer,
    )

    result = pipeline.run(audio_path)

    assert result.processing_status == "partial_failure"
    assert result.execution_metadata.error_stage == "analytics"
    assert result.transcription is not None
    assert result.transcription.transcript == "Valid local transcript."
    assert result.analytics is None


def test_evidence_validation_failure_marks_partial_failure(tmp_path: Path) -> None:
    audio_path = _sample_audio_file(tmp_path)
    mock_transcriber = MagicMock(spec=WhisperTranscriber)
    mock_transcriber.model_name = "base"
    mock_transcriber.requested_device = "cpu"
    mock_transcriber.transcribe.return_value = TranscriptionResult(
        audio_filename="call.wav",
        transcript="Some valid transcript.",
        detected_language="en",
        model_name="base",
        device_used="cpu",
        fp16_used=False,
        processing_time_seconds=0.2,
        audio_duration_seconds=3.0,
    )

    mock_gemini_client = MagicMock()
    mock_gemini_client.analyze_transcript.return_value = SpeechAnalytics(
        summary="Summary",
        sentiment=Sentiment(label="neutral", explanation="Neutral"),
    )
    failing_checker = MagicMock(spec=EvidenceChecker)
    failing_checker.validate.side_effect = RuntimeError("Unexpected checker bug")

    analyzer = TranscriptAnalyzer(
        gemini_client=mock_gemini_client,
        evidence_checker=failing_checker,
    )
    pipeline = SpeechAnalyticsPipeline(
        config=_sample_config(tmp_path),
        transcriber=mock_transcriber,
        analyzer=analyzer,
    )

    result = pipeline.run(audio_path)
    assert result.processing_status == "partial_failure"
    assert result.execution_metadata.error_stage == "evidence_validation"


def test_whisper_transcriber_device_caching_and_validation(tmp_path: Path) -> None:
    WhisperTranscriber.clear_model_cache()
    audio_path = _sample_audio_file(tmp_path, "sample.wav")

    fake_model = MagicMock()
    fake_model.transcribe.return_value = {
        "text": "Hello from local Whisper.",
        "language": "en",
        "segments": [{"start": 0.0, "end": 3.5}],
    }
    loader = MagicMock(return_value=fake_model)

    transcriber = WhisperTranscriber(
        model_name="tiny",
        device="auto",
        model_loader=loader,
        cuda_available_fn=lambda: False,
        duration_inspector=lambda _p: 3.5,
        check_ffmpeg=False,
    )

    res1 = transcriber.transcribe(audio_path)
    res2 = transcriber.transcribe(audio_path)

    assert res1.device_used == "cpu"
    assert res1.fp16_used is False
    assert res1.transcript == "Hello from local Whisper."
    assert res2.transcript == "Hello from local Whisper."
    # Cached after first load
    loader.assert_called_once_with("tiny", "cpu")
    assert fake_model.transcribe.call_args.kwargs["fp16"] is False

    # Missing file and bad extension checks
    with pytest.raises(AudioValidationError):
        transcriber.validate_audio_file(tmp_path / "missing.wav")

    bad_ext = tmp_path / "notes.txt"
    bad_ext.write_text("not audio", encoding="utf-8")
    with pytest.raises(AudioValidationError):
        transcriber.validate_audio_file(bad_ext)
