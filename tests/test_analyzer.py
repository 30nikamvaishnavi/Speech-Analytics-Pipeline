"""Tests for GeminiAnalyticsClient and TranscriptAnalyzer using mocked Gemini API."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.analytics.analyzer import TranscriptAnalyzer
from app.analytics.gemini_client import GeminiAnalyticsClient
from app.config import AppConfig
from app.exceptions import (
    ConfigurationError,
    GeminiClientError,
    GeminiRateLimitError,
    GeminiResponseParseError,
    SchemaValidationError,
    SpeechAnalyticsError,
)


def _make_config(*, api_key: str | None = "test-secret-key-12345", retries: int = 2) -> AppConfig:
    return AppConfig(
        gemini_api_key=api_key,
        gemini_model="gemini-4-hydrogen",
        whisper_model="base",
        whisper_device="cpu",
        gemini_max_retries=retries,
        gemini_retry_base_delay=0.0,
    )


def _valid_gemini_payload() -> dict:
    return {
        "summary": "Meera agreed to send the updated proposal by Friday.",
        "sentiment": {
            "label": "positive",
            "confidence": 0.88,
            "explanation": "Speakers expressed agreement and confidence on next steps.",
        },
        "topics": [
            {
                "name": "Proposal Timeline",
                "description": "Discussion of proposal delivery schedule.",
                "supporting_quote": "Meera: I will send the updated proposal by Friday.",
            }
        ],
        "entities": [
            {
                "name": "Meera",
                "role": "Account Manager",
                "evidence_quote": "Meera: I will send the updated proposal by Friday.",
            }
        ],
        "action_items": [
            {
                "task": "Send the updated proposal",
                "owner": "Meera",
                "deadline": "Friday",
                "evidence_quote": "Meera: I will send the updated proposal by Friday.",
                "verification_status": "unverified",
            }
        ],
        "overall_confidence": 0.9,
    }


def test_valid_gemini_json_and_analyzer_success() -> None:
    transcript = "Meera: I will send the updated proposal by Friday."
    mock_genai = MagicMock()
    mock_genai.models.generate_content.return_value = SimpleNamespace(
        text=json.dumps(_valid_gemini_payload())
    )

    client = GeminiAnalyticsClient(_make_config(), genai_client=mock_genai)
    analyzer = TranscriptAnalyzer(gemini_client=client)
    result = analyzer.analyze(transcript, detected_language="en")

    assert result.schema_valid is True
    assert result.analytics.sentiment.label == "positive"
    assert result.analytics.action_items[0].verification_status == "verified"
    assert result.evidence_validation.verified_entities == 1
    assert result.evidence_validation.verified_action_items == 1
    mock_genai.models.generate_content.assert_called_once()


def test_malformed_json_raises_parse_error() -> None:
    mock_genai = MagicMock()
    mock_genai.models.generate_content.return_value = SimpleNamespace(
        text="{not valid json"
    )
    client = GeminiAnalyticsClient(_make_config(retries=0), genai_client=mock_genai)

    with pytest.raises(GeminiResponseParseError):
        client.analyze_transcript("Hello world")


def test_api_exception_raises_sanitized_gemini_error() -> None:
    secret_key = "AIzaTestSuperSecretKey999"
    mock_genai = MagicMock()
    mock_genai.models.generate_content.side_effect = RuntimeError(
        f"Upstream error using key {secret_key}"
    )
    client = GeminiAnalyticsClient(
        _make_config(api_key=secret_key, retries=0),
        genai_client=mock_genai,
    )

    with pytest.raises(GeminiClientError) as exc_info:
        client.analyze_transcript("Some transcript")

    assert secret_key not in str(exc_info.value)
    assert "***REDACTED***" in str(exc_info.value)


def test_rate_limit_exception_retries_and_raises_rate_limit_error() -> None:
    mock_genai = MagicMock()
    mock_genai.models.generate_content.side_effect = RuntimeError(
        "429 RESOURCE_EXHAUSTED: quota exceeded"
    )
    sleep_mock = MagicMock()
    client = GeminiAnalyticsClient(
        _make_config(retries=2),
        genai_client=mock_genai,
        sleep_fn=sleep_mock,
    )

    with pytest.raises(GeminiRateLimitError):
        client.analyze_transcript("Valid transcript content")

    assert mock_genai.models.generate_content.call_count == 3
    assert sleep_mock.call_count == 2


def test_pydantic_validation_failure_raises_schema_error() -> None:
    invalid_payload = _valid_gemini_payload()
    invalid_payload["sentiment"]["label"] = "invalid_sentiment"
    mock_genai = MagicMock()
    mock_genai.models.generate_content.return_value = SimpleNamespace(
        text=json.dumps(invalid_payload)
    )
    client = GeminiAnalyticsClient(_make_config(), genai_client=mock_genai)

    with pytest.raises(SchemaValidationError):
        client.analyze_transcript("Transcript text")


def test_empty_transcript_rejected_without_api_call() -> None:
    mock_genai = MagicMock()
    client = GeminiAnalyticsClient(_make_config(), genai_client=mock_genai)
    analyzer = TranscriptAnalyzer(gemini_client=client)

    with pytest.raises(SpeechAnalyticsError):
        analyzer.analyze("   ")

    mock_genai.models.generate_content.assert_not_called()


def test_missing_api_key_fails_clearly_without_injected_client() -> None:
    config = _make_config(api_key=None)
    client = GeminiAnalyticsClient(config)

    with pytest.raises(ConfigurationError):
        client.analyze_transcript("Transcript content")
