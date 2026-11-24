"""Transcript analyzer orchestrating Gemini extraction, Pydantic validation, and evidence checks."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.analytics.gemini_client import GeminiAnalyticsClient
from app.analytics.schemas import SpeechAnalytics
from app.config import AppConfig
from app.exceptions import EvidenceValidationError, SpeechAnalyticsError
from app.validation.evidence_checker import EvidenceChecker, EvidenceValidationReport

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AnalysisResult:
    """Combined output from Gemini analytics and deterministic evidence verification."""

    analytics: SpeechAnalytics
    evidence_validation: EvidenceValidationReport
    schema_valid: bool = True


class TranscriptAnalyzer:
    """Analyzes transcripts via Gemini and validates outputs against transcript evidence."""

    def __init__(
        self,
        gemini_client: GeminiAnalyticsClient | None = None,
        evidence_checker: EvidenceChecker | None = None,
        config: AppConfig | None = None,
    ) -> None:
        if gemini_client is None:
            resolved_config = config or AppConfig.from_env()
            gemini_client = GeminiAnalyticsClient(resolved_config)
        self._gemini_client = gemini_client
        self._evidence_checker = evidence_checker or EvidenceChecker()

    def analyze(
        self,
        transcript: str,
        *,
        detected_language: str | None = None,
    ) -> AnalysisResult:
        """Analyze a non-empty transcript with Gemini and run deterministic evidence checks."""
        if not isinstance(transcript, str) or not transcript.strip():
            raise SpeechAnalyticsError(
                "Cannot run speech analytics on an empty transcript."
            )

        raw_analytics = self._gemini_client.analyze_transcript(
            transcript,
            detected_language=detected_language,
        )

        try:
            validated_analytics, evidence_report = self._evidence_checker.validate(
                raw_analytics,
                transcript,
                schema_valid=True,
            )
        except Exception as exc:
            raise EvidenceValidationError(
                f"Evidence verification failed: {exc}"
            ) from exc

        logger.info(
            "Analytics complete: entities=%d/%d verified, action_items=%d/%d verified",
            evidence_report.verified_entities,
            evidence_report.total_entities,
            evidence_report.verified_action_items,
            evidence_report.total_action_items,
        )

        return AnalysisResult(
            analytics=validated_analytics,
            evidence_validation=evidence_report,
            schema_valid=True,
        )
