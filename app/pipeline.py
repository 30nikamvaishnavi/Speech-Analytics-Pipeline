"""End-to-end orchestration for the two-stage Speech-Analytics-Pipeline."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app import __version__
from app.analytics.analyzer import TranscriptAnalyzer
from app.analytics.gemini_client import GeminiAnalyticsClient
from app.analytics.schemas import SpeechAnalytics
from app.config import AppConfig, sanitize_error_message
from app.exceptions import EvidenceValidationError, ExportError, SpeechAnalyticsError
from app.export.csv_exporter import export_all
from app.transcription.whisper_transcriber import TranscriptionResult, WhisperTranscriber
from app.validation.evidence_checker import EvidenceChecker, EvidenceValidationReport

logger = logging.getLogger(__name__)

ProcessingStatus = Literal["completed", "partial_failure", "failed"]


class ExecutionMetadata(BaseModel):
    """Non-secret execution metadata for auditing and reproducibility."""

    model_config = ConfigDict(extra="forbid")

    pipeline_version: str = __version__
    started_at_utc: str
    completed_at_utc: str
    total_duration_seconds: float = Field(..., ge=0.0)
    whisper_model: str
    whisper_device: str
    gemini_model: str
    stages_completed: list[str] = Field(default_factory=list)
    error_stage: str | None = None
    error_message: str | None = None
    saved_files: dict[str, str] = Field(default_factory=dict)


class PipelineResult(BaseModel):
    """Consolidated result returned by SpeechAnalyticsPipeline."""

    model_config = ConfigDict(extra="forbid")

    audio_filename: str
    transcription: TranscriptionResult | None = None
    analytics: SpeechAnalytics | None = None
    evidence_validation: EvidenceValidationReport | None = None
    execution_metadata: ExecutionMetadata
    processing_status: ProcessingStatus


class SpeechAnalyticsPipeline:
    """Central pipeline coordinating local Whisper transcription, Gemini analytics,
    Pydantic schema validation, deterministic evidence verification, and export.
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        transcriber: WhisperTranscriber | None = None,
        analyzer: TranscriptAnalyzer | None = None,
        evidence_checker: EvidenceChecker | None = None,
    ) -> None:
        self.config = config or AppConfig.from_env()
        self.transcriber = transcriber or WhisperTranscriber(
            model_name=self.config.whisper_model,
            device=self.config.whisper_device,
        )
        if analyzer is not None:
            self.analyzer = analyzer
        else:
            client = GeminiAnalyticsClient(self.config)
            self.analyzer = TranscriptAnalyzer(
                gemini_client=client,
                evidence_checker=evidence_checker or EvidenceChecker(),
                config=self.config,
            )

    def run(
        self,
        audio_path: str | Path,
        *,
        language: str | None = None,
        save_outputs: bool = False,
        output_dir: str | Path | None = None,
        progress_callback: Any | None = None,
    ) -> PipelineResult:
        """Execute the complete pipeline on an audio recording.

        Guarantees:
        - If transcription fails or produces an empty transcript, Gemini is NEVER called.
        - If Gemini fails, the local transcript is preserved and status is ``partial_failure``.
        - Never reports ``completed`` when any required stage fails.
        """
        path = Path(audio_path)
        started_at = datetime.now(timezone.utc).isoformat()
        start_clock = time.perf_counter()
        stages_completed: list[str] = []

        transcription: TranscriptionResult | None = None
        analytics: SpeechAnalytics | None = None
        evidence_validation: EvidenceValidationReport | None = None
        error_stage: str | None = None
        error_message: str | None = None
        saved_files: dict[str, str] = {}
        processing_status: ProcessingStatus = "failed"

        def _notify(stage: str, message: str) -> None:
            if callable(progress_callback):
                progress_callback(stage, message)

        # Stage 1 & 2: Validate audio file and transcribe locally with Whisper.
        try:
            _notify("transcription", f"Transcribing '{path.name}' locally with Whisper...")
            transcription = self.transcriber.transcribe(path, language=language)
            stages_completed.append("audio_validation")
            stages_completed.append("whisper_transcription")
        except Exception as exc:
            error_stage = "transcription"
            error_message = sanitize_error_message(str(exc), self.config.gemini_api_key)
            logger.error("Transcription stage failed: %s", error_message)
            return self._build_result(
                audio_filename=path.name,
                transcription=None,
                analytics=None,
                evidence_validation=None,
                started_at=started_at,
                start_clock=start_clock,
                stages_completed=stages_completed,
                error_stage=error_stage,
                error_message=error_message,
                saved_files=saved_files,
                processing_status="failed",
            )

        # Stage 3: Ensure transcript is non-empty before calling Gemini.
        if not transcription.transcript.strip():
            error_stage = "transcript_validation"
            error_message = (
                "Whisper produced an empty transcript; skipping Gemini analytics."
            )
            logger.warning(error_message)
            return self._build_result(
                audio_filename=path.name,
                transcription=transcription,
                analytics=None,
                evidence_validation=None,
                started_at=started_at,
                start_clock=start_clock,
                stages_completed=stages_completed,
                error_stage=error_stage,
                error_message=error_message,
                saved_files=saved_files,
                processing_status="failed",
            )
        stages_completed.append("transcript_validation")

        # Stage 4-7: Send transcript to Gemini, validate Pydantic schema, run evidence checks.
        try:
            _notify("analytics", "Running Gemini structured speech analytics & evidence checks...")
            analysis_result = self.analyzer.analyze(
                transcription.transcript,
                detected_language=transcription.detected_language,
            )
            analytics = analysis_result.analytics
            evidence_validation = analysis_result.evidence_validation
            stages_completed.extend(
                ["gemini_analytics", "pydantic_validation", "evidence_validation"]
            )
            processing_status = "completed"
        except EvidenceValidationError as exc:
            error_stage = "evidence_validation"
            error_message = sanitize_error_message(str(exc), self.config.gemini_api_key)
            processing_status = "partial_failure"
            logger.error("Evidence validation stage failed: %s", error_message)
        except SpeechAnalyticsError as exc:
            error_stage = "analytics"
            error_message = sanitize_error_message(str(exc), self.config.gemini_api_key)
            processing_status = "partial_failure"
            logger.error("Analytics stage failed: %s", error_message)
        except Exception as exc:
            error_stage = "analytics"
            error_message = sanitize_error_message(str(exc), self.config.gemini_api_key)
            processing_status = "partial_failure"
            logger.error("Unexpected analytics error: %s", error_message)

        result = self._build_result(
            audio_filename=path.name,
            transcription=transcription,
            analytics=analytics,
            evidence_validation=evidence_validation,
            started_at=started_at,
            start_clock=start_clock,
            stages_completed=stages_completed,
            error_stage=error_stage,
            error_message=error_message,
            saved_files=saved_files,
            processing_status=processing_status,
        )

        # Stage 8-9: Persist JSON and CSV outputs when requested.
        if save_outputs:
            target_dir = Path(output_dir) if output_dir is not None else self.config.output_dir
            try:
                _notify("export", f"Saving JSON and CSV reports to '{target_dir}'...")
                exported = export_all(result, target_dir)
                saved_files = {k: str(v) for k, v in exported.items()}
                stages_completed.append("export")
                result.execution_metadata.saved_files = saved_files
                result.execution_metadata.stages_completed = stages_completed
                # Re-write JSON so saved_files metadata is also captured if desired.
                export_all(result, target_dir)
            except ExportError as exc:
                result.execution_metadata.error_stage = result.execution_metadata.error_stage or "export"
                result.execution_metadata.error_message = str(exc)
                if result.processing_status == "completed":
                    result.processing_status = "partial_failure"

        return result

    def _build_result(
        self,
        *,
        audio_filename: str,
        transcription: TranscriptionResult | None,
        analytics: SpeechAnalytics | None,
        evidence_validation: EvidenceValidationReport | None,
        started_at: str,
        start_clock: float,
        stages_completed: list[str],
        error_stage: str | None,
        error_message: str | None,
        saved_files: dict[str, str],
        processing_status: ProcessingStatus,
    ) -> PipelineResult:
        completed_at = datetime.now(timezone.utc).isoformat()
        total_elapsed = round(max(0.0, time.perf_counter() - start_clock), 4)
        metadata = ExecutionMetadata(
            started_at_utc=started_at,
            completed_at_utc=completed_at,
            total_duration_seconds=total_elapsed,
            whisper_model=(
                transcription.model_name if transcription else self.transcriber.model_name
            ),
            whisper_device=(
                transcription.device_used
                if transcription
                else self.transcriber.requested_device
            ),
            gemini_model=self.config.gemini_model,
            stages_completed=stages_completed,
            error_stage=error_stage,
            error_message=error_message,
            saved_files=saved_files,
        )
        return PipelineResult(
            audio_filename=audio_filename,
            transcription=transcription,
            analytics=analytics,
            evidence_validation=evidence_validation,
            execution_metadata=metadata,
            processing_status=processing_status,
        )
