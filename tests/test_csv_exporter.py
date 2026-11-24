"""Tests for Unicode-safe CSV and JSON exporters."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.analytics.schemas import ActionItem, PersonEntity, Sentiment, SpeechAnalytics, Topic
from app.export.csv_exporter import (
    ACTION_ITEMS_HEADERS,
    REPORT_HEADERS,
    export_action_items_csv,
    export_report_csv,
    export_result_json,
)
from app.pipeline import ExecutionMetadata, PipelineResult
from app.transcription.whisper_transcriber import TranscriptionResult
from app.validation.evidence_checker import EvidenceChecker


def _build_pipeline_result(
    *,
    transcript: str,
    entities: list[PersonEntity],
    action_items: list[ActionItem],
) -> PipelineResult:
    raw_analytics = SpeechAnalytics(
        summary='Customer asked about "invoice #42", discounts, and delivery.',
        sentiment=Sentiment(
            label="mixed",
            confidence=0.75,
            explanation="Initial concern resolved by clear commitment.",
        ),
        topics=[
            Topic(
                name="Billing, Taxes & Delivery",
                description="Reviewing comma-separated invoice details.",
                supporting_quote=transcript,
            )
        ],
        entities=entities,
        action_items=action_items,
        overall_confidence=0.82,
    )
    checker = EvidenceChecker()
    validated_analytics, ev_report = checker.validate(raw_analytics, transcript)

    return PipelineResult(
        audio_filename="call_record.wav",
        transcription=TranscriptionResult(
            audio_filename="call_record.wav",
            transcript=transcript,
            detected_language="hi",
            model_name="base",
            device_used="cpu",
            fp16_used=False,
            processing_time_seconds=1.25,
            audio_duration_seconds=25.0,
        ),
        analytics=validated_analytics,
        evidence_validation=ev_report,
        execution_metadata=ExecutionMetadata(
            started_at_utc="2026-01-01T00:00:00+00:00",
            completed_at_utc="2026-01-01T00:00:02+00:00",
            total_duration_seconds=2.0,
            whisper_model="base",
            whisper_device="cpu",
            gemini_model="gemini-4-hydrogen",
            stages_completed=["whisper_transcription", "gemini_analytics", "evidence_validation"],
        ),
        processing_status="completed",
    )


def test_csv_headers_escaping_unicode_and_verification_export(tmp_path: Path) -> None:
    transcript = (
        'रोहित शर्मा ने कहा, "स्नेहा कुलकर्णी, कृपया शुक्रवार तक अपडेटेड इनवॉइस पाठवा."'
    )
    result = _build_pipeline_result(
        transcript=transcript,
        entities=[
            PersonEntity(
                name="रोहित शर्मा",
                role="लीड",
                evidence_quote="रोहित शर्मा ने कहा",
            )
        ],
        action_items=[
            ActionItem(
                task='अपडेटेड इनवॉइस पाठवा, "ग्राहक कॉपी"',
                owner="स्नेहा कुलकर्णी",
                deadline="शुक्रवार",
                evidence_quote="स्नेहा कुलकर्णी, कृपया शुक्रवार तक अपडेटेड इनवॉइस पाठवा.",
            ),
            ActionItem(
                task="Hallucinated task",
                owner="Ghost",
                deadline="Next Year",
                evidence_quote="Not present anywhere.",
            ),
        ],
    )

    report_path = export_report_csv(result, tmp_path / "speech_analytics_report.csv")
    actions_path = export_action_items_csv(result, tmp_path / "action_items.csv")

    with report_path.open("r", encoding="utf-8", newline="") as f:
        report_reader = csv.DictReader(f)
        assert report_reader.fieldnames == REPORT_HEADERS
        report_rows = list(report_reader)
        assert len(report_rows) == 1
        row = report_rows[0]
        assert row["audio_filename"] == "call_record.wav"
        assert "रोहित शर्मा" in row["transcript"]
        assert '"invoice #42"' in row["summary"]
        assert row["processing_status"] == "completed"
        exported_actions = json.loads(row["action_items"])
        assert exported_actions[0]["verification_status"] == "verified"
        assert exported_actions[1]["verification_status"] == "unverified"

    with actions_path.open("r", encoding="utf-8", newline="") as f:
        action_reader = csv.DictReader(f)
        assert action_reader.fieldnames == ACTION_ITEMS_HEADERS
        action_rows = list(action_reader)
        assert len(action_rows) == 2
        assert action_rows[0]["task"] == 'अपडेटेड इनवॉइस पाठवा, "ग्राहक कॉपी"'
        assert action_rows[0]["owner"] == "स्नेहा कुलकर्णी"
        assert action_rows[0]["verification_status"] == "verified"
        assert "verified" in action_rows[0]["verification_reason"].lower()
        assert action_rows[1]["verification_status"] == "unverified"


def test_empty_action_items_writes_header_only_csv(tmp_path: Path) -> None:
    result = _build_pipeline_result(
        transcript="Simple greeting call with no follow-up tasks.",
        entities=[],
        action_items=[],
    )

    actions_path = export_action_items_csv(result, tmp_path / "action_items.csv")
    with actions_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ACTION_ITEMS_HEADERS
        rows = list(reader)
        assert rows == []


def test_json_serialization_preserves_unicode_and_excludes_secrets(tmp_path: Path) -> None:
    result = _build_pipeline_result(
        transcript="प्रिया: मी उद्या अहवाल पाठवेन.",
        entities=[PersonEntity(name="प्रिया", evidence_quote="प्रिया: मी उद्या अहवाल पाठवेन.")],
        action_items=[],
    )

    json_path = export_result_json(result, tmp_path / "result.json")
    raw_text = json_path.read_text(encoding="utf-8")
    assert "प्रिया" in raw_text
    assert "gemini_api_key" not in raw_text.lower()

    parsed = json.loads(raw_text)
    assert parsed["audio_filename"] == "call_record.wav"
    assert parsed["processing_status"] == "completed"
    assert parsed["evidence_validation"]["schema_valid"] is True
