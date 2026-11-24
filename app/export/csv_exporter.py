"""Unicode-safe CSV and JSON exporters for Speech-Analytics-Pipeline results."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.exceptions import ExportError

if TYPE_CHECKING:
    from app.pipeline import PipelineResult

REPORT_HEADERS: list[str] = [
    "audio_filename",
    "transcript",
    "summary",
    "sentiment",
    "topics",
    "entities",
    "action_items",
    "processing_status",
]

ACTION_ITEMS_HEADERS: list[str] = [
    "audio_filename",
    "task",
    "owner",
    "deadline",
    "evidence_quote",
    "verification_status",
    "verification_reason",
]


def _json_cell(payload: Any) -> str:
    """Serialize structured cell payloads preserving Unicode characters."""
    return json.dumps(payload, ensure_ascii=False)


def _lookup_entity_verifications(result: PipelineResult) -> list[dict[str, Any]]:
    """Build serializable entity records merged with deterministic verification metadata."""
    if result.analytics is None:
        return []

    entity_records = []
    if result.evidence_validation is not None:
        entity_records = [
            rec for rec in result.evidence_validation.records if rec.item_type == "entity"
        ]

    enriched: list[dict[str, Any]] = []
    for idx, entity in enumerate(result.analytics.entities):
        matched_record = entity_records[idx] if idx < len(entity_records) else None
        enriched.append(
            {
                "name": entity.name,
                "role": entity.role,
                "evidence_quote": entity.evidence_quote,
                "verification_status": (
                    matched_record.verification_status if matched_record else "unverified"
                ),
                "verification_reason": (
                    matched_record.verification_reason
                    if matched_record
                    else "Evidence verification was not executed."
                ),
            }
        )
    return enriched


def _lookup_action_item_rows(result: PipelineResult) -> list[dict[str, str]]:
    """Build flattened action-item dictionaries including verification status and reason."""
    if result.analytics is None or not result.analytics.action_items:
        return []

    action_records = []
    if result.evidence_validation is not None:
        action_records = [
            rec for rec in result.evidence_validation.records if rec.item_type == "action_item"
        ]

    rows: list[dict[str, str]] = []
    for idx, item in enumerate(result.analytics.action_items):
        matched_record = action_records[idx] if idx < len(action_records) else None
        status = (
            matched_record.verification_status
            if matched_record
            else item.verification_status
        )
        reason = (
            matched_record.verification_reason
            if matched_record
            else "Evidence verification was not executed."
        )
        rows.append(
            {
                "audio_filename": result.audio_filename,
                "task": item.task,
                "owner": item.owner or "",
                "deadline": item.deadline or "",
                "evidence_quote": item.evidence_quote,
                "verification_status": status,
                "verification_reason": reason,
            }
        )
    return rows


def build_report_csv_string(result: PipelineResult) -> str:
    """Build the full ``speech_analytics_report.csv`` content as a UTF-8 string."""
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=REPORT_HEADERS,
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )
    writer.writeheader()

    transcript = result.transcription.transcript if result.transcription else ""
    if result.analytics is not None:
        summary = result.analytics.summary
        sentiment_json = _json_cell(result.analytics.sentiment.model_dump(mode="json"))
        topics_json = _json_cell(
            [topic.model_dump(mode="json") for topic in result.analytics.topics]
        )
        entities_json = _json_cell(_lookup_entity_verifications(result))
        action_items_json = _json_cell(_lookup_action_item_rows(result))
    else:
        summary = ""
        sentiment_json = ""
        topics_json = _json_cell([])
        entities_json = _json_cell([])
        action_items_json = _json_cell([])

    writer.writerow(
        {
            "audio_filename": result.audio_filename,
            "transcript": transcript,
            "summary": summary,
            "sentiment": sentiment_json,
            "topics": topics_json,
            "entities": entities_json,
            "action_items": action_items_json,
            "processing_status": result.processing_status,
        }
    )
    return buffer.getvalue()


def build_action_items_csv_string(result: PipelineResult) -> str:
    """Build ``action_items.csv`` content as a UTF-8 string.

    When there are no action items, only the CSV header row is written so
    empty lists never produce misleading blank records.
    """
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=ACTION_ITEMS_HEADERS,
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )
    writer.writeheader()
    for row in _lookup_action_item_rows(result):
        writer.writerow(row)
    return buffer.getvalue()


def build_result_json_string(result: PipelineResult) -> str:
    """Serialize the complete PipelineResult to pretty-printed Unicode JSON."""
    payload = result.model_dump(mode="json")
    return json.dumps(payload, ensure_ascii=False, indent=2)


def export_report_csv(result: PipelineResult, output_path: str | Path) -> Path:
    """Write ``speech_analytics_report.csv`` to ``output_path`` using UTF-8."""
    target = Path(output_path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        content = build_report_csv_string(result)
        target.write_text(content, encoding="utf-8", newline="")
        return target
    except Exception as exc:
        raise ExportError(f"Failed to export report CSV to '{target}': {exc}") from exc


def export_action_items_csv(result: PipelineResult, output_path: str | Path) -> Path:
    """Write ``action_items.csv`` to ``output_path`` using UTF-8."""
    target = Path(output_path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        content = build_action_items_csv_string(result)
        target.write_text(content, encoding="utf-8", newline="")
        return target
    except Exception as exc:
        raise ExportError(f"Failed to export action items CSV to '{target}': {exc}") from exc


def export_result_json(result: PipelineResult, output_path: str | Path) -> Path:
    """Write the full structured pipeline result as UTF-8 JSON."""
    target = Path(output_path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        content = build_result_json_string(result)
        target.write_text(content, encoding="utf-8")
        return target
    except Exception as exc:
        raise ExportError(f"Failed to export JSON result to '{target}': {exc}") from exc


def export_all(
    result: PipelineResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Export report CSV, action-items CSV, and full JSON into ``output_dir``."""
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(result.audio_filename).stem or "recording"
    report_csv_path = export_report_csv(result, base_dir / "speech_analytics_report.csv")
    action_csv_path = export_action_items_csv(result, base_dir / "action_items.csv")
    json_path = export_result_json(result, base_dir / f"{stem}_analytics.json")

    return {
        "report_csv": report_csv_path,
        "action_items_csv": action_csv_path,
        "json": json_path,
    }
