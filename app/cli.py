"""Command-line interface for Speech-Analytics-Pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.config import (
    SUPPORTED_WHISPER_DEVICES,
    SUPPORTED_WHISPER_MODELS,
    AppConfig,
)
from app.exceptions import ConfigurationError
from app.pipeline import PipelineResult, SpeechAnalyticsPipeline


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description=(
            "Speech-Analytics-Pipeline: Transcribe audio locally with OpenAI Whisper, "
            "extract structured insights with Google Gemini, and run deterministic "
            "transcript evidence verification."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    process_parser = subparsers.add_parser(
        "process",
        help="Process an audio recording end-to-end.",
    )
    process_parser.add_argument(
        "--audio",
        required=True,
        type=Path,
        help="Path to the input audio recording (WAV, MP3, M4A, or FLAC).",
    )
    process_parser.add_argument(
        "--model",
        choices=SUPPORTED_WHISPER_MODELS,
        default=None,
        help="Local Whisper model size (overrides WHISPER_MODEL).",
    )
    process_parser.add_argument(
        "--device",
        choices=SUPPORTED_WHISPER_DEVICES,
        default=None,
        help="Whisper execution device (overrides WHISPER_DEVICE).",
    )
    process_parser.add_argument(
        "--language",
        default=None,
        help="Optional ISO-639-1 language hint for Whisper (default: auto-detect).",
    )
    process_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for CSV and JSON reports (overrides OUTPUT_DIR).",
    )
    process_parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving CSV and JSON report files to disk.",
    )
    return parser


def _print_progress(stage: str, message: str) -> None:
    print(f"[{stage.upper()}] {message}", flush=True)


def _print_summary(result: PipelineResult) -> None:
    """Print human-readable pipeline summary without exposing secrets."""
    print("\n=== Pipeline Execution Summary ===")
    print(f"Audio file        : {result.audio_filename}")
    print(f"Processing status : {result.processing_status}")
    print(f"Whisper model     : {result.execution_metadata.whisper_model}")
    print(f"Whisper device    : {result.execution_metadata.whisper_device}")
    print(f"Gemini model      : {result.execution_metadata.gemini_model}")
    print(f"Total time (s)    : {result.execution_metadata.total_duration_seconds}")

    if result.transcription is not None:
        print(f"Detected language : {result.transcription.detected_language}")
        print(f"Transcript chars  : {len(result.transcription.transcript)}")

    if result.evidence_validation is not None:
        ev = result.evidence_validation
        print("\n=== Validation Results ===")
        print(f"Schema valid      : {ev.schema_valid}")
        print(f"Evidence status   : {ev.overall_status}")
        print(
            f"Entities          : {ev.verified_entities}/{ev.total_entities} verified "
            f"({ev.unverified_entities} unverified)"
        )
        print(
            f"Action items      : {ev.verified_action_items}/{ev.total_action_items} verified "
            f"({ev.unverified_action_items} unverified)"
        )

    if result.execution_metadata.saved_files:
        print("\n=== Exported Files ===")
        for label, file_path in result.execution_metadata.saved_files.items():
            print(f"- {label}: {file_path}")

    if result.execution_metadata.error_message:
        print(
            f"\n[ERROR] Stage '{result.execution_metadata.error_stage}': "
            f"{result.execution_metadata.error_message}",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint returning process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command != "process":
        parser.print_help()
        return 1

    try:
        config = AppConfig.from_env()
        if args.model is not None:
            config.whisper_model = args.model
        if args.device is not None:
            config.whisper_device = args.device
        if args.output is not None:
            config.output_dir = args.output
    except ConfigurationError as exc:
        print(f"[CONFIGURATION ERROR] {exc}", file=sys.stderr)
        return 1

    pipeline = SpeechAnalyticsPipeline(config=config)
    result = pipeline.run(
        args.audio,
        language=args.language,
        save_outputs=not args.no_save,
        output_dir=config.output_dir,
        progress_callback=_print_progress,
    )
    _print_summary(result)

    return 0 if result.processing_status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
