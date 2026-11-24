"""Streamlit web dashboard for Speech-Analytics-Pipeline."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure repository root is on sys.path when launched via `streamlit run streamlit_app/app.py`.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.config import (  # noqa: E402
    SUPPORTED_WHISPER_DEVICES,
    SUPPORTED_WHISPER_MODELS,
    AppConfig,
)
from app.export.csv_exporter import (  # noqa: E402
    build_action_items_csv_string,
    build_report_csv_string,
    build_result_json_string,
)
from app.pipeline import PipelineResult, SpeechAnalyticsPipeline  # noqa: E402

_AUDIO_MIME_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
}


def _detect_cuda_status() -> str:
    """Safely inspect PyTorch CUDA availability without downloading models."""
    try:
        import torch

        if torch.cuda.is_available():
            return f"CUDA available ({torch.cuda.get_device_name(0)})"
        return "CPU only (CUDA unavailable)"
    except ImportError:
        return "PyTorch not detected in current environment"


def _format_file_size(num_bytes: int) -> str:
    """Format file byte count into human-readable units."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.2f} MB"


def _format_duration(seconds: float | None) -> str:
    """Format audio duration in seconds cleanly."""
    if seconds is None:
        return "Unavailable"
    mins, secs = divmod(int(round(seconds)), 60)
    if mins > 0:
        return f"{mins}m {secs}s ({seconds:.1f}s)"
    return f"{seconds:.1f}s"


def _sentiment_badge(label: str) -> str:
    badges = {
        "positive": "Positive",
        "negative": "Negative",
        "neutral": "Neutral",
        "mixed": "Mixed",
    }
    return badges.get(label, label.title())


def _render_empty_state() -> None:
    """Render clean onboarding state with zero fabricated analytics."""
    st.info(
        "Upload a call recording (WAV, MP3, M4A, or FLAC) above, select your local "
        "Whisper model and device, and click **Process Recording** to generate a local "
        "transcript and evidence-checked Gemini analytics."
    )
    st.markdown(
        """
**How the pipeline works:**
1. **Local Whisper Transcription** — Audio stays on your machine; only Whisper runs locally.
2. **Gemini Structured Analytics** — Only the generated transcript text is sent to Gemini.
3. **Pydantic v2 Schema Validation** — Enforces strict structure, bounds, and types.
4. **Deterministic Evidence Verification** — Checks extracted names, quotes, tasks, owners, and deadlines against the original transcript.
"""
    )


def _render_results(result: PipelineResult) -> None:
    """Render transcription, analytics, validation overview, and export controls."""
    if result.execution_metadata.error_message:
        st.error(
            f"Pipeline reported `{result.processing_status}` at stage "
            f"`{result.execution_metadata.error_stage}`: "
            f"{result.execution_metadata.error_message}"
        )

    # B. Transcription Section
    st.header("1. Local Whisper Transcription")
    if result.transcription is not None:
        tr = result.transcription
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Detected Language", tr.detected_language.upper())
        col2.metric("Whisper Model", f"{tr.model_name} ({tr.device_used})")
        col3.metric("Processing Time", f"{tr.processing_time_seconds:.2f}s")
        col4.metric("Audio Duration", _format_duration(tr.audio_duration_seconds))

        if tr.long_recording_warning:
            st.warning(tr.long_recording_warning)

        st.caption("Use the copy icon in the top-right corner of the transcript block to copy.")
        st.code(tr.transcript, language=None)
    else:
        st.warning("No transcript was produced.")

    if result.analytics is None:
        # Allow transcript export even on partial failure after Whisper succeeded.
        if result.transcription is not None:
            st.subheader("Partial Export")
            st.download_button(
                label="Download Transcript (TXT)",
                data=result.transcription.transcript,
                file_name=f"{Path(result.audio_filename).stem}_transcript.txt",
                mime="text/plain",
            )
        return

    analytics = result.analytics
    ev = result.evidence_validation

    # G. Validation Overview (prominent distinction between Schema Valid & Evidence Verified)
    st.header("2. Validation & Evidence Overview")
    if ev is not None:
        v1, v2, v3, v4 = st.columns(4)
        v1.metric(
            "Schema Validation",
            "Schema Valid" if ev.schema_valid else "Schema Invalid",
        )
        v2.metric(
            "Names / Entities Verified",
            f"{ev.verified_entities} / {ev.total_entities}",
            delta=f"{ev.unverified_entities} unverified" if ev.unverified_entities else "All verified",
            delta_color="inverse" if ev.unverified_entities else "normal",
        )
        v3.metric(
            "Action Items Verified",
            f"{ev.verified_action_items} / {ev.total_action_items}",
            delta=(
                f"{ev.unverified_action_items} unverified"
                if ev.unverified_action_items
                else "All verified"
            ),
            delta_color="inverse" if ev.unverified_action_items else "normal",
        )
        v4.metric("Evidence Status", ev.overall_status.replace("_", " ").title())

        with st.expander("Structural Validation vs. Evidence Verification", expanded=False):
            st.markdown(
                """
- **Schema Valid**: The Gemini JSON output satisfies all Pydantic v2 field types, sentiment categories, and confidence bounds (`[0.0, 1.0]`).
- **Evidence Verified**: Deterministic Unicode-aware matching confirmed that the extracted name, quote, task keywords, and any explicit owner/deadline appear together in the transcript.
- **Evidence Unverified**: The claim was preserved for auditability, but lacked exact transcript quote support, task alignment, or explicit owner/deadline backing. **Do not treat unverified items as confirmed facts or commitments.**
"""
            )
            st.caption(ev.limitations_note)

    # C. Executive Summary
    st.header("3. Executive Summary")
    sum_col, sent_col = st.columns([2, 1])
    with sum_col:
        st.subheader("Call Summary")
        st.write(analytics.summary)
    with sent_col:
        st.subheader("Overall Sentiment")
        st.metric("Sentiment", _sentiment_badge(analytics.sentiment.label))
        if analytics.sentiment.confidence is not None:
            st.progress(analytics.sentiment.confidence)
            st.caption(
                f"Model-reported sentiment confidence: {analytics.sentiment.confidence:.0%} "
                "(indicative score, not factual accuracy)."
            )
        if analytics.overall_confidence is not None:
            st.caption(
                f"Model-reported overall confidence: {analytics.overall_confidence:.0%} "
                "(indicative score, not factual accuracy)."
            )
        st.write(analytics.sentiment.explanation)

    # D. Topics Section
    st.header("4. Discussion Topics")
    if analytics.topics:
        for topic in analytics.topics:
            with st.container(border=True):
                st.markdown(f"**{topic.name}**")
                st.write(topic.description)
                st.caption(f"Supporting quote: \"{topic.supporting_quote}\"")
    else:
        st.info("No distinct discussion topics were extracted.")

    # E. People and Entities Section
    st.header("5. People & Named Entities")
    if analytics.entities:
        entity_records = (
            [r for r in ev.records if r.item_type == "entity"] if ev is not None else []
        )
        for idx, entity in enumerate(analytics.entities):
            rec = entity_records[idx] if idx < len(entity_records) else None
            status = rec.verification_status if rec else "unverified"
            reason = rec.verification_reason if rec else "Not checked."

            with st.container(border=True):
                left, right = st.columns([3, 1])
                role_label = f" — *{entity.role}*" if entity.role else ""
                left.markdown(f"**{entity.name}**{role_label}")
                left.caption(f"Evidence quote: \"{entity.evidence_quote}\"")
                if status == "verified":
                    right.success("Evidence Verified")
                else:
                    right.warning("Evidence Unverified")
                    left.caption(f"Reason: {reason}")
    else:
        st.info("No people or named entities were identified in the transcript.")

    # F. Action Items Section
    st.header("6. Action Items")
    if analytics.action_items:
        action_records = (
            [r for r in ev.records if r.item_type == "action_item"] if ev is not None else []
        )
        rows = []
        for idx, item in enumerate(analytics.action_items):
            rec = action_records[idx] if idx < len(action_records) else None
            status = rec.verification_status if rec else item.verification_status
            reason = rec.verification_reason if rec else "Not verified."
            rows.append(
                {
                    "Status": (
                        "Evidence Verified" if status == "verified" else "Evidence Unverified"
                    ),
                    "Task": item.task,
                    "Owner": item.owner or "Unassigned",
                    "Deadline": item.deadline or "Not specified",
                    "Evidence Quote": item.evidence_quote,
                    "Verification Reason": reason,
                    "_is_verified": status == "verified",
                }
            )

        verified_rows = [r for r in rows if r["_is_verified"]]
        unverified_rows = [r for r in rows if not r["_is_verified"]]

        st.subheader(f"Verified Action Items ({len(verified_rows)})")
        if verified_rows:
            st.dataframe(
                pd.DataFrame(verified_rows).drop(columns=["_is_verified"]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No action items met conservative evidence verification criteria.")

        st.subheader(f"Unverified Action Items — Needs Human Review ({len(unverified_rows)})")
        if unverified_rows:
            st.warning(
                "The items below could not be fully verified against transcript evidence "
                "and must not be treated as confirmed commitments without manual review."
            )
            st.dataframe(
                pd.DataFrame(unverified_rows).drop(columns=["_is_verified"]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No unverified action items.")
    else:
        st.info("No action items were identified in the transcript.")

    # H. Export Section
    st.header("7. Export Results")
    stem = Path(result.audio_filename).stem or "recording"
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        st.download_button(
            label="Download Transcript (TXT)",
            data=result.transcription.transcript if result.transcription else "",
            file_name=f"{stem}_transcript.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with e2:
        st.download_button(
            label="Download Analytics (JSON)",
            data=build_result_json_string(result),
            file_name=f"{stem}_analytics.json",
            mime="application/json",
            use_container_width=True,
        )
    with e3:
        st.download_button(
            label="Download Action Items (CSV)",
            data=build_action_items_csv_string(result),
            file_name="action_items.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with e4:
        st.download_button(
            label="Download Full Report (CSV)",
            data=build_report_csv_string(result),
            file_name="speech_analytics_report.csv",
            mime="text/csv",
            use_container_width=True,
        )


def main() -> None:
    """Render the Speech Analytics Pipeline Streamlit dashboard."""
    st.set_page_config(
        page_title="Speech Analytics Pipeline",
        layout="wide",
    )

    st.title("Speech Analytics Pipeline")
    st.caption(
        "Transform call recordings into structured, evidence-checked insights using Whisper and Gemini."
    )

    config = AppConfig.from_env()

    with st.sidebar:
        st.header("Pipeline Configuration")
        selected_model = st.selectbox(
            "Local Whisper Model",
            options=list(SUPPORTED_WHISPER_MODELS),
            index=list(SUPPORTED_WHISPER_MODELS).index(config.whisper_model),
            help="Larger models improve accuracy at the cost of local CPU/GPU time and RAM.",
        )
        selected_device = st.selectbox(
            "Whisper Execution Device",
            options=list(SUPPORTED_WHISPER_DEVICES),
            index=list(SUPPORTED_WHISPER_DEVICES).index(config.whisper_device),
            help="Use 'auto' to select CUDA when available, otherwise CPU (FP32).",
        )
        st.text(f"Hardware: {_detect_cuda_status()}")
        st.text(f"Gemini Model: {config.gemini_model}")
        if config.has_valid_gemini_key:
            st.success("Gemini API key configured in .env")
        else:
            st.warning(
                "GEMINI_API_KEY is not configured in .env. "
                "Set it locally before processing recordings."
            )

    # A. Audio Upload
    uploaded_file = st.file_uploader(
        "Upload a call recording",
        type=["wav", "mp3", "m4a", "flac"],
        help="Audio is transcribed locally with Whisper. Only the transcript is sent to Gemini.",
    )

    if uploaded_file is not None:
        meta1, meta2 = st.columns(2)
        meta1.text(f"Filename: {uploaded_file.name}")
        meta2.text(f"File size: {_format_file_size(uploaded_file.size)}")

        suffix = Path(uploaded_file.name).suffix.lower()
        st.audio(uploaded_file.getvalue(), format=_AUDIO_MIME_TYPES.get(suffix, "audio/wav"))

        process_clicked = st.button(
            "Process Recording",
            type="primary",
            disabled=not config.has_valid_gemini_key,
        )

        if process_clicked:
            config.whisper_model = selected_model  # type: ignore[assignment]
            config.whisper_device = selected_device  # type: ignore[assignment]

            tmp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = Path(tmp_file.name)

                pipeline = SpeechAnalyticsPipeline(config=config)
                with st.spinner("Running local Whisper transcription and Gemini analytics..."):
                    result = pipeline.run(tmp_path, save_outputs=False)
                # Preserve original uploaded filename in the result and exports.
                result = result.model_copy(update={"audio_filename": uploaded_file.name})
                if result.transcription is not None:
                    result.transcription = result.transcription.model_copy(
                        update={"audio_filename": uploaded_file.name}
                    )
                st.session_state["pipeline_result"] = result
            finally:
                if tmp_path is not None and tmp_path.exists():
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

    stored_result: PipelineResult | None = st.session_state.get("pipeline_result")
    if stored_result is None:
        _render_empty_state()
    else:
        _render_results(stored_result)


if __name__ == "__main__":
    main()
