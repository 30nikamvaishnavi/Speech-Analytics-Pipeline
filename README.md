# Speech-Analytics-Pipeline

An end-to-end, two-stage AI speech analytics system that transcribes call recordings locally using **OpenAI Whisper**, extracts structured conversation intelligence using the **Google Gemini API (`google-genai`)**, enforces strict **Pydantic v2** schema contracts, and runs a deterministic **transcript evidence verification** layer before displaying or exporting results.

---

## 1. Project Overview

**Speech-Analytics-Pipeline** converts raw call recordings (`.wav`, `.mp3`, `.m4a`, `.flac`) into structured, auditable analytics—call summaries, sentiment classification, discussion topics, named entities, and action items with explicit owners and deadlines—without sending raw audio files to external APIs.

### End-to-End Workflow

```text
Audio → Local Whisper → Transcript → Gemini → Pydantic Validation → Evidence Checking → Streamlit → CSV/JSON
```

---

## 2. Problem Statement

Call centers, sales teams, and engineering support desks generate large volumes of recorded conversations, yet raw audio is difficult to search, audit, and convert into reliable follow-ups. Purely generative LLM pipelines introduce two critical risks:
1. **Privacy & Cost**: Sending raw audio to third-party transcription services increases exposure and recurring per-minute API costs.
2. **Hallucinations**: LLMs may fabricate speaker names, action items, owners, or deadlines that were never stated in the recording.

This project addresses both challenges by running **OpenAI Whisper locally** for transcription and pairing **Gemini** structured JSON extraction with **Pydantic v2 validation** and **deterministic Unicode-aware evidence verification**.

---

## 3. Features

- **100% Local Transcription**: Uses `openai-whisper` locally on CPU (FP32) or CUDA (FP16) with model caching (`tiny`, `base`, `small`).
- **Multilingual Support**: Supports English and multilingual audio (including Hindi and Marathi Devanagari scripts) wherever supported by Whisper.
- **Structured Gemini Analytics**: Extracts summary, sentiment (`positive`, `negative`, `neutral`, `mixed`), topics, people/entities, and action items using the official `google-genai` SDK.
- **Strict Pydantic v2 Validation**: Rejects malformed payloads, invalid sentiment categories, missing required fields, and out-of-range confidence values (`[0.0, 1.0]`).
- **Deterministic Evidence Verification**: Verifies extracted names, quotes, action items, owners, and deadlines against the original transcript using Unicode-aware boundary matching without deleting unverified records.
- **Interactive Streamlit Dashboard**: Audio upload, playback, copyable transcript, visual validation status distinction (`Schema Valid`, `Evidence Verified`, `Evidence Unverified`), and one-click exports.
- **CLI & Automated Exports**: Full command-line runner producing `speech_analytics_report.csv`, `action_items.csv`, and structured JSON.
- **Offline Mocked Test Suite**: Comprehensive Pytest coverage without downloading Whisper weights or making live Gemini API requests.

---

## 4. Architecture

```text
┌─────────────────────────┐
│  Audio Recording        │  (WAV, MP3, M4A, FLAC)
└───────────┬─────────────┘
            │  Local file validation + duration check (SoundFile / Librosa)
            ▼
┌─────────────────────────┐
│  WhisperTranscriber     │  Local OpenAI Whisper (tiny / base / small)
│  (CPU FP32 / CUDA FP16) │  Audio NEVER leaves the local machine
└───────────┬─────────────┘
            │  Transcript text + detected language + timing metadata
            ▼
┌─────────────────────────┐
│  GeminiAnalyticsClient  │  Official google-genai SDK (transcript text only)
│  + Prompts              │  Retries with exponential backoff, no fabricated fallbacks
└───────────┬─────────────┘
            │  Structured JSON payload
            ▼
┌─────────────────────────┐
│  Pydantic v2 Schemas    │  Structural validation (types, bounds, literals)
└───────────┬─────────────┘
            │  Schema-valid SpeechAnalytics
            ▼
┌─────────────────────────┐
│  EvidenceChecker        │  Deterministic Unicode boundary & quote verification
│  (Validators)           │  Flags verified vs. unverified; preserves all records
└───────────┬─────────────┘
            │  Consolidated PipelineResult
            ▼
┌───────────────────────────────────────────┐
│  Streamlit UI  |  CLI  |  CSV/JSON Export │
└───────────────────────────────────────────┘
```

---

## 5. Technology Stack

- **Language**: Python 3.10+
- **Local Speech-to-Text**: OpenAI Whisper (`openai-whisper`), PyTorch (`torch`), SoundFile, Librosa, FFmpeg
- **Generative AI Analytics**: Google Gemini API via official `google-genai` Python SDK
- **Data Validation**: Pydantic v2 (`pydantic`)
- **Configuration**: `python-dotenv`
- **Dashboard & Data**: Streamlit, Pandas
- **Testing & Linting**: Pytest, `pytest-mock`, Ruff

---

## 6. Directory Structure

```text
Speech-Analytics-Pipeline/
│
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── exceptions.py
│   │
│   ├── transcription/
│   │   ├── __init__.py
│   │   └── whisper_transcriber.py
│   │
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── gemini_client.py
│   │   ├── prompts.py
│   │   ├── schemas.py
│   │   └── analyzer.py
│   │
│   ├── validation/
│   │   ├── __init__.py
│   │   ├── evidence_checker.py
│   │   └── validators.py
│   │
│   ├── export/
│   │   ├── __init__.py
│   │   └── csv_exporter.py
│   │
│   ├── pipeline.py
│   └── cli.py
│
├── streamlit_app/
│   └── app.py
│
├── data/
│   └── audio/
│       └── .gitkeep
│
├── results/
│   └── .gitkeep
│
├── tests/
│   ├── __init__.py
│   ├── test_schemas.py
│   ├── test_evidence_checker.py
│   ├── test_analyzer.py
│   ├── test_pipeline.py
│   └── test_csv_exporter.py
│
├── .env.example
├── .gitignore
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 7. Installation Instructions

Clone the repository and navigate into the project directory:

```bash
git clone <your-repository-url>
cd Speech-Analytics-Pipeline
```

---

## 8. Python Environment Setup

Create and activate an isolated Python 3.10+ virtual environment, then install dependencies:

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 9. FFmpeg Installation

OpenAI Whisper requires **FFmpeg** to decode audio formats (`.wav`, `.mp3`, `.m4a`, `.flac`). Install FFmpeg separately and verify it is available on your system `PATH`:

- **Ubuntu / Debian**: `sudo apt update && sudo apt install -y ffmpeg`
- **macOS (Homebrew)**: `brew install ffmpeg`
- **Windows (winget)**: `winget install Gyan.FFmpeg`

Verify installation:

```bash
ffmpeg -version
```

---

## 10. Whisper Model Setup

Whisper weights (`tiny`, `base`, `small`) are downloaded automatically by `openai-whisper` on first local transcription and cached under your user cache directory (excluded from Git via `.gitignore`).

| Model | Relative Speed | Memory Requirement | Recommended Use |
|---|---|---|---|
| `tiny` | Fastest | ~1 GB | Quick drafts / constrained CPU |
| `base` | Balanced | ~1 GB | Default general transcription |
| `small` | Higher accuracy | ~2 GB | Multilingual or noisy calls |

- **Device Selection**: `WHISPER_DEVICE=auto` selects CUDA (`fp16=True`) when a compatible GPU is detected by PyTorch, and falls back to CPU (`fp16=False` / FP32).
- **Long Recordings**: Recordings are processed across 30-second sliding windows without silent truncation. Longer files (>30 minutes) require proportionally more RAM/VRAM and CPU/GPU time.

---

## 11. Gemini API Key Configuration

1. Obtain an API key from Google AI Studio.
2. Never hardcode the key in source files or commit `.env` to Git.
3. Choose a supported Gemini model (e.g., `gemini-4-hydrogen`).

---

## 12. `.env` Setup

Copy `.env.example` to `.env` and populate your values:

```bash
cp .env.example .env
```

Edit `.env`:

```env
GEMINI_API_KEY=your_actual_gemini_api_key
GEMINI_MODEL=gemini-4-hydrogen
WHISPER_MODEL=base
WHISPER_DEVICE=auto
OUTPUT_DIR=results
```

---

## 13. CLI Usage

Show CLI help:

```bash
python -m app.cli --help
```

Run the complete pipeline on a recording:

```bash
python -m app.cli process --audio data/audio/call.wav
```

Specify Whisper model, execution device, and output directory:

```bash
python -m app.cli process --audio data/audio/call.wav --model base --device cpu --output results/
```

Optional language hint (skip auto-detection):

```bash
python -m app.cli process --audio data/audio/call.wav --language hi --output results/
```

---

## 14. Streamlit Usage

Launch the interactive dashboard:

```bash
streamlit run streamlit_app/app.py
```

1. Upload a `.wav`, `.mp3`, `.m4a`, or `.flac` recording.
2. Listen to playback and choose the local Whisper model and execution device.
3. Click **Process Recording** (Gemini is only called after you explicitly start processing).
4. Review the transcript, executive summary, topics, entities, action items, and validation status.
5. Export `.txt`, `.json`, `action_items.csv`, or `speech_analytics_report.csv`.

---

## 15. Pydantic Schema Explanation

Defined in `app/analytics/schemas.py` using Pydantic v2 (`extra="forbid"`):

- **`Sentiment`**: `label` (`positive | negative | neutral | mixed`), `confidence` (`0.0..1.0` or `null`), `explanation`.
- **`Topic`**: `name`, `description`, `supporting_quote`.
- **`PersonEntity`**: `name`, `role` (optional), `evidence_quote`.
- **`ActionItem`**: `task`, `owner` (optional), `deadline` (optional), `evidence_quote`, `verification_status` (`verified | unverified`).
- **`SpeechAnalytics`**: Top-level model aggregating `summary`, `sentiment`, `topics`, `entities`, `action_items`, and `overall_confidence`.

---

## 16. Evidence Validation Methodology

Defined in `app/validation/evidence_checker.py` and `app/validation/validators.py`:

1. **Schema Validity vs. Evidence Validity**:
   - **Schema Valid** confirms the Gemini JSON matches Pydantic types and bounds.
   - **Evidence Verified** confirms the claim is grounded in the Whisper transcript.
   - **Evidence Unverified** preserves the LLM extraction for auditability while flagging it as unconfirmed.
2. **Person/Entity Verification**:
   - Performs NFC Unicode normalization, whitespace collapsing, and case-insensitive English matching while preserving Devanagari (Hindi/Marathi) characters and combining marks.
   - Enforces Unicode word boundaries (`L`, `N`, `M`, `_`) so partial substrings (e.g., `"Ana"` in `"Ananya"` or `"रोह"` in `"रोहिता"`) are never falsely accepted.
   - Checks that the entity also appears inside its `evidence_quote`.
3. **Conservative Action-Item Verification**:
   - Verifies `evidence_quote` appears verbatim in the transcript.
   - Checks meaningful content-token overlap between `task` and `evidence_quote` and rejects affirmative tasks paired with negated quotes.
   - Independently verifies `owner` and `deadline` (when present) inside the transcript and the action item's local quote context.
   - Never deletes unverified items; marks them `unverified` with an explicit `verification_reason`.

---

## 17. CSV and JSON Exports

Defined in `app/export/csv_exporter.py`:

- **`speech_analytics_report.csv`**: Contains `audio_filename`, `transcript`, `summary`, `sentiment`, `topics`, `entities`, `action_items`, and `processing_status` (including verification status and reasons inside `entities` and `action_items`).
- **`action_items.csv`**: Contains `audio_filename`, `task`, `owner`, `deadline`, `evidence_quote`, `verification_status`, and `verification_reason`. Empty action-item lists emit only the header row.
- **JSON Export (`*_analytics.json`)**: Full UTF-8 `PipelineResult` excluding all secrets and API keys.
- **Unicode & Escaping**: All exports use UTF-8 (`ensure_ascii=False`) and standard RFC-4180 CSV quoting to preserve commas, quotes, Hindi, and Marathi text.

---

## 18. Pytest Instructions

All automated tests run offline with mocked Whisper/Gemini dependencies and temporary files:

```bash
pytest
```

---

## 19. Ruff Instructions

Check code formatting and lint rules:

```bash
ruff check .
```

---

## 20. Security Considerations

- `GEMINI_API_KEY` is loaded exclusively from `.env` via `python-dotenv`.
- API keys are masked in `AppConfig.__repr__`, redacted from logs via `SecretRedactingFilter`, sanitized out of exception messages, and excluded from Streamlit UI and exports.
- `.env`, `.streamlit/secrets.toml`, audio files, and generated reports are excluded in `.gitignore`.

---

## 21. Limitations

- **Audio is processed locally by Whisper**, and **only the resulting transcript is sent to Gemini**.
- **Gemini analytics may contain unsupported interpretations** or misclassifications, especially on noisy or ambiguous transcripts.
- **Evidence checking reduces unsupported extraction but cannot guarantee semantic correctness**—it is a deterministic lexical/quote safeguard, not a factual truth oracle.
- **Transcription errors can affect all downstream analytics**: if Whisper mishears a name, number, or negation, Gemini and the evidence checker operate on that imperfect transcript.
- **API usage may incur charges** depending on your Google Gemini account tier and selected model.

---

## 22. Future Improvements

- Speaker diarization (e.g., `pyannote.audio`) to attribute multi-speaker turns deterministically before analytics.
- Batch folder processing and asynchronous queue support for large call archives.
- Optional local-only open-weight LLM mode for fully air-gapped deployments.
- Timestamp-aligned transcript playback in the Streamlit UI.

---

### Experimental Results

**Status: Pending actual execution.**

No actual audio recordings have been evaluated yet.

| Metric | Result |
|---|---|
| Audio recordings processed | Pending |
| Transcription status | Pending |
| Gemini analytics status | Pending |
| Verified action items | Pending |
| Unverified action items | Pending |
