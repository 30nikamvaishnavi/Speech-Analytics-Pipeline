"""Google Gemini API client using the official google-genai Python SDK.

Only Whisper-generated transcript text is ever sent to Gemini. Audio files are
never uploaded or transmitted externally.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from app.analytics.prompts import (
    SPEECH_ANALYTICS_SYSTEM_INSTRUCTION,
    build_analytics_user_prompt,
)
from app.analytics.schemas import SpeechAnalytics
from app.config import AppConfig, sanitize_error_message
from app.exceptions import (
    GeminiClientError,
    GeminiRateLimitError,
    GeminiResponseParseError,
    SchemaValidationError,
)

logger = logging.getLogger(__name__)

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)


def _strip_markdown_fences(raw_text: str) -> str:
    """Strip optional markdown code fences around JSON responses."""
    trimmed = raw_text.strip()
    match = _CODE_FENCE_RE.match(trimmed)
    return match.group(1).strip() if match else trimmed


def _is_rate_limit_error(exc: Exception) -> bool:
    """Heuristically detect 429 / RESOURCE_EXHAUSTED rate-limit exceptions."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        token in text
        for token in (
            "429",
            "resource_exhausted",
            "rate limit",
            "ratelimit",
            "quota",
            "too many requests",
        )
    )


def _is_retryable_error(exc: Exception) -> bool:
    """Return True for transient network, server, or rate-limit errors."""
    if _is_rate_limit_error(exc):
        return True
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        token in text
        for token in (
            "500",
            "502",
            "503",
            "504",
            "unavailable",
            "timeout",
            "timed out",
            "temporarily",
            "connection reset",
            "connection aborted",
        )
    )


class GeminiAnalyticsClient:
    """Reusable client for structured transcript analytics via Google Gemini.

    Parameters
    ----------
    config:
        Application configuration holding ``gemini_api_key`` and ``gemini_model``.
    genai_client:
        Optional pre-constructed ``google.genai.Client`` (or mock) for testing.
    sleep_fn:
        Optional sleep callable for deterministic retry testing.
    """

    def __init__(
        self,
        config: AppConfig,
        *,
        genai_client: Any | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._injected_client = genai_client
        self._client: Any | None = genai_client
        self._sleep = sleep_fn

    def _get_client(self) -> Any:
        """Lazily validate credentials and initialize the official google-genai client."""
        if self._client is not None:
            return self._client

        api_key = self._config.validate_gemini_config()
        try:
            from google import genai
        except ImportError as exc:
            raise GeminiClientError(
                "google-genai SDK is not installed. Install dependencies from requirements.txt."
            ) from exc

        self._client = genai.Client(api_key=api_key)
        return self._client

    def _build_generate_config(self) -> Any:
        """Build GenerateContentConfig requesting structured JSON matching SpeechAnalytics."""
        try:
            from google.genai import types

            return types.GenerateContentConfig(
                system_instruction=SPEECH_ANALYTICS_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=SpeechAnalytics,
                temperature=0.1,
            )
        except ImportError:
            return {
                "system_instruction": SPEECH_ANALYTICS_SYSTEM_INSTRUCTION,
                "response_mime_type": "application/json",
                "response_schema": SpeechAnalytics.model_json_schema(),
                "temperature": 0.1,
            }

    def analyze_transcript(
        self,
        transcript: str,
        *,
        detected_language: str | None = None,
    ) -> SpeechAnalytics:
        """Send a transcript string to Gemini and return Pydantic-validated SpeechAnalytics.

        Raises
        ------
        GeminiClientError
            If the transcript is empty, API credentials are missing, or API calls fail.
        GeminiRateLimitError
            If rate-limit retries are exhausted.
        GeminiResponseParseError
            If Gemini returns empty text or malformed JSON.
        SchemaValidationError
            If Gemini JSON does not satisfy the SpeechAnalytics Pydantic schema.
        """
        if not isinstance(transcript, str) or not transcript.strip():
            raise GeminiClientError("Transcript is empty; refusing to call Gemini API.")

        # Validate credentials before making any API call unless a test mock client is injected.
        if self._injected_client is None:
            self._config.validate_gemini_config()

        client = self._get_client()
        user_prompt = build_analytics_user_prompt(
            transcript=transcript,
            detected_language=detected_language,
        )
        gen_config = self._build_generate_config()

        max_attempts = self._config.gemini_max_retries + 1
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(
                    "Sending transcript to Gemini model '%s' (attempt %d/%d, chars=%d)",
                    self._config.gemini_model,
                    attempt,
                    max_attempts,
                    len(transcript),
                )
                response = client.models.generate_content(
                    model=self._config.gemini_model,
                    contents=user_prompt,
                    config=gen_config,
                )
                raw_text = getattr(response, "text", None)
                if not isinstance(raw_text, str) or not raw_text.strip():
                    raise GeminiResponseParseError(
                        "Gemini returned an empty response payload."
                    )
                return self._parse_and_validate(raw_text)

            except (GeminiResponseParseError, SchemaValidationError):
                raise
            except Exception as exc:
                last_error = exc
                safe_msg = sanitize_error_message(str(exc), self._config.gemini_api_key)
                is_rate_limit = _is_rate_limit_error(exc)
                retryable = _is_retryable_error(exc)

                if retryable and attempt < max_attempts:
                    delay = self._config.gemini_retry_base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "Gemini API call failed on attempt %d/%d (%s). Retrying in %.2fs.",
                        attempt,
                        max_attempts,
                        safe_msg,
                        delay,
                    )
                    if delay > 0:
                        self._sleep(delay)
                    continue

                if is_rate_limit:
                    raise GeminiRateLimitError(
                        f"Gemini API rate limit or quota exceeded after {attempt} attempt(s): "
                        f"{safe_msg}"
                    ) from exc

                raise GeminiClientError(
                    f"Gemini API request failed: {safe_msg}"
                ) from exc

        # Defensive safeguard (loop always returns or raises above).
        safe_msg = sanitize_error_message(str(last_error or "Unknown error"), self._config.gemini_api_key)
        raise GeminiClientError(f"Gemini API request failed: {safe_msg}")

    @staticmethod
    def _parse_and_validate(raw_text: str) -> SpeechAnalytics:
        """Parse Gemini JSON string and validate against SpeechAnalytics."""
        cleaned = _strip_markdown_fences(raw_text)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise GeminiResponseParseError(
                f"Gemini returned malformed JSON: {exc.msg} (line {exc.lineno}, col {exc.colno})."
            ) from exc

        if not isinstance(payload, dict):
            raise SchemaValidationError(
                "Gemini JSON root must be an object matching SpeechAnalytics."
            )

        try:
            return SpeechAnalytics.model_validate(payload)
        except ValidationError as exc:
            raise SchemaValidationError(
                f"Gemini response failed Pydantic SpeechAnalytics validation: {exc.error_count()} error(s)."
            ) from exc
