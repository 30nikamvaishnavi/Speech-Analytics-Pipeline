"""Environment configuration and secret-safe logging utilities."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from app.exceptions import ConfigurationError

WhisperModelSize = Literal["tiny", "base", "small"]
WhisperDeviceOption = Literal["auto", "cpu", "cuda"]

SUPPORTED_WHISPER_MODELS: tuple[str, ...] = ("tiny", "base", "small")
SUPPORTED_WHISPER_DEVICES: tuple[str, ...] = ("auto", "cpu", "cuda")
DEFAULT_GEMINI_MODEL = "gemini-4-hydrogen"
DEFAULT_WHISPER_MODEL: WhisperModelSize = "base"
DEFAULT_WHISPER_DEVICE: WhisperDeviceOption = "auto"
DEFAULT_OUTPUT_DIR = "results"

_PLACEHOLDER_KEYS = {
    "",
    "your_gemini_api_key_here",
    "your_actual_api_key",
    "my_gemini_api_key",
    "changeme",
}

_PLACEHOLDER_MODELS = {
    "",
    "your_supported_gemini_model",
}


class SecretRedactingFilter(logging.Filter):
    """Logging filter that redacts known secret values from log records."""

    def __init__(self, secrets: list[str] | None = None) -> None:
        super().__init__()
        self._secrets = [s for s in (secrets or []) if s and len(s) >= 6]

    def update_secrets(self, secrets: list[str]) -> None:
        """Update the list of secret strings to redact."""
        self._secrets = [s for s in secrets if s and len(s) >= 6]

    def filter(self, record: logging.LogRecord) -> bool:
        if not self._secrets:
            return True
        message = record.getMessage()
        redacted = message
        for secret in self._secrets:
            if secret in redacted:
                redacted = redacted.replace(secret, "***REDACTED***")
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


def configure_logging(
    level: int = logging.INFO,
    secrets_to_redact: list[str] | None = None,
) -> logging.Logger:
    """Configure structured package logging with secret redaction."""
    logger = logging.getLogger("app")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    redacting_filter = SecretRedactingFilter(secrets_to_redact)
    for handler in logger.handlers:
        handler.filters = [f for f in handler.filters if not isinstance(f, SecretRedactingFilter)]
        handler.addFilter(redacting_filter)
    return logger


def sanitize_error_message(message: str, api_key: str | None = None) -> str:
    """Remove any accidental API key occurrences from exception messages."""
    sanitized = message
    if api_key and len(api_key) >= 6 and api_key in sanitized:
        sanitized = sanitized.replace(api_key, "***REDACTED***")
    env_key = os.getenv("GEMINI_API_KEY", "")
    if env_key and len(env_key) >= 6 and env_key in sanitized:
        sanitized = sanitized.replace(env_key, "***REDACTED***")
    return sanitized


def _load_dotenv_safe(env_file: str | Path | None = None) -> None:
    """Load variables from .env using python-dotenv if available."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - defensive fallback if dotenv not installed
        return

    if env_file is not None:
        load_dotenv(dotenv_path=Path(env_file), override=False)
    else:
        load_dotenv(override=False)


@dataclass(slots=True)
class AppConfig:
    """Application configuration loaded from environment variables and .env.

    The Gemini API key is intentionally omitted from ``repr`` output and
    never exposed in logs or exports.
    """

    gemini_api_key: str | None = field(default=None, repr=False)
    gemini_model: str = DEFAULT_GEMINI_MODEL
    whisper_model: WhisperModelSize = DEFAULT_WHISPER_MODEL
    whisper_device: WhisperDeviceOption = DEFAULT_WHISPER_DEVICE
    output_dir: Path = field(default_factory=lambda: Path(DEFAULT_OUTPUT_DIR))
    gemini_max_retries: int = 3
    gemini_retry_base_delay: float = 1.0

    def __post_init__(self) -> None:
        if self.whisper_model not in SUPPORTED_WHISPER_MODELS:
            raise ConfigurationError(
                f"Unsupported WHISPER_MODEL '{self.whisper_model}'. "
                f"Choose one of: {', '.join(SUPPORTED_WHISPER_MODELS)}."
            )
        if self.whisper_device not in SUPPORTED_WHISPER_DEVICES:
            raise ConfigurationError(
                f"Unsupported WHISPER_DEVICE '{self.whisper_device}'. "
                f"Choose one of: {', '.join(SUPPORTED_WHISPER_DEVICES)}."
            )
        if self.gemini_max_retries < 0:
            raise ConfigurationError("GEMINI_MAX_RETRIES must be >= 0.")
        if self.gemini_retry_base_delay < 0:
            raise ConfigurationError("GEMINI_RETRY_BASE_DELAY must be >= 0.")
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

    @property
    def has_valid_gemini_key(self) -> bool:
        """Return True when a non-placeholder Gemini API key is configured."""
        if not self.gemini_api_key:
            return False
        normalized = self.gemini_api_key.strip().lower()
        return normalized not in _PLACEHOLDER_KEYS

    def validate_gemini_config(self) -> str:
        """Validate Gemini credentials and model before calling the Gemini API.

        Returns the validated API key. Raises ConfigurationError if invalid.
        """
        if not self.has_valid_gemini_key:
            raise ConfigurationError(
                "GEMINI_API_KEY is missing or still set to a placeholder. "
                "Set a valid GEMINI_API_KEY in your .env file before running analytics."
            )
        model_clean = (self.gemini_model or "").strip()
        if model_clean.lower() in _PLACEHOLDER_MODELS:
            raise ConfigurationError(
                "GEMINI_MODEL is missing or set to a placeholder. "
                "Set a supported model name (e.g., gemini-4-hydrogen) in .env."
            )
        assert self.gemini_api_key is not None
        return self.gemini_api_key.strip()

    def to_safe_dict(self) -> dict[str, str | int | float | bool]:
        """Return configuration metadata safe for UI display and exports (no secrets)."""
        return {
            "gemini_model": self.gemini_model,
            "gemini_api_key_configured": self.has_valid_gemini_key,
            "whisper_model": self.whisper_model,
            "whisper_device": self.whisper_device,
            "output_dir": str(self.output_dir),
            "gemini_max_retries": self.gemini_max_retries,
        }

    @classmethod
    def from_env(
        cls,
        env_file: str | Path | None = None,
        *,
        load_dotenv_file: bool = True,
    ) -> AppConfig:
        """Build AppConfig from environment variables (and optionally .env)."""
        if load_dotenv_file:
            _load_dotenv_safe(env_file)

        raw_key = os.getenv("GEMINI_API_KEY")
        api_key = raw_key.strip() if raw_key and raw_key.strip() else None

        raw_model = (os.getenv("GEMINI_MODEL") or "").strip()
        if not raw_model or raw_model.lower() in _PLACEHOLDER_MODELS:
            gemini_model = DEFAULT_GEMINI_MODEL
        else:
            gemini_model = raw_model

        whisper_model_raw = (os.getenv("WHISPER_MODEL") or DEFAULT_WHISPER_MODEL).strip().lower()
        whisper_device_raw = (os.getenv("WHISPER_DEVICE") or DEFAULT_WHISPER_DEVICE).strip().lower()
        output_dir_raw = (os.getenv("OUTPUT_DIR") or DEFAULT_OUTPUT_DIR).strip()

        max_retries_raw = os.getenv("GEMINI_MAX_RETRIES", "3").strip()
        retry_delay_raw = os.getenv("GEMINI_RETRY_BASE_DELAY", "1.0").strip()

        try:
            max_retries = int(max_retries_raw)
        except ValueError as exc:
            raise ConfigurationError("GEMINI_MAX_RETRIES must be an integer.") from exc

        try:
            retry_delay = float(retry_delay_raw)
        except ValueError as exc:
            raise ConfigurationError("GEMINI_RETRY_BASE_DELAY must be a number.") from exc

        config = cls(
            gemini_api_key=api_key,
            gemini_model=gemini_model,
            whisper_model=whisper_model_raw,  # type: ignore[arg-type]
            whisper_device=whisper_device_raw,  # type: ignore[arg-type]
            output_dir=Path(output_dir_raw),
            gemini_max_retries=max_retries,
            gemini_retry_base_delay=retry_delay,
        )
        configure_logging(secrets_to_redact=[api_key] if api_key else [])
        return config
