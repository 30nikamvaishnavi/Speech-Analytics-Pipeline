"""Domain-specific exception hierarchy for Speech-Analytics-Pipeline."""


class SpeechAnalyticsError(Exception):
    """Base exception for all pipeline errors."""


class ConfigurationError(SpeechAnalyticsError):
    """Raised when required configuration or environment variables are invalid or missing."""


class AudioValidationError(SpeechAnalyticsError):
    """Raised when an input audio file fails validation checks."""


class TranscriptionError(SpeechAnalyticsError):
    """Raised when local Whisper transcription fails."""


class GeminiClientError(SpeechAnalyticsError):
    """Raised when communicating with the Gemini API fails."""


class GeminiRateLimitError(GeminiClientError):
    """Raised when the Gemini API reports rate limiting or quota exhaustion."""


class GeminiResponseParseError(GeminiClientError):
    """Raised when the Gemini response is empty or contains malformed JSON."""


class SchemaValidationError(SpeechAnalyticsError):
    """Raised when structured analytics fail Pydantic schema validation."""


class EvidenceValidationError(SpeechAnalyticsError):
    """Raised when deterministic evidence validation encounters an unrecoverable error."""


class ExportError(SpeechAnalyticsError):
    """Raised when exporting reports to CSV or JSON fails."""
