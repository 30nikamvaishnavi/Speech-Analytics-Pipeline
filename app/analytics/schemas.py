"""Strict Pydantic v2 schemas for structured Gemini speech analytics."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SentimentLabel = Literal["positive", "negative", "neutral", "mixed"]
VerificationStatus = Literal["verified", "unverified"]


def _validate_confidence_value(value: Any) -> float | None:
    """Validate optional confidence score in [0.0, 1.0], rejecting booleans and NaN/inf."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Confidence must be a float between 0 and 1, not a boolean.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Confidence must be a numeric value between 0 and 1.") from exc
    if math.isnan(numeric) or math.isinf(numeric):
        raise ValueError("Confidence must be a finite number between 0 and 1.")
    if numeric < 0.0 or numeric > 1.0:
        raise ValueError("Confidence must be between 0.0 and 1.0 inclusive.")
    return round(numeric, 4)


def _normalize_optional_string(value: Any) -> str | None:
    """Convert blank/whitespace-only optional strings to None."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Value must be a string or null.")
    stripped = value.strip()
    return stripped if stripped else None


class Sentiment(BaseModel):
    """Overall sentiment classification supported by the call transcript."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    label: SentimentLabel
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    explanation: str = Field(..., min_length=1, max_length=1000)

    @field_validator("confidence", mode="before")
    @classmethod
    def _check_confidence(cls, value: Any) -> float | None:
        return _validate_confidence_value(value)


class Topic(BaseModel):
    """Discussion topic grounded in a supporting transcript quote."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(..., min_length=1, max_length=150)
    description: str = Field(..., min_length=1, max_length=600)
    supporting_quote: str = Field(..., min_length=1, max_length=1000)


class PersonEntity(BaseModel):
    """Person or named entity explicitly mentioned in the transcript."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(..., min_length=1, max_length=150)
    role: str | None = Field(default=None, max_length=150)
    evidence_quote: str = Field(..., min_length=1, max_length=1000)

    @field_validator("role", mode="before")
    @classmethod
    def _clean_role(cls, value: Any) -> str | None:
        return _normalize_optional_string(value)


class ActionItem(BaseModel):
    """Actionable task extracted from the transcript with verification status."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    task: str = Field(..., min_length=1, max_length=500)
    owner: str | None = Field(default=None, max_length=150)
    deadline: str | None = Field(default=None, max_length=150)
    evidence_quote: str = Field(..., min_length=1, max_length=1000)
    verification_status: VerificationStatus = Field(default="unverified")

    @field_validator("owner", "deadline", mode="before")
    @classmethod
    def _clean_optional_fields(cls, value: Any) -> str | None:
        return _normalize_optional_string(value)


class SpeechAnalytics(BaseModel):
    """Top-level structured speech analytics schema validated via Pydantic v2."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    summary: str = Field(..., min_length=1, max_length=3000)
    sentiment: Sentiment
    topics: list[Topic] = Field(default_factory=list, max_length=25)
    entities: list[PersonEntity] = Field(default_factory=list, max_length=50)
    action_items: list[ActionItem] = Field(default_factory=list, max_length=50)
    overall_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("overall_confidence", mode="before")
    @classmethod
    def _check_overall_confidence(cls, value: Any) -> float | None:
        return _validate_confidence_value(value)
