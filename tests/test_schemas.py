"""Tests for strict Pydantic v2 speech analytics schemas."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.analytics.schemas import (
    ActionItem,
    PersonEntity,
    Sentiment,
    SpeechAnalytics,
    Topic,
)


def test_valid_sentiment_labels() -> None:
    for label in ("positive", "negative", "neutral", "mixed"):
        sentiment = Sentiment(
            label=label,  # type: ignore[arg-type]
            confidence=0.85,
            explanation="Grounded in speaker tone and statements.",
        )
        assert sentiment.label == label
        assert sentiment.confidence == 0.85


def test_invalid_sentiment_label_rejected() -> None:
    with pytest.raises(ValidationError):
        Sentiment(
            label="ecstatic",  # type: ignore[arg-type]
            confidence=0.9,
            explanation="Invalid label.",
        )


def test_valid_action_item_and_defaults() -> None:
    item = ActionItem(
        task="Send the Q3 budget report",
        owner="Priya",
        deadline="Friday",
        evidence_quote="Priya: I will send the Q3 budget report by Friday.",
    )
    assert item.verification_status == "unverified"
    assert item.owner == "Priya"
    assert item.deadline == "Friday"

    item_blank_opt = ActionItem(
        task="Review draft",
        owner="   ",
        deadline="",
        evidence_quote="We need to review the draft.",
        verification_status="verified",
    )
    assert item_blank_opt.owner is None
    assert item_blank_opt.deadline is None
    assert item_blank_opt.verification_status == "verified"


def test_missing_required_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        SpeechAnalytics.model_validate(
            {
                "summary": "Call summary",
                # missing sentiment
            }
        )

    with pytest.raises(ValidationError):
        ActionItem.model_validate({"task": "Follow up"})  # missing evidence_quote


@pytest.mark.parametrize("bad_confidence", [-0.1, 1.01, float("nan"), float("inf"), True])
def test_invalid_confidence_values_rejected(bad_confidence: float | bool) -> None:
    with pytest.raises(ValidationError):
        Sentiment(
            label="neutral",
            confidence=bad_confidence,  # type: ignore[arg-type]
            explanation="Test explanation.",
        )

    with pytest.raises(ValidationError):
        SpeechAnalytics(
            summary="Summary",
            sentiment=Sentiment(label="neutral", explanation="Neutral tone."),
            overall_confidence=bad_confidence,  # type: ignore[arg-type]
        )


def test_empty_topics_and_entities_allowed() -> None:
    analytics = SpeechAnalytics(
        summary="Brief status check-in with no named entities or tasks.",
        sentiment=Sentiment(label="neutral", confidence=0.6, explanation="Brief factual call."),
        topics=[],
        entities=[],
        action_items=[],
        overall_confidence=None,
    )
    assert analytics.topics == []
    assert analytics.entities == []
    assert analytics.action_items == []


def test_unicode_hindi_marathi_and_json_serialization() -> None:
    analytics = SpeechAnalytics(
        summary="ग्राहक सेवा कॉल सारांश: बिलिंग त्रुटीवर चर्चा झाली.",
        sentiment=Sentiment(
            label="mixed",
            confidence=0.72,
            explanation="शुरुआत में ग्राहक नाराज थे, लेकिन समाधान के बाद संतुष्ट हुए।",
        ),
        topics=[
            Topic(
                name="बिलिंग समस्या",
                description="इनवॉइस में अतिरिक्त शुल्क की जांच।",
                supporting_quote="रोहित जी, कृपया इस बिल की जांच करें।",
            )
        ],
        entities=[
            PersonEntity(
                name="रोहित शर्मा",
                role="सपोर्ट लीड",
                evidence_quote="रोहित शर्मा कल तक अपडेट देंगे।",
            )
        ],
        action_items=[
            ActionItem(
                task="अपडेटेड बिल पाठवणे",
                owner="स्नेहा कुलकर्णी",
                deadline="शुक्रवार",
                evidence_quote="स्नेहा कुलकर्णी शुक्रवारपर्यंत अपडेटेड बिल पाठवतील.",
                verification_status="unverified",
            )
        ],
        overall_confidence=0.8,
    )

    json_str = analytics.model_dump_json()
    parsed = json.loads(json_str)
    assert parsed["entities"][0]["name"] == "रोहित शर्मा"
    assert parsed["action_items"][0]["owner"] == "स्नेहा कुलकर्णी"
    assert parsed["action_items"][0]["deadline"] == "शुक्रवार"
