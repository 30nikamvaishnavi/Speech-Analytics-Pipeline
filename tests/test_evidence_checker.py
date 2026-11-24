"""Tests for deterministic Unicode-aware evidence verification."""

from __future__ import annotations

from app.analytics.schemas import ActionItem, PersonEntity, Sentiment, SpeechAnalytics
from app.validation.evidence_checker import EvidenceChecker


def _make_analytics(
    *,
    entities: list[PersonEntity] | None = None,
    action_items: list[ActionItem] | None = None,
) -> SpeechAnalytics:
    return SpeechAnalytics(
        summary="Call summary.",
        sentiment=Sentiment(label="neutral", explanation="Balanced call."),
        topics=[],
        entities=entities or [],
        action_items=action_items or [],
    )


def test_entity_exists_in_transcript_verified() -> None:
    transcript = "Ananya Verma joined the call and confirmed the budget numbers."
    checker = EvidenceChecker()
    entity = PersonEntity(
        name="Ananya Verma",
        role="Finance Lead",
        evidence_quote="Ananya Verma joined the call",
    )
    record = checker.verify_entity(entity, transcript)
    assert record.verification_status == "verified"
    assert record.matched_excerpt is not None


def test_entity_absent_from_transcript_unverified() -> None:
    transcript = "We reviewed the server logs and planned the deployment."
    checker = EvidenceChecker()
    entity = PersonEntity(
        name="Vikram Rao",
        role=None,
        evidence_quote="Vikram Rao will deploy",
    )
    record = checker.verify_entity(entity, transcript)
    assert record.verification_status == "unverified"


def test_english_case_insensitive_entity_matching() -> None:
    transcript = "priya sharma will coordinate with the vendor today."
    checker = EvidenceChecker()
    entity = PersonEntity(
        name="Priya Sharma",
        evidence_quote="PRIYA SHARMA will coordinate with the vendor",
    )
    record = checker.verify_entity(entity, transcript)
    assert record.verification_status == "verified"


def test_unicode_hindi_and_marathi_name_matching() -> None:
    transcript = "रोहित शर्मा ने कहा कि स्नेहा कुलकर्णी उद्या अहवाल सादर करतील."
    checker = EvidenceChecker()
    hindi_entity = PersonEntity(
        name="रोहित शर्मा",
        evidence_quote="रोहित शर्मा ने कहा",
    )
    marathi_entity = PersonEntity(
        name="स्नेहा कुलकर्णी",
        evidence_quote="स्नेहा कुलकर्णी उद्या अहवाल सादर करतील",
    )
    assert checker.verify_entity(hindi_entity, transcript).verification_status == "verified"
    assert checker.verify_entity(marathi_entity, transcript).verification_status == "verified"


def test_partial_name_false_positives_rejected() -> None:
    transcript = "Ananya and Christopher spoke with रोहिता about the release."
    checker = EvidenceChecker()

    ana = PersonEntity(name="Ana", evidence_quote="Ananya and Christopher")
    chris = PersonEntity(name="Chris", evidence_quote="Ananya and Christopher")
    rohit = PersonEntity(name="रोहित", evidence_quote="रोहिता about the release")

    assert checker.verify_entity(ana, transcript).verification_status == "unverified"
    assert checker.verify_entity(chris, transcript).verification_status == "unverified"
    assert checker.verify_entity(rohit, transcript).verification_status == "unverified"


def test_supported_action_item_verified() -> None:
    transcript = (
        "Rahul: Let's finalize the onboarding deck. "
        "Meera: I will send the onboarding deck to HR by Friday."
    )
    checker = EvidenceChecker()
    item = ActionItem(
        task="Send the onboarding deck to HR",
        owner="Meera",
        deadline="Friday",
        evidence_quote="Meera: I will send the onboarding deck to HR by Friday.",
        verification_status="unverified",
    )
    updated, records = checker.verify_action_item(item, transcript)
    assert updated.verification_status == "verified"
    assert all(rec.verification_status == "verified" for rec in records)


def test_evidence_quote_absent_marks_action_item_unverified() -> None:
    transcript = "We had a quick discussion about customer feedback."
    checker = EvidenceChecker()
    item = ActionItem(
        task="Send survey link to customers",
        owner=None,
        deadline=None,
        evidence_quote="I will email the survey link tomorrow.",
    )
    updated, records = checker.verify_action_item(item, transcript)
    assert updated.verification_status == "unverified"
    assert records[0].verification_status == "unverified"


def test_quote_exists_but_does_not_support_task_marks_unverified() -> None:
    transcript = "Good morning everyone, glad we could connect on this call."
    checker = EvidenceChecker()
    item = ActionItem(
        task="Approve the enterprise security audit",
        owner=None,
        deadline=None,
        evidence_quote="Good morning everyone, glad we could connect on this call.",
    )
    updated, _ = checker.verify_action_item(item, transcript)
    assert updated.verification_status == "unverified"


def test_negated_quote_does_not_verify_affirmative_task() -> None:
    transcript = "Arjun: I cannot send the signed contract this week."
    checker = EvidenceChecker()
    item = ActionItem(
        task="Send the signed contract",
        owner="Arjun",
        deadline=None,
        evidence_quote="Arjun: I cannot send the signed contract this week.",
    )
    updated, _ = checker.verify_action_item(item, transcript)
    assert updated.verification_status == "unverified"


def test_unsupported_owner_marks_action_item_unverified() -> None:
    transcript = "Priya opened the call. Later, Sam said: I will update the billing sheet."
    checker = EvidenceChecker()

    # Owner absent from transcript completely
    missing_owner_item = ActionItem(
        task="Update the billing sheet",
        owner="Neha",
        deadline=None,
        evidence_quote="I will update the billing sheet.",
    )
    res1, recs1 = checker.verify_action_item(missing_owner_item, transcript)
    assert res1.verification_status == "unverified"
    assert any(r.item_type == "owner" and r.verification_status == "unverified" for r in recs1)

    # Owner mentioned elsewhere, not linked to the action item quote
    wrong_owner_item = ActionItem(
        task="Update the billing sheet",
        owner="Priya",
        deadline=None,
        evidence_quote="I will update the billing sheet.",
    )
    res2, recs2 = checker.verify_action_item(wrong_owner_item, transcript)
    assert res2.verification_status == "unverified"
    assert any(r.item_type == "owner" and r.verification_status == "unverified" for r in recs2)


def test_unsupported_deadline_marks_action_item_unverified() -> None:
    transcript = "Karan: I will prepare the release notes."
    checker = EvidenceChecker()
    item = ActionItem(
        task="Prepare the release notes",
        owner="Karan",
        deadline="Next Monday",
        evidence_quote="Karan: I will prepare the release notes.",
    )
    updated, records = checker.verify_action_item(item, transcript)
    assert updated.verification_status == "unverified"
    assert any(
        r.item_type == "deadline" and r.verification_status == "unverified" for r in records
    )


def test_conservative_validation_preserves_unverified_records() -> None:
    transcript = "Meera: I will share the invoice."
    analytics = _make_analytics(
        entities=[
            PersonEntity(name="Meera", evidence_quote="Meera: I will share the invoice."),
            PersonEntity(name="GhostPerson", evidence_quote="GhostPerson agreed."),
        ],
        action_items=[
            ActionItem(
                task="Share the invoice",
                owner="Meera",
                evidence_quote="Meera: I will share the invoice.",
            ),
            ActionItem(
                task="Delete production database",
                owner="GhostPerson",
                deadline="Tomorrow",
                evidence_quote="Fabricated quote.",
            ),
        ],
    )

    checker = EvidenceChecker()
    updated, report = checker.validate(analytics, transcript)

    # All items must be preserved (never silently deleted).
    assert len(updated.entities) == 2
    assert len(updated.action_items) == 2
    assert report.schema_valid is True
    assert report.overall_status == "partially_verified"
    assert report.verified_entities == 1
    assert report.unverified_entities == 1
    assert report.verified_action_items == 1
    assert report.unverified_action_items == 1
    assert updated.action_items[0].verification_status == "verified"
    assert updated.action_items[1].verification_status == "unverified"
