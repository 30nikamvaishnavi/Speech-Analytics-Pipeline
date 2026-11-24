"""Deterministic transcript evidence verification layer.

This module checks Gemini-extracted people, entities, action items, owners,
and deadlines against the original Whisper transcript using deterministic,
Unicode-aware matching rules. It never deletes unverified items; instead, it
preserves all extracted records and flags unsupported claims clearly.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.analytics.schemas import ActionItem, PersonEntity, SpeechAnalytics
from app.validation.validators import (
    contains_negation,
    extract_meaningful_tokens,
    find_exact_phrase_match,
    normalize_unicode_text,
    verify_quote_in_transcript,
)

EVIDENCE_LIMITATIONS_NOTE = (
    "Evidence verification checks whether extracted names, quotes, tasks, owners, "
    "and deadlines are lexically supported by the Whisper transcript. It is a "
    "hallucination-reduction safeguard, not a guarantee that the audio transcription "
    "or semantic interpretation is factually accurate."
)


class VerificationRecord(BaseModel):
    """Audit record describing deterministic verification for one extracted claim."""

    model_config = ConfigDict(extra="forbid")

    item_type: Literal["entity", "action_item", "owner", "deadline"]
    extracted_value: str
    evidence_quote: str | None = None
    verification_status: Literal["verified", "unverified"]
    verification_reason: str
    matched_excerpt: str | None = None


class EvidenceValidationReport(BaseModel):
    """Consolidated evidence validation summary across entities and action items."""

    model_config = ConfigDict(extra="forbid")

    schema_valid: bool = True
    overall_status: Literal[
        "verified",
        "partially_verified",
        "unverified",
        "no_claims_to_verify",
    ]
    total_entities: int = Field(default=0, ge=0)
    verified_entities: int = Field(default=0, ge=0)
    unverified_entities: int = Field(default=0, ge=0)
    total_action_items: int = Field(default=0, ge=0)
    verified_action_items: int = Field(default=0, ge=0)
    unverified_action_items: int = Field(default=0, ge=0)
    records: list[VerificationRecord] = Field(default_factory=list)
    limitations_note: str = EVIDENCE_LIMITATIONS_NOTE


class EvidenceChecker:
    """Conservative deterministic evidence checker for SpeechAnalytics outputs."""

    def __init__(self, *, min_task_token_overlap: float = 0.4) -> None:
        if not 0.0 < min_task_token_overlap <= 1.0:
            raise ValueError("min_task_token_overlap must be in (0.0, 1.0].")
        self.min_task_token_overlap = min_task_token_overlap

    def verify_entity(self, entity: PersonEntity, transcript: str) -> VerificationRecord:
        """Verify that an extracted person or named entity appears in the transcript."""
        matched_name = find_exact_phrase_match(entity.name, transcript)
        if matched_name is None:
            return VerificationRecord(
                item_type="entity",
                extracted_value=entity.name,
                evidence_quote=entity.evidence_quote,
                verification_status="unverified",
                verification_reason=(
                    f"Entity '{entity.name}' was not found as a whole name/token in the transcript."
                ),
                matched_excerpt=None,
            )

        quote_found, quote_excerpt, _ = verify_quote_in_transcript(
            entity.evidence_quote, transcript
        )
        if not quote_found:
            return VerificationRecord(
                item_type="entity",
                extracted_value=entity.name,
                evidence_quote=entity.evidence_quote,
                verification_status="unverified",
                verification_reason=(
                    f"Entity '{entity.name}' appears in transcript, but its evidence_quote "
                    "was not found verbatim."
                ),
                matched_excerpt=matched_name,
            )

        # Ensure the evidence quote itself mentions the entity (prevents mismatched quotes).
        if find_exact_phrase_match(entity.name, entity.evidence_quote) is None:
            return VerificationRecord(
                item_type="entity",
                extracted_value=entity.name,
                evidence_quote=entity.evidence_quote,
                verification_status="unverified",
                verification_reason=(
                    f"Entity '{entity.name}' appears in transcript, but is not mentioned "
                    "inside the supplied evidence_quote."
                ),
                matched_excerpt=matched_name,
            )

        return VerificationRecord(
            item_type="entity",
            extracted_value=entity.name,
            evidence_quote=entity.evidence_quote,
            verification_status="verified",
            verification_reason="Entity name and evidence_quote verified in transcript.",
            matched_excerpt=quote_excerpt or matched_name,
        )

    def _is_task_supported_by_quote(
        self, task: str, evidence_quote: str
    ) -> tuple[bool, str]:
        """Conservatively verify that ``evidence_quote`` supports ``task``."""
        cleaned_quote = normalize_unicode_text(evidence_quote)
        if len(cleaned_quote) < 4:
            return False, "Evidence quote is too short to substantiate an action item."

        # Reject affirmative tasks paired with negated quotes.
        if contains_negation(evidence_quote) and not contains_negation(task):
            return (
                False,
                "Evidence quote expresses negation or refusal that conflicts with the task.",
            )

        task_tokens = extract_meaningful_tokens(task)
        quote_tokens = set(extract_meaningful_tokens(evidence_quote))

        if not task_tokens:
            return False, "Task contains insufficient meaningful tokens for verification."

        overlap = [tok for tok in task_tokens if tok in quote_tokens]
        unique_task_tokens = list(dict.fromkeys(task_tokens))
        unique_overlap = list(dict.fromkeys(overlap))

        if len(unique_task_tokens) == 1:
            required_matches = 1
        elif len(unique_task_tokens) == 2:
            required_matches = 2
        else:
            required_matches = max(
                2, math.ceil(len(unique_task_tokens) * self.min_task_token_overlap)
            )

        if len(unique_overlap) < required_matches:
            return (
                False,
                "Evidence quote appears in transcript but does not sufficiently support "
                f"the extracted task (matched {len(unique_overlap)}/{len(unique_task_tokens)} "
                "key task terms).",
            )

        return True, "Evidence quote appears in transcript and supports the extracted task."

    def _verify_owner(
        self,
        owner: str,
        evidence_quote: str,
        quote_context: str | None,
        transcript: str,
    ) -> VerificationRecord:
        """Independently verify an action item owner against transcript and quote context."""
        transcript_match = find_exact_phrase_match(owner, transcript)
        if transcript_match is None:
            return VerificationRecord(
                item_type="owner",
                extracted_value=owner,
                evidence_quote=evidence_quote,
                verification_status="unverified",
                verification_reason=f"Owner '{owner}' was not found in the transcript.",
                matched_excerpt=None,
            )

        in_quote = find_exact_phrase_match(owner, evidence_quote)
        in_context = (
            find_exact_phrase_match(owner, quote_context) if quote_context else None
        )
        if in_quote is None and in_context is None:
            return VerificationRecord(
                item_type="owner",
                extracted_value=owner,
                evidence_quote=evidence_quote,
                verification_status="unverified",
                verification_reason=(
                    f"Owner '{owner}' appears in transcript, but is not associated with "
                    "the action item's evidence quote context."
                ),
                matched_excerpt=transcript_match,
            )

        return VerificationRecord(
            item_type="owner",
            extracted_value=owner,
            evidence_quote=evidence_quote,
            verification_status="verified",
            verification_reason=f"Owner '{owner}' verified in action item quote context.",
            matched_excerpt=in_quote or in_context or transcript_match,
        )

    def _verify_deadline(
        self,
        deadline: str,
        evidence_quote: str,
        quote_context: str | None,
        transcript: str,
    ) -> VerificationRecord:
        """Independently verify an action item deadline against transcript and quote context."""
        exact_transcript = find_exact_phrase_match(deadline, transcript)
        deadline_tokens = extract_meaningful_tokens(deadline)

        if exact_transcript is None:
            # Check if multi-word deadline tokens are absent from transcript.
            transcript_tokens = set(extract_meaningful_tokens(transcript))
            if not deadline_tokens or not all(t in transcript_tokens for t in deadline_tokens):
                return VerificationRecord(
                    item_type="deadline",
                    extracted_value=deadline,
                    evidence_quote=evidence_quote,
                    verification_status="unverified",
                    verification_reason=(
                        f"Deadline '{deadline}' is not explicitly supported in the transcript."
                    ),
                    matched_excerpt=None,
                )

        # Verify deadline appears in the quote or immediate surrounding context.
        in_quote = find_exact_phrase_match(deadline, evidence_quote)
        in_context = (
            find_exact_phrase_match(deadline, quote_context) if quote_context else None
        )
        if in_quote is None and in_context is None:
            context_tokens = set(extract_meaningful_tokens(quote_context or evidence_quote))
            if not deadline_tokens or not all(t in context_tokens for t in deadline_tokens):
                return VerificationRecord(
                    item_type="deadline",
                    extracted_value=deadline,
                    evidence_quote=evidence_quote,
                    verification_status="unverified",
                    verification_reason=(
                        f"Deadline '{deadline}' is not verified in the action item's "
                        "evidence quote context."
                    ),
                    matched_excerpt=exact_transcript,
                )

        return VerificationRecord(
            item_type="deadline",
            extracted_value=deadline,
            evidence_quote=evidence_quote,
            verification_status="verified",
            verification_reason=f"Deadline '{deadline}' verified in action item quote context.",
            matched_excerpt=in_quote or in_context or exact_transcript,
        )

    def verify_action_item(
        self, item: ActionItem, transcript: str
    ) -> tuple[ActionItem, list[VerificationRecord]]:
        """Verify an action item, its quote, owner, and deadline conservatively."""
        records: list[VerificationRecord] = []

        quote_found, matched_quote, quote_context = verify_quote_in_transcript(
            item.evidence_quote, transcript
        )

        failure_reasons: list[str] = []
        if not quote_found:
            failure_reasons.append("Evidence quote was not found verbatim in the transcript.")
        else:
            task_ok, task_reason = self._is_task_supported_by_quote(
                item.task, item.evidence_quote
            )
            if not task_ok:
                failure_reasons.append(task_reason)

        if item.owner is not None:
            owner_record = self._verify_owner(
                item.owner,
                item.evidence_quote,
                quote_context if quote_found else None,
                transcript,
            )
            records.append(owner_record)
            if owner_record.verification_status != "verified":
                failure_reasons.append(owner_record.verification_reason)

        if item.deadline is not None:
            deadline_record = self._verify_deadline(
                item.deadline,
                item.evidence_quote,
                quote_context if quote_found else None,
                transcript,
            )
            records.append(deadline_record)
            if deadline_record.verification_status != "verified":
                failure_reasons.append(deadline_record.verification_reason)

        is_verified = len(failure_reasons) == 0
        status: Literal["verified", "unverified"] = "verified" if is_verified else "unverified"
        reason = (
            "Task, evidence quote, and explicit metadata verified against transcript."
            if is_verified
            else " ".join(failure_reasons)
        )

        action_record = VerificationRecord(
            item_type="action_item",
            extracted_value=item.task,
            evidence_quote=item.evidence_quote,
            verification_status=status,
            verification_reason=reason,
            matched_excerpt=matched_quote,
        )
        # Put the primary action_item record first, followed by owner/deadline sub-records.
        ordered_records = [action_record, *records]

        updated_item = item.model_copy(update={"verification_status": status})
        return updated_item, ordered_records

    def validate(
        self,
        analytics: SpeechAnalytics,
        transcript: str,
        *,
        schema_valid: bool = True,
    ) -> tuple[SpeechAnalytics, EvidenceValidationReport]:
        """Validate all entities and action items against the transcript.

        Returns an updated ``SpeechAnalytics`` copy (preserving all records and
        setting deterministic ``verification_status`` values) alongside an
        ``EvidenceValidationReport``.
        """
        records: list[VerificationRecord] = []

        verified_entities_count = 0
        for entity in analytics.entities:
            entity_record = self.verify_entity(entity, transcript)
            records.append(entity_record)
            if entity_record.verification_status == "verified":
                verified_entities_count += 1

        updated_action_items: list[ActionItem] = []
        verified_actions_count = 0
        for item in analytics.action_items:
            updated_item, item_records = self.verify_action_item(item, transcript)
            updated_action_items.append(updated_item)
            records.extend(item_records)
            if updated_item.verification_status == "verified":
                verified_actions_count += 1

        total_entities = len(analytics.entities)
        unverified_entities_count = total_entities - verified_entities_count
        total_actions = len(updated_action_items)
        unverified_actions_count = total_actions - verified_actions_count

        total_claims = total_entities + total_actions
        total_verified = verified_entities_count + verified_actions_count

        if total_claims == 0:
            overall_status: Literal[
                "verified",
                "partially_verified",
                "unverified",
                "no_claims_to_verify",
            ] = "no_claims_to_verify"
        elif total_verified == total_claims:
            overall_status = "verified"
        elif total_verified == 0:
            overall_status = "unverified"
        else:
            overall_status = "partially_verified"

        updated_analytics = analytics.model_copy(
            update={"action_items": updated_action_items}
        )
        report = EvidenceValidationReport(
            schema_valid=schema_valid,
            overall_status=overall_status,
            total_entities=total_entities,
            verified_entities=verified_entities_count,
            unverified_entities=unverified_entities_count,
            total_action_items=total_actions,
            verified_action_items=verified_actions_count,
            unverified_action_items=unverified_actions_count,
            records=records,
            limitations_note=EVIDENCE_LIMITATIONS_NOTE,
        )
        return updated_analytics, report
