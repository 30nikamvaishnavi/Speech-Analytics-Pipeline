"""Unicode-aware text normalization and boundary-safe matching primitives."""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_SPACE_RE = re.compile(r"\s+([,.;:!?।])")
_QUOTE_CHARS = "\"'“”‘’`"

# Common non-content stopwords across English, Hindi (Devanagari), and Marathi (Devanagari).
_STOPWORDS: frozenset[str] = frozenset(
    {
        # English
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "he",
        "her",
        "his",
        "i",
        "in",
        "is",
        "it",
        "its",
        "me",
        "must",
        "my",
        "need",
        "needs",
        "of",
        "on",
        "or",
        "our",
        "please",
        "she",
        "should",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "us",
        "we",
        "will",
        "with",
        "you",
        "your",
        "action",
        "item",
        "task",
        # Hindi
        "और",
        "का",
        "की",
        "के",
        "को",
        "पर",
        "में",
        "से",
        "है",
        "हैं",
        "था",
        "थे",
        "थी",
        "कर",
        "करना",
        "करने",
        "करें",
        "चाहिए",
        "यह",
        "वह",
        "हम",
        "आप",
        "मैं",
        # Marathi
        "आणि",
        "आहे",
        "आहेत",
        "चा",
        "ची",
        "चे",
        "ला",
        "ना",
        "ने",
        "ते",
        "मध्ये",
        "वर",
        "पासून",
        "करा",
        "करावे",
        "करायचे",
        "मी",
        "आम्ही",
        "तुम्ही",
    }
)

_NEGATION_MARKERS: frozenset[str] = frozenset(
    {
        "not",
        "no",
        "never",
        "cannot",
        "cant",
        "can't",
        "wont",
        "won't",
        "dont",
        "don't",
        "didnt",
        "didn't",
        "refuse",
        "refused",
        "decline",
        "declined",
        "नहीं",
        "मत",
        "नाही",
        "नको",
    }
)


def is_word_char(char: str) -> bool:
    """Return True if char is a Unicode letter, number, combining mark, or underscore.

    Including Unicode combining marks (category ``M*``) prevents false word
    boundaries inside Devanagari words containing matras or virama.
    """
    if not char:
        return False
    if char == "_":
        return True
    category = unicodedata.category(char)
    return category[0] in ("L", "N", "M")


def normalize_unicode_text(text: str) -> str:
    """Normalize text to NFC Unicode form and collapse repeated whitespace."""
    nfc = unicodedata.normalize("NFC", text)
    return _WHITESPACE_RE.sub(" ", nfc).strip()


def normalize_for_matching(text: str) -> str:
    """Normalize Unicode NFC, collapse whitespace, and casefold for matching."""
    return normalize_unicode_text(text).casefold()


def find_exact_phrase_match(needle: str, haystack: str) -> str | None:
    """Find ``needle`` inside ``haystack`` respecting Unicode word boundaries.

    Rejects partial-word matches such as matching ``"Ana"`` inside ``"Ananya"``
    or ``"रोह"`` inside ``"रोहिता"``. Returns the matched excerpt from
    ``haystack`` (preserving original Unicode casing) or ``None`` if absent.
    """
    needle_norm = normalize_unicode_text(needle)
    haystack_norm = normalize_unicode_text(haystack)
    if not needle_norm or not haystack_norm:
        return None

    needle_folded = needle_norm.casefold()
    haystack_folded = haystack_norm.casefold()
    same_length = len(haystack_norm) == len(haystack_folded)

    start_idx = 0
    needle_len = len(needle_folded)
    haystack_len = len(haystack_folded)

    while start_idx <= haystack_len - needle_len:
        idx = haystack_folded.find(needle_folded, start_idx)
        if idx == -1:
            return None

        end_idx = idx + needle_len
        left_ok = idx == 0 or not is_word_char(haystack_folded[idx - 1])
        right_ok = end_idx == haystack_len or not is_word_char(haystack_folded[end_idx])

        if left_ok and right_ok:
            if same_length:
                return haystack_norm[idx:end_idx]
            return haystack_folded[idx:end_idx]

        start_idx = idx + 1

    return None


def verify_quote_in_transcript(
    quote: str,
    transcript: str,
    *,
    context_chars: int = 80,
) -> tuple[bool, str | None, str | None]:
    """Verify whether ``quote`` appears verbatim in ``transcript``.

    Returns ``(is_found, matched_excerpt, local_context_window)``.
    """
    cleaned_quote = normalize_unicode_text(quote).strip(_QUOTE_CHARS).strip()
    transcript_norm = normalize_unicode_text(transcript)
    if not cleaned_quote or not transcript_norm:
        return False, None, None

    quote_folded = cleaned_quote.casefold()
    transcript_folded = transcript_norm.casefold()

    idx = transcript_folded.find(quote_folded)
    if idx != -1:
        end_idx = idx + len(quote_folded)
        same_len = len(transcript_norm) == len(transcript_folded)
        excerpt = (
            transcript_norm[idx:end_idx] if same_len else transcript_folded[idx:end_idx]
        )
        ctx_start = max(0, idx - context_chars)
        ctx_end = min(len(transcript_folded), end_idx + context_chars)
        context = (
            transcript_norm[ctx_start:ctx_end]
            if same_len
            else transcript_folded[ctx_start:ctx_end]
        )
        return True, excerpt, context

    # Secondary pass: normalize minor punctuation spacing (e.g., "word ," -> "word,").
    quote_punct = _PUNCT_SPACE_RE.sub(r"\1", quote_folded)
    transcript_punct = _PUNCT_SPACE_RE.sub(r"\1", transcript_folded)
    idx_p = transcript_punct.find(quote_punct)
    if idx_p != -1:
        end_p = idx_p + len(quote_punct)
        ctx_start = max(0, idx_p - context_chars)
        ctx_end = min(len(transcript_punct), end_p + context_chars)
        return True, transcript_punct[idx_p:end_p], transcript_punct[ctx_start:ctx_end]

    return False, None, None


def _tokenize_unicode_words(text: str) -> list[str]:
    """Split normalized text into Unicode word tokens preserving combining marks."""
    norm = normalize_for_matching(text)
    tokens: list[str] = []
    current: list[str] = []
    for ch in norm:
        if is_word_char(ch):
            current.append(ch)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def _light_normalize_token(token: str) -> str:
    """Apply conservative ASCII inflection normalization without stripping Unicode roots."""
    if not token.isascii() or len(token) <= 3:
        return token
    for suffix in ("ing", "ied", "ies", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            if suffix == "ies":
                return token[:-3] + "y"
            if suffix == "ied":
                return token[:-3] + "y"
            return token[: -len(suffix)]
    return token


def extract_meaningful_tokens(text: str) -> list[str]:
    """Extract normalized, non-stopword Unicode content tokens from text."""
    raw_tokens = _tokenize_unicode_words(text)
    meaningful: list[str] = []
    for token in raw_tokens:
        if token in _STOPWORDS:
            continue
        if len(token) <= 1 and not token.isdigit():
            continue
        meaningful.append(_light_normalize_token(token))
    return meaningful


def contains_negation(text: str) -> bool:
    """Return True if text contains explicit negation or refusal markers."""
    tokens = set(_tokenize_unicode_words(text))
    if tokens & _NEGATION_MARKERS:
        return True
    lowered = normalize_for_matching(text)
    return any(phrase in lowered for phrase in ("will not", "going to not", "won't", "can't"))
