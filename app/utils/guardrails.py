"""
Input & Output Guardrails.

Input:  Sanitizes user messages before processing.
Output: Ensures bot responses never leak internal details.
"""

import re
import logging

logger = logging.getLogger(__name__)

# ── Patterns that suggest prompt injection ────────────────────────
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"you\s+are\s+now\s+a",
    r"system\s*:\s*",
    r"<\s*script",
    r"javascript\s*:",
    r"forget\s+(everything|all)",
    r"new\s+instructions?\s*:",
    r"act\s+as\s+(a|an)\s+",
    r"do\s+anything\s+now",
    r"dan\s+mode",
    r"reveal\s+(your\s+)?prompt",
    r"what\s+are\s+your\s+instructions",
    r"ignore\s+above",
    r"disregard",
    r"pretend\s+you\s+are",
    r"[\u200B-\u200F\u202A-\u202E]",  # Unicode zero-width/direction overrides
]

# ── Internal terms that should never appear in user-facing text ──
INTERNAL_TERMS = [
    "Room-Ratesplan",
    "Booking_Sessions",
    "room_ratesplan",
    "booking_sessions",
    "ObjectId",
    "ratePlanId",
    "roomId",
    "RM_",
    "RP_",
    "baseAdults",
    "baseChildren",
    "maxOccupancy",
    "hotelId",
    "_id",
    "motor",
    "pymongo",
    "MongoDB",
    "ChromaDB",
    "LangGraph",
]


def sanitize_input(message: str, max_length: int = 500) -> str:
    """
    Clean and validate user input.

    - Strips HTML tags
    - Truncates to max length
    - Detects prompt injection attempts
    """
    if not message or not message.strip():
        return ""

    # Strip HTML/script tags
    cleaned = re.sub(r"<[^>]+>", "", message)

    # Truncate
    cleaned = cleaned[:max_length].strip()

    return cleaned


def check_injection(message: str) -> bool:
    """Returns True if the message looks like a prompt injection attempt."""
    # Strip out IMAGE_ANALYSIS blocks from the injection check
    message_to_check = re.sub(
        r"\[IMAGE_ANALYSIS\].*?\[/IMAGE_ANALYSIS\]",
        "",
        message,
        flags=re.DOTALL | re.IGNORECASE,
    )
    lower = message_to_check.lower().strip()

    # Legitimate booking resets must never be treated as injection
    if re.match(
        r"^(please\s+)?(start\s+over|start\s+fresh|start\s+again|fresh\s+booking"
        r"|bro\s+start\s+again|reset(\s+booking)?)\s*[.!]?\s*$",
        lower,
    ):
        return False

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lower):
            logger.warning(f"Prompt injection detected: {pattern}")
            return True
    return False


def sanitize_output(response: str) -> str:
    """
    Scrub internal technical terms from bot responses.
    Ensures no database names, collection names, or internal IDs leak.
    """
    cleaned = response
    for term in INTERNAL_TERMS:
        if term in cleaned:
            # Replace with generic alternatives
            cleaned = cleaned.replace(term, "")

    # Clean up any double spaces created by removals (but preserve newlines)
    cleaned = re.sub(r" {2,}", " ", cleaned).strip()

    return cleaned
