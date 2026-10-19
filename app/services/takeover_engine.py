"""
Takeover Engine — Human Takeover Mode + conversation override lifecycle.

Phase 1 (this module):
  - Detect / accept staff participation
  - Pause AI replies while takeover is active
  - Reset 10-minute inactivity timer on every staff message
  - Persist takeover state on BookingSession.conversation_overrides
  - Keep human_interventions as an audit/transcript trail

Does NOT apply pricing/payment/amenity/inventory/booking overrides yet.
pending_decisions may be captured at resume for later phases.
"""

from __future__ import annotations

import logging
import json
import re
from datetime import datetime, timedelta, UTC
from typing import Any

from bson import ObjectId

from app.config.settings import get_settings
from app.db.mongodb import human_interventions_collection
from app.models.conversation_overrides import ConversationOverrides
from app.models.schemas import HumanIntervention, InterventionMessage, BookingSession
from app.repositories import session_repository

logger = logging.getLogger(__name__)


def _timeout_seconds() -> int:
    try:
        return int(get_settings().TAKEOVER_TIMEOUT_SECONDS)
    except Exception:
        return 300


# Backward-compatible module constant (resolved at call time via _timeout_seconds).
# Kept in step with the settings default so a reader who greps this name is not
# told 10 minutes while every live path uses 5.
TAKEOVER_TIMEOUT_SECONDS = 300  # default; prefer settings.TAKEOVER_TIMEOUT_SECONDS


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _ensure_aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def get_overrides(session: BookingSession) -> ConversationOverrides:
    raw = getattr(session, "conversation_overrides", None)
    if isinstance(raw, ConversationOverrides):
        return raw
    if isinstance(raw, dict):
        return ConversationOverrides(**raw)
    return ConversationOverrides()


def is_takeover_active(session: BookingSession, *, now: datetime | None = None) -> bool:
    """True if human takeover is active and the inactivity window has not expired."""
    now = _ensure_aware(now) or _utcnow()
    overrides = get_overrides(session)
    ht = overrides.human_takeover
    if not ht.active or not ht.expires_at:
        return False
    expires = _ensure_aware(ht.expires_at)
    return bool(expires and now < expires)


async def activate_or_extend_takeover(
    session: BookingSession,
    *,
    staff_id: int | str | None = None,
    staff_message: str = "",
    reason: str = "STAFF_MESSAGE",
    message_id: str | None = None,
    message_timestamp: datetime | None = None,
) -> BookingSession:
    """
    Activate Human Takeover Mode, or extend the timer on a new staff message.
    Persists conversation_overrides on the session (SSOT).
    """
    now = _utcnow()
    msg_at = _ensure_aware(message_timestamp) or now
    expires_at = msg_at + timedelta(seconds=_timeout_seconds())

    overrides = get_overrides(session)
    ht = overrides.human_takeover

    if message_id and message_id in ht.seen_message_ids:
        # Idempotent — already processed this staff message
        return session

    if message_id:
        ht.seen_message_ids.append(message_id)
        # Cap growth
        if len(ht.seen_message_ids) > 200:
            ht.seen_message_ids = ht.seen_message_ids[-200:]

    if ht.active and ht.intervention_id:
        intervention_id = ht.intervention_id
        await _extend_intervention_audit(
            intervention_id=intervention_id,
            staff_message=staff_message,
            last_staff_message=msg_at,
            expires_at=expires_at,
        )
        outcome = "extended"
    else:
        intervention_id = await _start_intervention_audit(
            hotel_id=session.hotel_id,
            session_id=session.session_id,
            staff_id=staff_id or "unknown",
            staff_message=staff_message,
            reason=reason,
            started_at=msg_at,
            expires_at=expires_at,
        )
        ht.activated_at = msg_at
        ht.intervention_id = intervention_id
        outcome = "started"

    ht.active = True
    ht.staff_id = staff_id if staff_id is not None else ht.staff_id
    ht.reason = reason
    ht.last_staff_message_at = msg_at
    ht.expires_at = expires_at

    # Parse deterministic staff commands (Tasks 7 + 8)
    from app.services.staff_message_detector import parse_staff_command
    cmd = parse_staff_command(staff_message, session_id=session.session_id)
    if cmd:
        c_type = cmd["command"]
        c_val = cmd["value"]
        if c_type == "price_override":
            overrides.pricing["override_price"] = float(c_val)
            overrides.pricing["override_set_at"] = now.isoformat()
            overrides.pricing["override_offered"] = False
            overrides.pricing["payable_total"] = float(c_val)
            logger.info("STAFF_OVERRIDE_STORED | price=%s | sess=%s", c_val, session.session_id)
        elif c_type == "discount":
            if isinstance(c_val, dict) and c_val.get("type") == "percent":
                overrides.pricing["discount_percent"] = float(c_val["value"])
            elif isinstance(c_val, dict) and c_val.get("type") == "fixed":
                overrides.pricing["discount_amount"] = float(c_val["value"])
        elif c_type == "skip_payment":
            overrides.payment["skip_payment_ask"] = True
        elif c_type == "room_upgrade":
            overrides.booking["upgrade"] = str(c_val)
        elif c_type == "force_confirm":
            from app.models.schemas import BookingStage
            session.booking_stage = BookingStage.CONFIRMED

    overrides.human_takeover = ht
    overrides.touch()
    session.conversation_overrides = overrides

    await session_repository.upsert_session(session)

    logger.info(
        "HUMAN_TAKEOVER_%s | session_id=%s | staff_id=%s | expires_at=%s",
        outcome.upper(),
        session.session_id,
        staff_id,
        expires_at.isoformat(),
    )
    try:
        from app.observability.metrics import metrics
        metrics.incr(
            "human_takeover_total",
            outcome=outcome,
            hotel_id=session.hotel_id,
            reason=reason,
        )
    except Exception:
        pass

    return session


async def sync_staff_messages_into_takeover(
    session: BookingSession,
    customer_phone: str,
) -> BookingSession:
    """
    Pull new staff messages from the backend Chat_Sessions and activate/extend
    Human Takeover for each new one (timer resets to latest).

    Stale staff messages older than the inactivity window do not activate
    takeover (they are marked seen to avoid reprocessing).
    """
    from app.services.staff_message_detector import fetch_staff_messages

    overrides = get_overrides(session)
    ht = overrides.human_takeover
    seen = set(ht.seen_message_ids)

    # Only look for messages newer than last processed staff message when possible
    since = _ensure_aware(ht.last_staff_message_at) if ht.last_staff_message_at else None

    staff_msgs = await fetch_staff_messages(
        session.hotel_id,
        customer_phone or session.customer_identifier or "",
        since=since,
        exclude_message_ids=seen,
    )
    if not staff_msgs:
        return session

    latest_ts = _ensure_aware(staff_msgs[-1].get("timestamp")) or _utcnow()
    age_seconds = (_utcnow() - latest_ts).total_seconds()
    fresh = age_seconds <= _timeout_seconds()

    if not ht.active and not fresh:
        # Historical staff chatter — record ids, do not pause AI
        for msg in staff_msgs:
            mid = msg.get("message_id")
            if mid and mid not in ht.seen_message_ids:
                ht.seen_message_ids.append(mid)
        if len(ht.seen_message_ids) > 200:
            ht.seen_message_ids = ht.seen_message_ids[-200:]
        overrides.human_takeover = ht
        overrides.touch()
        session.conversation_overrides = overrides
        await session_repository.upsert_session(session)
        logger.info(
            "STAFF_MSG_STALE_IGNORED | session_id=%s | count=%s | age_s=%.0f",
            session.session_id,
            len(staff_msgs),
            age_seconds,
        )
        return session

    for msg in staff_msgs:
        session = await activate_or_extend_takeover(
            session,
            staff_id=msg.get("staff_id"),
            staff_message=msg.get("text") or "",
            reason="STAFF_MESSAGE",
            message_id=msg.get("message_id"),
            message_timestamp=msg.get("timestamp"),
        )

    return session


async def evaluate_takeover_for_guest_turn(
    session: BookingSession,
    *,
    customer_phone: str,
    guest_message: str,
) -> tuple[BookingSession, str]:
    """
    Pre-graph gate for every guest message.

    Returns (session, action) where action is:
      - "pause"   → staff is handling; bot must not reply
      - "resume"  → inactivity window expired; AI may continue
      - "continue"→ no active takeover
    """
    # 1) Sync any new staff messages from backend chat history
    session = await sync_staff_messages_into_takeover(session, customer_phone)

    overrides = get_overrides(session)
    ht = overrides.human_takeover
    now = _utcnow()

    # ── Check 60-second price override timer (Task 8) ──
    override_price = overrides.pricing.get("override_price")
    override_set_at = overrides.pricing.get("override_set_at")
    override_offered = overrides.pricing.get("override_offered", False)

    if override_price and override_set_at and not override_offered:
        try:
            set_dt = datetime.fromisoformat(str(override_set_at).replace("Z", "+00:00"))
            set_dt = _ensure_aware(set_dt)
            elapsed_sec = (now - set_dt).total_seconds()
            if elapsed_sec >= 60.0:
                logger.info(
                    "STAFF_PRICE_OVERRIDE_TRIGGERED | price=%s | elapsed_sec=%.1f | sess=%s",
                    override_price,
                    elapsed_sec,
                    session.session_id,
                )
                overrides.pricing["override_offered"] = True
                session.total_price = float(override_price)
                session.price_override = float(override_price)
                overrides.touch()
                session.conversation_overrides = overrides
                await session_repository.upsert_session(session)
                if ht.active:
                    session = await resolve_takeover(session, reason="PRICE_OVERRIDE_60S_TIMEOUT")
                    return session, "resume"
        except Exception as e:
            logger.warning("Error evaluating 60s price override timer: %s", e)

    if not ht.active or not ht.expires_at:
        return session, "continue"

    expires = _ensure_aware(ht.expires_at)
    if expires and now < expires:
        if ht.intervention_id:
            try:
                await log_guest_message_to_intervention(ht.intervention_id, guest_message)
            except Exception as e:
                logger.warning("Failed to log guest message to intervention: %s", e)
        logger.info(
            "AI_PAUSED | session_id=%s | expires_at=%s",
            session.session_id,
            expires.isoformat(),
        )
        return session, "pause"

    # Timer expired → resume AI (no booking/pricing mutation this phase)
    session = await resolve_takeover(session, reason="TIMEOUT")
    return session, "resume"


_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
# Rows that are neither staff nor guest. The bot's own prior replies must not be
# read back to the extractor as if a human had committed to them.
_BOT_OR_SYSTEM_SENDERS = frozenset({"bot", "assistant", "ai", "system"})
# Largest rooms-x-nights factor allowed when deriving a total from a stated rate.
# A quote of "1850 per room for 3 rooms" is a factor of 3; a hallucinated total is
# unlikely to be an exact small multiple of a rate staff actually named.
_MAX_RATE_MULTIPLE = 30

# "2550 per day", "₹3,250/night", "daily rate of 20800", "1850 per room".
# Measured live 2026-10-18: staff wrote a per-day rate, the extractor put it in
# `payable_total`, and the session's whole-stay price became one night's money
# — guest 918918122852 was quoted ₹2,550 for a 2-night ₹5,100 stay, and
# 919064956832 was "adjusted to 2880" the same way.
_RATE_FRAMING_RE = re.compile(
    r"(?:per|a|/|each)\s*(?:day|night|nite|din|raat|room|room\s*night)"
    r"|daily\s*rate|nightly\s*rate|rate\s*(?:of|is)?\s*₹?\s*\d+\s*(?:per|/)",
    re.IGNORECASE,
)
# A rate frames the number it sits beside, not every number in the sentence.
# "total 5100 for 2 days (2550 per day)" must flag only the 2550.
_CLAUSE_SPLIT_RE = re.compile(r"[,;()\n]|(?:\band\b)|(?:\bfor\b)")

def _number_is_framed_as_a_rate(total: float, staff_lines: list[str]) -> bool:
    """
    True when staff stated this exact number as a per-day/night/room rate.

    Why a phrasing check at all, after `_corroborate_payable_total`'s docstring
    argued against one: that argument is about *commitment* ("I'll do 3000"),
    which is a judgement call and would reject ordinary agreements. This is a
    different question with an answer in the text itself — is the number
    followed by "per day"? "3000 me dedo" has no rate framing and is untouched.
    """
    for line in staff_lines:
        for clause in _CLAUSE_SPLIT_RE.split(line or ""):
            if not _RATE_FRAMING_RE.search(clause):
                continue
            # "₹2,550/night" — the thousands separator would otherwise split the
            # number into 2 and 550 and the comparison would never match.
            for match in _NUMBER_RE.finditer(clause.replace(",", "")):
                try:
                    if abs(float(match.group(0)) - total) < 1.0:
                        return True
                except (TypeError, ValueError):
                    continue
    return False



def _message_text(msg: dict[str, Any]) -> str:
    """Text of one chat-history row, across the shapes the two fetchers return.

    `text_body` is the CRM's spelling (`/api/backend/messages/` returns
    `{sender, text_body, timestamp}`). It was missing here, and because a row
    with no text is dropped rather than mislabelled, every CRM row — the guest,
    the bot *and* the staff turns — vanished before the extractor saw it. That
    went unnoticed while the context refresh never reached the CRM at all; once
    `fetch_chat_history` was pointed at the working endpoint it became the next
    thing silently eating the staff lines.
    """
    for key in ("content", "text", "message", "text_body"):
        value = msg.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _label_messages(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """
    Label chat-history rows as ("STAFF"|"GUEST", text), preserving conversation order.

    Order matters as much as the label: the extractor is deciding what staff
    committed to, and a staff line only means something next to the guest turn it
    answers. Grouping all staff lines together first reads as one long monologue
    and loses the pairing — measured on real transcripts, that cost `num_rooms`
    on three of six cases.

    The staff/guest split itself has to be right too: a guest's own claim ("I'll
    pay 2000") mislabelled as staff speech becomes a decision the bot enforces.
    Two predicates because the two fetchers disagree on shape — `crm_service`
    returns `sender`, the Mongo path returns `role`.
    """
    from app.services.staff_message_detector import (
        is_crm_staff_message,
        is_mongo_staff_message,
    )

    labelled: list[tuple[str, str]] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        text = _message_text(msg)
        if not text:
            continue
        if is_crm_staff_message(msg) or is_mongo_staff_message(msg):
            labelled.append(("STAFF", text))
            continue
        sender = str(msg.get("sender") or msg.get("role") or "").strip().lower()
        if sender in _BOT_OR_SYSTEM_SENDERS:
            continue
        # Unknown senders count as guest on purpose: mislabelling a guest turn as
        # staff is the expensive direction, the reverse only loses a line.
        labelled.append(("GUEST", text))
    return labelled


def _split_messages(messages: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Staff and guest turns as separate lists, for corroboration checks."""
    labelled = _label_messages(messages)
    return (
        [text for role, text in labelled if role == "STAFF"],
        [text for role, text in labelled if role == "GUEST"],
    )


def _corroborate_payable_total(decisions: dict[str, Any], staff_lines: list[str]) -> dict[str, Any]:
    """
    Drop a `payable_total` that no staff message actually states.

    `_sanitize_domain_decisions` type-checks the extraction but cannot check
    intent — it sees `3000.0` and has no way to know whether staff said "3000"
    or "we could maybe do 3000 if you book three nights". So ask a question the
    model cannot get wrong by reasoning: does that number appear in a message
    staff sent? If not, the rate survives as a note and the pricing engine keeps
    its own figure.

    Corroboration rather than phrasing, deliberately. Reusing
    `parse_staff_command`'s commitment patterns as the guard would reject
    "I'll do it for 3000" — just as explicit, and far more common in the wild.

    Replaces `payable_total` rather than deleting the whole `pricing` domain:
    `discount_amount` and `rate_note` are useful to the model even when the
    total itself is not trustworthy enough to charge.
    """
    pricing = decisions.get("pricing")
    if not isinstance(pricing, dict):
        return decisions
    total = pricing.get("payable_total")
    if total is None:
        return decisions

    try:
        total = float(total)
    except (TypeError, ValueError):
        return decisions

    stated: set[float] = set()
    for line in staff_lines:
        for match in _NUMBER_RE.finditer(line or ""):
            try:
                stated.add(float(match.group(0)))
            except (TypeError, ValueError):
                continue

    if any(abs(n - total) < 1.0 for n in stated):
        if _number_is_framed_as_a_rate(total, staff_lines):
            # The number is in the staff's words, but as a RATE. Applied as
            # `payable_total` it becomes the whole stay's price.
            logger.warning(
                "STAFF_PRICE_IS_A_RATE | payable_total=%s is stated as a "
                "per-day/per-night/per-room rate — kept as a note, not applied",
                total,
            )
            remaining = {k: v for k, v in pricing.items() if k != "payable_total"}
            out = dict(decisions)
            if remaining:
                out["pricing"] = remaining
            else:
                out.pop("pricing", None)
            return out
        return decisions

    # Staff usually quote a RATE, not a lump sum — "1850 per room for 3 rooms" —
    # and the total is the model's arithmetic on it. Requiring the total itself to
    # appear dropped a genuinely agreed price on a real transcript, which is the
    # very bug this change exists to fix. A total that is an exact small multiple
    # of a number staff stated is corroborated by that rate: the guest can be
    # charged it because staff named both the rate and the count.
    for n in stated:
        if n <= 0:
            continue
        k = total / n
        if abs(k - round(k)) < 1e-6 and 1 <= round(k) <= _MAX_RATE_MULTIPLE:
            logger.info(
                "STAFF_PRICE_DERIVED | payable_total=%s = %s x %d stated by staff",
                total, n, round(k),
            )
            return decisions

    logger.warning(
        "STAFF_PRICE_UNCORROBORATED | payable_total=%s | staff_messages=%d | "
        "no staff message states this number — kept as a note, not applied",
        total,
        len(staff_lines),
    )
    remaining = {k: v for k, v in pricing.items() if k != "payable_total"}
    out = dict(decisions)
    if remaining:
        out["pricing"] = remaining
    else:
        out.pop("pricing", None)
    return out


async def extract_staff_decisions_from_messages(
    messages: list[dict[str, Any]],
) -> tuple[dict[str, Any], str | None]:
    """
    Typed staff decisions read from a raw chat-history window.

    The takeover path reads a stored intervention transcript; the context-refresh
    path has only rows it fetched itself. This wraps the same extractor so both
    produce identical decisions, and adds the corroboration guard the transcript
    path does not have (see `_corroborate_payable_total`).

    Never raises. A failed extraction returns no decisions, because the caller
    still holds a free-text summary and a crash here would lose that too.
    """
    labelled = _label_messages(messages)
    staff_lines = [text for role, text in labelled if role == "STAFF"]
    if not staff_lines:
        return {}, None

    # In order, so each staff line still sits next to the guest turn it answers.
    transcript = "\n".join(f"{role}: {text}" for role, text in labelled)
    try:
        decisions, summary = await _extract_staff_decisions(transcript)
    except Exception as e:
        logger.error("STAFF_DECISIONS_EXTRACT_FAILED | error=%s", e, exc_info=True)
        return {}, None

    decisions = _corroborate_payable_total(decisions, staff_lines)
    logger.info(
        "STAFF_DECISIONS_EXTRACTED | source=messages | domains=%s",
        list(decisions.keys()),
    )
    return decisions, summary


async def apply_staff_decisions(
    session: BookingSession,
    domain_decisions: dict[str, Any] | None = None,
    *,
    summary: str | None = None,
    source: str = "staff_takeover",
    reason: str = "",
    staff_id: int | str | None = None,
    intervention_id: str | None = None,
) -> dict[str, Any]:
    """
    Record staff decisions on the session and mirror them onto what prices read.

    `apply_decisions` alone only *records* — it fills the override domains and
    `prompt_summary()` renders them so the model stops contradicting staff. That
    is not enough to change a price: the number a guest is charged comes from
    `session.total_price` / `price_override` through `resolve_effective_total`,
    which never consults the overrides. Both halves live here so the takeover
    path and the context-refresh path cannot drift apart.

    Does not persist — the caller upserts, because it usually has more to write.
    Returns the change set from `apply_decisions`.
    """
    decisions = domain_decisions if isinstance(domain_decisions, dict) else {}
    if not decisions and not summary:
        return {}

    overrides = get_overrides(session)
    change_set = overrides.apply_decisions(
        decisions,
        source=source,
        reason=reason,
        staff_id=staff_id,
        intervention_id=intervention_id,
        summary=summary,
    )
    # Keep a raw snapshot for debugging
    overrides.pending_decisions = {
        **decisions,
        "_resume_reason": reason,
        "_captured_at": _utcnow().isoformat(),
    }
    logger.info(
        "STAFF_DECISIONS_APPLIED | session_id=%s | source=%s | domains=%s",
        session.session_id,
        source,
        list(change_set.keys()),
    )

    # Mirror payable_total onto legacy price_override for older quote paths
    payable = overrides.get("pricing", "payable_total")
    if payable is not None:
        try:
            session.price_override = float(payable)
        except (TypeError, ValueError):
            pass

    # Soft-apply payment confirmed amounts onto session payment fields
    if overrides.has("payment", "amount_paid_confirmed"):
        try:
            session.amount_paid = float(overrides.get("payment", "amount_paid_confirmed"))
        except (TypeError, ValueError):
            pass
    total_for_due = session.price_override or session.total_price
    if overrides.has("payment", "amount_due_override"):
        try:
            session.amount_due = float(overrides.get("payment", "amount_due_override"))
        except (TypeError, ValueError):
            pass
    elif total_for_due is not None and session.amount_paid is not None:
        session.amount_due = max(0.0, float(total_for_due) - float(session.amount_paid or 0))

    if overrides.has("pricing", "payable_total"):
        try:
            session.total_price = float(overrides.get("pricing", "payable_total"))
        except (TypeError, ValueError):
            pass

    # Booking add-ons
    booking = overrides.booking or {}
    if "breakfast_included" in booking:
        session.included_breakfast = bool(booking["breakfast_included"])
    if "late_checkout" in booking:
        session.late_checkout = bool(booking["late_checkout"])
    if "airport_pickup" in booking:
        session.airport_pickup = bool(booking["airport_pickup"])
    if booking.get("special_notes"):
        session.special_notes = str(booking["special_notes"])
    if booking.get("room_type"):
        session.room_type_preference = str(booking["room_type"])

    overrides.touch()
    session.conversation_overrides = overrides
    return change_set


async def resolve_takeover(
    session: BookingSession,
    *,
    reason: str = "TIMEOUT",
    intervention_id: str | None = None,
) -> BookingSession:
    """
    End Human Takeover Mode.

    Extracts staff decisions from the takeover transcript and applies them as
    conversation-level overrides (session only — never hotel master DB).
    Silent adoption: no guest-facing announcement.
    """
    overrides = get_overrides(session)
    ht = overrides.human_takeover
    intv_id = intervention_id or ht.intervention_id

    domain_decisions: dict[str, Any] = {}
    gemini_summary: str | None = None

    if intv_id:
        try:
            domain_decisions, gemini_summary = await _resolve_intervention_audit(intv_id)
        except Exception as e:
            logger.warning(
                "Intervention audit resolve failed | id=%s | error=%s",
                intv_id,
                e,
            )

    await apply_staff_decisions(
        session,
        domain_decisions,
        summary=gemini_summary,
        source="staff_takeover",
        reason=reason,
        staff_id=ht.staff_id,
        intervention_id=intv_id,
    )

    # Re-read: apply_staff_decisions may have replaced session.conversation_overrides
    # (get_overrides builds a new object when the stored value is a plain dict), so
    # the `overrides`/`ht` bound above can be stale by the time we clear the flag.
    overrides = get_overrides(session)
    ht = overrides.human_takeover
    ht.active = False
    ht.expires_at = None
    overrides.human_takeover = ht
    overrides.touch()
    session.conversation_overrides = overrides

    await session_repository.upsert_session(session)
    logger.info(
        "AI_RESUMED | session_id=%s | reason=%s",
        session.session_id,
        reason,
    )
    return session


# ═══════════════════════════════════════════════════════════════════
# Public API used by admin routes (push path from CRM)
# ═══════════════════════════════════════════════════════════════════

async def start_or_update_intervention(
    hotel_id: int,
    session_id: str,
    staff_id: int | str,
    staff_message: str,
    reason: str = "NEGOTIATION",
) -> str:
    """
    CRM/admin push: staff sent a message → activate/extend takeover on session.
    """
    session = await session_repository.get_session(session_id)
    if not session:
        session = BookingSession(session_id=session_id, hotel_id=hotel_id)
    elif session.hotel_id != hotel_id:
        session.hotel_id = hotel_id

    session = await activate_or_extend_takeover(
        session,
        staff_id=staff_id,
        staff_message=staff_message,
        reason=reason or "NEGOTIATION",
    )
    ht = get_overrides(session).human_takeover
    return ht.intervention_id or ""


async def get_active_intervention(hotel_id: int, session_id: str) -> dict | None:
    """
    Backward-compatible helper. Prefers session SSOT; falls back to audit collection.
    """
    session = await session_repository.get_session(session_id)
    if session and session.hotel_id == hotel_id and is_takeover_active(session):
        ht = get_overrides(session).human_takeover
        return {
            "intervention_id": ht.intervention_id,
            "hotel_id": hotel_id,
            "session_id": session_id,
            "status": "ACTIVE",
            "expires_at": ht.expires_at,
            "last_staff_message": ht.last_staff_message_at,
            "staff_id": ht.staff_id,
            "reason": ht.reason,
        }

    col = human_interventions_collection()
    return await col.find_one(
        {"hotel_id": hotel_id, "session_id": session_id, "status": "ACTIVE"}
    )


async def log_guest_message_to_intervention(intervention_id: str, guest_message: str) -> None:
    """Append a guest message to the active intervention transcript."""
    col = human_interventions_collection()
    msg = InterventionMessage(sender="guest", text=guest_message, timestamp=_utcnow())
    await col.update_one(
        {"intervention_id": intervention_id},
        {"$push": {"transcript": msg.model_dump()}},
    )


async def resolve_intervention(intervention_id: str) -> None:
    """
    Admin/manual resume: resolve audit doc + clear session takeover flag.
    """
    col = human_interventions_collection()
    doc = await col.find_one({"intervention_id": intervention_id})
    if not doc:
        return

    session = await session_repository.get_session(doc["session_id"])
    if session:
        await resolve_takeover(
            session,
            reason="MANUAL_RESUME",
            intervention_id=intervention_id,
        )
    else:
        await _resolve_intervention_audit(intervention_id)


# ═══════════════════════════════════════════════════════════════════
# Audit trail helpers (human_interventions collection)
# ═══════════════════════════════════════════════════════════════════

async def _start_intervention_audit(
    *,
    hotel_id: int,
    session_id: str,
    staff_id: int | str,
    staff_message: str,
    reason: str,
    started_at: datetime,
    expires_at: datetime,
) -> str:
    col = human_interventions_collection()
    intervention_id = f"INTV-{ObjectId()}"
    msg = InterventionMessage(sender="staff", text=staff_message, timestamp=started_at)
    doc = HumanIntervention(
        intervention_id=intervention_id,
        hotel_id=hotel_id,
        session_id=session_id,
        staff_id=staff_id,
        reason=reason,
        started_at=started_at,
        last_staff_message=started_at,
        expires_at=expires_at,
        transcript=[msg] if staff_message else [],
    )
    await col.insert_one(doc.model_dump())
    return intervention_id


async def _extend_intervention_audit(
    *,
    intervention_id: str,
    staff_message: str,
    last_staff_message: datetime,
    expires_at: datetime,
) -> None:
    col = human_interventions_collection()
    update: dict[str, Any] = {
        "$set": {
            "last_staff_message": last_staff_message,
            "expires_at": expires_at,
            "status": "ACTIVE",
        }
    }
    if staff_message:
        msg = InterventionMessage(sender="staff", text=staff_message, timestamp=last_staff_message)
        update["$push"] = {"transcript": msg.model_dump()}
    await col.update_one({"intervention_id": intervention_id}, update)


async def _resolve_intervention_audit(intervention_id: str) -> tuple[dict, str | None]:
    """
    Mark intervention RESOLVED and extract domain-structured staff decisions.
    Returns (domain_decisions, summary). Does NOT write hotel master DB.
    """
    col = human_interventions_collection()
    doc = await col.find_one({"intervention_id": intervention_id, "status": "ACTIVE"})
    if not doc:
        return {}, None

    intervention = HumanIntervention(**doc)
    transcript_text = "\n".join(
        f"{m.sender.upper()}: {m.text}" for m in intervention.transcript
    )

    domain_decisions: dict = {}
    gemini_summary: str | None = None
    if transcript_text.strip():
        try:
            domain_decisions, gemini_summary = await _extract_staff_decisions(transcript_text)
            logger.info(
                "STAFF_DECISIONS_EXTRACTED | intervention_id=%s | domains=%s",
                intervention_id,
                list(domain_decisions.keys()),
            )
        except Exception as e:
            logger.error(
                "STAFF_DECISIONS_FAILED | intervention_id=%s | error=%s",
                intervention_id,
                e,
            )

    now = _utcnow()
    started = _ensure_aware(intervention.started_at) or now
    duration = int((now - started).total_seconds())

    await col.update_one(
        {"intervention_id": intervention_id},
        {
            "$set": {
                "status": "RESOLVED",
                "ended_at": now,
                "duration_seconds": duration,
                "gemini_summary": gemini_summary,
                "overrides_applied": domain_decisions or None,
                "pending_decisions": domain_decisions or None,
            }
        },
    )
    logger.info("HUMAN_INTERVENTION_ENDED | intervention_id=%s", intervention_id)
    return domain_decisions, gemini_summary


async def _extract_staff_decisions(transcript: str) -> tuple[dict, str]:
    """
    Extract explicit staff decisions into typed conversation override domains.
    Only explicit facts — never invent. Empty domains when staff made no decision.
    """
    from app.services.gemini_service import _get_model

    model = _get_model()
    prompt = f"""You extract hotel STAFF decisions from a takeover transcript.
These become conversation-level overrides for the AI (session only). Extract ONLY
explicit staff statements of fact or commitments. Ignore greetings, fillers, and
guest-only claims that staff did not confirm.

Return ONLY strict JSON:
{{
  "summary": "One sentence of what staff decided, or empty string if nothing",
  "pricing": {{
    "payable_total": float or null,
    "discount_amount": float or null,
    "discount_percent": float or null,
    "rate_note": string or null
  }},
  "payment": {{
    "amount_paid_confirmed": float or null,
    "amount_due_override": float or null,
    "payment_status": "paid"|"partial"|"pending"|"waived"|null,
    "skip_payment_ask": boolean or null
  }},
  "amenities": {{
    "<amenity_slug>": {{"available": boolean, "note": string or null, "name": string}}
  }},
  "inventory": {{
    "availability_confirmed": boolean or null,
    "room_type": string or null,
    "rooms_available_note": string or null
  }},
  "booking": {{
    "late_checkout": boolean or null,
    "early_checkin": boolean or null,
    "breakfast_included": boolean or null,
    "airport_pickup": boolean or null,
    "upgrade": string or null,
    "special_notes": string or null,
    "guest_name": string or null,
    "check_in": string or null,
    "check_out": string or null,
    "num_rooms": int or null,
    "adults": int or null,
    "children": int or null,
    "room_type": string or null
  }}
}}

Rules:
- payable_total = final amount guest should pay for the whole stay if staff quoted a specific total.
- payable_total is NEVER a per-night, per-day or per-room rate. If staff said "2550 per day" or "3250/night", leave payable_total null and put that rate in rate_note instead.
- amount_paid_confirmed = amount staff said guest already paid.
- amenities: e.g. staff says pool is available → {{"pool": {{"available": true, "name": "Swimming Pool"}}}}
- Use null for anything not explicitly decided by staff.
- amenity slugs: lowercase letters only (pool, gym, spa, wifi, parking, restaurant).

TRANSCRIPT:
{transcript}
"""
    import google.generativeai as genai

    response = await model.generate_content_async(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )
    text = (response.text or "").strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]

    data = json.loads(text.strip())
    summary = str(data.get("summary") or "").strip()
    return _sanitize_domain_decisions(data), summary


def _sanitize_domain_decisions(data: dict) -> dict[str, Any]:
    """Python validation — only keep well-typed explicit values."""
    out: dict[str, Any] = {}

    pricing_in = data.get("pricing") or {}
    pricing: dict[str, Any] = {}
    for key in ("payable_total", "discount_amount", "discount_percent"):
        if pricing_in.get(key) is not None:
            try:
                val = float(pricing_in[key])
                if val >= 0:
                    pricing[key] = val
            except (TypeError, ValueError):
                pass
    if pricing_in.get("rate_note"):
        pricing["rate_note"] = str(pricing_in["rate_note"])
    if pricing:
        out["pricing"] = pricing

    payment_in = data.get("payment") or {}
    payment: dict[str, Any] = {}
    for key in ("amount_paid_confirmed", "amount_due_override"):
        if payment_in.get(key) is not None:
            try:
                val = float(payment_in[key])
                if val >= 0:
                    payment[key] = val
            except (TypeError, ValueError):
                pass
    status = payment_in.get("payment_status")
    if isinstance(status, str) and status.lower() in ("paid", "partial", "pending", "waived", "comped"):
        payment["payment_status"] = status.lower()
    if isinstance(payment_in.get("skip_payment_ask"), bool):
        payment["skip_payment_ask"] = payment_in["skip_payment_ask"]
    # If staff confirmed full payment, skip asking
    if payment.get("payment_status") == "paid":
        payment["skip_payment_ask"] = True
    if payment:
        out["payment"] = payment

    amenities_in = data.get("amenities") or {}
    amenities: dict[str, Any] = {}
    if isinstance(amenities_in, dict):
        for slug, meta in amenities_in.items():
            if not isinstance(meta, dict):
                continue
            clean_slug = "".join(ch for ch in str(slug).lower() if ch.isalnum())
            if not clean_slug:
                continue
            entry: dict[str, Any] = {}
            if isinstance(meta.get("available"), bool):
                entry["available"] = meta["available"]
            if meta.get("note"):
                entry["note"] = str(meta["note"])
            if meta.get("name"):
                entry["name"] = str(meta["name"])
            if entry:
                amenities[clean_slug] = entry
    if amenities:
        out["amenities"] = amenities

    inventory_in = data.get("inventory") or {}
    inventory: dict[str, Any] = {}
    if isinstance(inventory_in.get("availability_confirmed"), bool):
        inventory["availability_confirmed"] = inventory_in["availability_confirmed"]
    if inventory_in.get("room_type"):
        inventory["room_type"] = str(inventory_in["room_type"])
    if inventory_in.get("rooms_available_note"):
        inventory["rooms_available_note"] = str(inventory_in["rooms_available_note"])
    if inventory:
        out["inventory"] = inventory

    booking_in = data.get("booking") or {}
    booking: dict[str, Any] = {}
    for bool_key in ("late_checkout", "early_checkin", "breakfast_included", "airport_pickup"):
        if isinstance(booking_in.get(bool_key), bool):
            booking[bool_key] = booking_in[bool_key]
    for str_key in ("upgrade", "special_notes", "guest_name", "room_type"):
        if booking_in.get(str_key):
            booking[str_key] = str(booking_in[str_key])
    # A stay date must be a date here too. Staff summaries routinely carry the
    # hotel's clock times ("check-in at 1:00 PM"), and a time in this dict is
    # rendered into the model prompt as STAFF CONVERSATION TRUTH — which is how
    # guest 918918122852's dates became "1:00 PM"/"11:00 AM" on 2026-10-18.
    from app.utils.helpers import is_valid_date, normalize_date

    for date_key in ("check_in", "check_out"):
        if not booking_in.get(date_key):
            continue
        raw = str(booking_in[date_key]).strip()
        if is_valid_date(raw) or is_valid_date(normalize_date(raw) or ""):
            booking[date_key] = raw
        else:
            logger.warning(
                "STAFF_DECISION_DATE_REJECTED | field=%s | value=%r — not a date",
                date_key,
                raw,
            )
    for int_key in ("num_rooms", "adults", "children"):
        if booking_in.get(int_key) is not None:
            try:
                booking[int_key] = int(booking_in[int_key])
            except (TypeError, ValueError):
                pass
    if booking:
        out["booking"] = booking

    return out
