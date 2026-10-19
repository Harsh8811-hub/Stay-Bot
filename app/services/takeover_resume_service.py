"""Make the bot *speak* when a human takeover ends.

`POST /api/admin/context-refresh` has always summarised the staff conversation
and written it into the session — and then stopped. It saved memory and sent
nothing, so a guest sitting in a paused chat heard silence until they happened to
write again, and the bot's first real turn after a takeover began from scratch as
far as the guest could tell.

The CRM does its half: `run_override_scheduler` calls this endpoint the moment
its `ManualOverride` expires. This module is the other half — one message, from
the bot, continuing the conversation with whatever the colleague settled.

Deliberately *not* a graph turn. There is no guest message to answer, and driving
`run_chat_turn` would mean inventing one and writing it into history as though the
guest had said it. The staff decisions are already applied to the session by
`save_context_to_session` (which runs first), so the pricing and policy paths
honour them on the guest's *next* real turn regardless of what this sends; this
is the hand-back, not the transaction.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

# A resume message is a conversational hand-back, so it must not be long. The
# composer is told once; this is the backstop for when it is ignored.
MAX_RESUME_CHARS = 600


def _parse_iso(value: Any) -> datetime | None:
    """Parse the CRM's `bot_off_time`.

    The CRM sends `override.bot_off_time.isoformat() + "Z"`, and mongoengine
    stores that field naive-UTC, so the usual shape is
    `2026-10-08T13:22:03.123456Z`. Tolerate an offset-bearing value too rather
    than guess: a misread timestamp silently disables the guard below.
    """
    if not value:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return None
        # Try in order: as sent; `Z` read as UTC; `Z` simply dropped. That last
        # one covers the CRM appending `Z` to an isoformat() that already ended
        # in "+00:00" — malformed, but real, and the parse must not silently
        # fail into the fallback cutoff.
        candidates = [text]
        if text.endswith("Z"):
            candidates += [text[:-1] + "+00:00", text[:-1]]
        dt = None
        for candidate in candidates:
            try:
                dt = datetime.fromisoformat(candidate)
                break
            except ValueError:
                continue
        if dt is None:
            logger.warning(f"TAKEOVER_RESUME_BAD_TIMESTAMP | value={text[:40]}")
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _already_engaged(session, cutoff: datetime) -> bool:
    """True if the guest or the bot has said something since the takeover began.

    Both cases mean the hand-back has already happened on its own, and a
    proactive message now would talk over a live conversation. The system note
    `save_context_to_session` appends is excluded — it is `role="system"`, and it
    is written seconds before this runs.
    """
    for m in (getattr(session, "messages", None) or []):
        role = getattr(m, "role", None)
        sender = getattr(m, "sender_type", None)
        guest_spoke = role == "user" and sender == "guest"
        bot_spoke = role == "assistant"
        if not (guest_spoke or bot_spoke):
            continue
        ts = getattr(m, "timestamp", None)
        if ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        if ts > cutoff:
            return True
    return False


async def resume_after_takeover(
    *,
    hotel_id: int | str,
    phone: str,
    started_at: Any = None,
) -> bool:
    """Send one continuation message if the bot has been silent since `started_at`.

    Returns True only when a message was actually sent. Every failure path
    returns False and logs: this runs inside the CRM's background request, and a
    raise here would take the context write down with it.
    """
    settings = get_settings()
    if not getattr(settings, "TAKEOVER_RESUME_ENABLED", False):
        return False

    from app.repositories import session_repository
    from app.utils.phone import digits_only

    digits = digits_only(phone) or str(phone).strip()
    if not digits:
        return False

    try:
        hid = int(hotel_id) if str(hotel_id).isdigit() else hotel_id
    except (TypeError, ValueError):
        hid = hotel_id

    # `started_at` is when the colleague took over. Without it, fall back to one
    # timeout window ago — that is the moment the CRM decided the takeover ended.
    cutoff = _parse_iso(started_at) or (
        datetime.now(UTC) - timedelta(seconds=int(settings.TAKEOVER_TIMEOUT_SECONDS))
    )

    try:
        session = await session_repository.find_active_session(hid, digits)
    except Exception as e:
        logger.error(f"TAKEOVER_RESUME_SESSION_LOOKUP_FAILED | phone={digits} | error={e}")
        return False

    if session is None:
        # No live conversation to continue. The summary is already parked in
        # `pending_context` and will apply on the guest's next turn.
        logger.info(f"TAKEOVER_RESUME_SKIP | phone={digits} | reason=no_active_session")
        return False

    if _already_engaged(session, cutoff):
        logger.info(f"TAKEOVER_RESUME_SKIP | phone={digits} | reason=already_engaged")
        return False

    from app.services.gemini_service import generate_takeover_resume_message

    summary = _latest_staff_summary(session)
    decisions = _staff_decisions(session)
    hotel_name = "our hotel"
    try:
        from app.repositories import hotel_repository

        details = await hotel_repository.get_hotel_details(hid) or {}
        hotel_name = (
            (details.get("property") or {}).get("name")
            or details.get("hotel_name")
            or details.get("name")
            or hotel_name
        )
    except Exception as e:
        logger.warning(f"TAKEOVER_RESUME_HOTEL_LOOKUP_FAILED | hotel={hid} | error={e}")

    try:
        body = await generate_takeover_resume_message(
            hotel_name=hotel_name,
            summary=summary,
            decisions=decisions,
            recent_messages=[
                m.model_dump(mode="json") if hasattr(m, "model_dump") else dict(m)
                for m in (getattr(session, "messages", None) or [])
            ],
        )
    except Exception as e:
        logger.error(f"TAKEOVER_RESUME_COMPOSE_FAILED | phone={digits} | error={e}")
        return False

    body = (body or "").strip()
    if not body:
        # The composer returns "" on a Gemini failure, and a blank WhatsApp send
        # is worse than silence.
        logger.info(f"TAKEOVER_RESUME_SKIP | phone={digits} | reason=empty_message")
        return False
    if len(body) > MAX_RESUME_CHARS:
        body = body[: MAX_RESUME_CHARS - 1].rstrip() + "…"

    from app.services.whatsapp_service import whatsapp_service

    try:
        sent = whatsapp_service.send_message(
            to_number=digits,
            message_body=body,
            hotel_id=hid,
        )
    except Exception as e:
        logger.error(f"TAKEOVER_RESUME_SEND_FAILED | phone={digits} | error={e}")
        return False

    if not sent:
        logger.warning(f"TAKEOVER_RESUME_SEND_REJECTED | phone={digits}")
        return False

    # Record it, so the guest's next turn sees what the bot said — and so the
    # `_already_engaged` guard refuses a second resume for this takeover.
    try:
        from app.models.schemas import MessageRecord

        session.messages.append(
            MessageRecord(role="assistant", content=body, sender_type="bot")
        )
        await session_repository.upsert_session(session)
    except Exception as e:
        logger.error(f"TAKEOVER_RESUME_PERSIST_FAILED | phone={digits} | error={e}")

    logger.info(
        f"TAKEOVER_RESUME_SENT | hotel={hid} | phone={digits} | chars={len(body)}"
    )
    return True


def _latest_staff_summary(session) -> str:
    """The prose `save_context_to_session` just wrote, read back off the session.

    The refresh endpoint applies the summary through `apply_staff_decisions`
    (filing it in the audit trail and `add_note`) or appends it as a system
    message; this reads whichever landed rather than passing it twice through
    two call sites that could drift.
    """
    for m in reversed(getattr(session, "messages", None) or []):
        if getattr(m, "role", None) == "system" and "Summary:" in (getattr(m, "content", "") or ""):
            return str(m.content).split("Summary:", 1)[1].strip()
    return ""


def _staff_decisions(session) -> dict:
    """The typed decisions already applied to the session, read off the overrides.

    These are the real agreed values — `apply_staff_decisions` files them in the
    domain dicts. `notes` is deliberately not read: it holds the prose summary,
    which is already passed separately, and the two would just duplicate.
    """
    overrides = getattr(session, "conversation_overrides", None)
    if overrides is None:
        return {}

    agreed: dict[str, Any] = {}
    for domain in ("pricing", "payment", "amenities", "inventory", "booking"):
        values = getattr(overrides, domain, None)
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            if value not in (None, "", [], {}):
                agreed[f"{domain}.{key}"] = value
    return agreed
