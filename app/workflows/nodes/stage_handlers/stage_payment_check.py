"""
Stage 1b Handler — Payment Under Check.

Sir's Specification:
- A guest enters Stage 1b when sending a payment screenshot OR claiming payment ("maine payment kar diya").
- The bot detects the payment claim/proof, thanks the guest, and explains front desk is verifying.
- Raises an owner task containing payment message or image.
- Booking remains PENDING status.
- The bot MUST NOT say the booking is confirmed.
- The bot MUST NOT build a receipt-style booking card.
- Do NOT ask upsell questions, meal plans, or add-ons.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.config.settings import get_settings
from app.prompts.templates import text_card
from app.repositories import booking_repository, task_repository
from app.workflows.state import ConversationState

logger = logging.getLogger(__name__)

# A booking in any of these states has already been settled (or killed) and must
# never be walked backwards into "pending_payment" by a payment claim. Kept as
# the *union* of the two vocabularies, because `stage_resolver` reads both
# `booking_status` and `stage` and either one being set to pending_payment /
# payment_check overrides everything else.
TERMINAL_BOOKING_STATUSES = ("confirmed", "checked_in", "checked_out", "cancelled")
TERMINAL_STAGES = ("confirmed", "in_house", "checked_out", "cancelled")

# ── What the guest is actually asking, while their payment is verified ───────
#
# These are deliberately narrow, and they only ever choose between replies that
# all state the same fact — the payment is still under verification, the
# booking is not confirmed yet. None of them can promote a booking; a wrong
# match costs a slightly less relevant sentence, never a false confirmation.
_CLAIM_RE = re.compile(
    r"\b(paid|payment|pay\s*kar|paisa|paise|utr|txn|transaction|screenshot|"
    r"upi|transferred?|bhej\s*diya|kar\s*diya|done)\b",
    re.IGNORECASE,
)
_CONFIRM_ASK_RE = re.compile(
    r"\b(confirmation|confirmed?|receipt|voucher|invoice|slip|booking\s+details|"
    r"booking\s+id|booking\s+reference)\b",
    re.IGNORECASE,
)
_URGENT_RE = re.compile(
    r"\b(today|aaj|jaldi|urgent|abhi|right\s+now|check\s*-?\s*in|checkout|"
    r"check\s*-?\s*out|baje|tonight)\b",
    re.IGNORECASE,
)
_GREETING_RE = re.compile(r"^\s*(hi+|hello+|hey+|namaste|namaskar|salaam|good\s+\w+)\s*[!.🙏]*\s*$", re.IGNORECASE)


def _status_reply(user_msg: str, booking_id: str) -> str | None:
    """
    A reply that answers the guest's turn instead of repeating the same paragraph.

    Returns None when the message is a payment claim (or unrecognised), so the
    caller keeps the original, deliberately fixed wording.

    The urgency case exists because the live guest had a check-in hours away and
    was told to wait, in the same words, for the third time. Acknowledging the
    time pressure — and saying the front desk has the message — is the only
    thing the bot can honestly do before a human verifies the payment.
    """
    msg = (user_msg or "").strip()
    if not msg:
        return None

    if _GREETING_RE.match(msg):
        return (
            f"Namaste! 🙏 Aapki booking `{booking_id}` ka payment abhi front desk "
            "verify kar rahi hai.\n\n"
            "• *Current Status:* ⏳ **PENDING VERIFICATION**\n\n"
            "Verify hote hi confirmation slip bhej di jayegi. Kuch aur chahiye "
            "to bata dijiye."
        )

    if _URGENT_RE.search(msg):
        return (
            "Aapka message front desk ko bhej diya hai — wo aapke check-in/"
            "check-out ki timing dekh rahe hain. 🙏\n\n"
            f"• *Booking Reference:* `{booking_id}`\n"
            "• *Payment Status:* ⏳ **PENDING VERIFICATION**\n"
            "• *Booking Status:* verification ke baad hi confirm hoga\n\n"
            "Payment verify hote hi wo aapko isi chat par confirmation bhej "
            "denge, taaki aap check-in kar sakein."
        )

    if _CONFIRM_ASK_RE.search(msg):
        return (
            "Confirmation abhi isliye nahi bhej sakte ki aapka payment front "
            "desk verify kar rahi hai. 🙏\n\n"
            f"• *Booking Reference:* `{booking_id}`\n"
            "• *Current Status:* ⏳ **PENDING VERIFICATION**\n\n"
            "Verify hote hi official confirmation slip isi chat par aa jayegi. "
            "Aapko kuch aur karne ki zaroorat nahi hai."
        )

    return None


async def handle_stage_payment_check(state: ConversationState) -> ConversationState:
    hotel_id = state.get("hotel_id", 111111)
    str_hid = str(hotel_id)
    phone = state.get("customer_identifier") or ""
    user_msg = state.get("user_message") or ""
    booking_id = state.get("active_booking_id") or state.get("booking_id") or "BK-PENDING"

    logger.info(
        "STAGE_1B_PAYMENT_CHECK_TRIGGERED | hotel=%s phone=%s booking_id=%s msg='%s'",
        str_hid, phone, booking_id, user_msg[:60],
    )

    # 1. Ensure booking in DB is flagged as payment_proof_submitted and stays pending
    try:
        from app.db.mongodb import bookings_collection
        coll = bookings_collection()
        await coll.update_one(
            # NEVER demote a stay that is already settled. The `$set` below
            # writes `booking_status: pending_payment` and `stage: payment_check`,
            # and `stage_resolver` treats both as overriding everything else — so
            # without this guard a guest whose booking is CONFIRMED who says
            # "already paid" has their confirmation erased by the bot itself, and
            # every later turn resolves to payment_check. The bot can then no
            # longer see the booking it confirmed, and re-enters the payment flow.
            #
            # Guarding inside the FILTER keeps this a single atomic update: a
            # settled booking simply does not match, so nothing is written and
            # the row is left exactly as it was.
            {"hotel_id": {"$in": [str_hid, int(str_hid) if str_hid.isdigit() else str_hid]},
             "booking_status": {"$nin": TERMINAL_BOOKING_STATUSES},
             "stage": {"$nin": TERMINAL_STAGES},
             "$or": [{"booking_id": booking_id}, {"customer_phone": phone}]},
            {"$set": {
                "stage": "payment_check",
                "payment_check_pending": True,
                "payment_proof_submitted": True,
                "claimed_paid": True,
                "booking_status": "pending_payment",
            },
            # The guest has already paid and sent proof — this booking is now
            # waiting on a human, not on the guest. The TTL index (see
            # booking_repository.ensure_indexes) deletes any pending_payment
            # hold 300s past hold_expires_at, so leaving the field set would
            # erase the booking, the proof and the payment record while the
            # front desk is still verifying it.
            "$unset": {
                "hold_expires_at": "",
            }}
        )
    except Exception as e:
        logger.warning(f"Error marking booking pending in stage 1b: {e}")

    # 2. Raise Owner Task for front desk verification (idempotent: avoid spamming duplicate open tasks)
    try:
        existing_tasks, _ = await task_repository.list_tasks(
            hotel_id=hotel_id,
            status="OPEN",
            limit=5,
        )
        task_already_exists = any(
            (t.get("subtype") == "PAYMENT_CONFIRMATION" or t.get("task_type") == "payment_check" or (t.get("metadata") or {}).get("task_type") == "payment_check")
            and (t.get("booking_id") == booking_id or t.get("customer_phone") == phone)
            for t in existing_tasks
        )

        if not task_already_exists:
            from app.workflows.nodes.stage_handlers import create_stage_task
            await create_stage_task(
                hotel_id=hotel_id,
                session_id=state.get("session_id") or "",
                title=f"Payment Verification Needed ({booking_id})",
                description=f"Guest submitted payment proof / claim: '{user_msg}'. Phone: {phone}. Verify transaction in bank / gateway.",
                task_type="FRONT_DESK",
                subtype="PAYMENT_CONFIRMATION",
                department="FRONT_DESK",
                priority="HIGH",
                booking_id=booking_id if booking_id != "BK-PENDING" else None,
                customer_phone=phone,
                metadata={
                    "guest_message": user_msg,
                    "stage": "payment_check",
                    "task_type": "payment_check",
                },
            )
            logger.info("STAGE_1B_OWNER_TASK_RAISED | booking=%s", booking_id)
    except Exception as e:
        logger.warning(f"Error raising owner task in stage 1b: {e}")

    # 3. Reassuring response matching Sir's exact requirements:
    # Thank guest, state front desk is verifying, keep pending, NO confirmed, NO upsell, NO receipt card
    response_text = (
        "Namaste! 🙏 Humne aapka payment confirmation note kar liya hai.\n\n"
        "Humari front desk team abhi aapka payment verify kar rahi hai. "
        "Jaise hi bank/UPI account se verify hota hai, hum aapko official confirmation slip bhej denge.\n\n"
        f"• *Booking Reference:* `{booking_id}`\n"
        "• *Current Status:* ⏳ **PENDING VERIFICATION**\n\n"
        "Aapko kuch aur karne ki zaroorat nahi hai, kripya thoda intezaar karein."
    )

    # The paragraph above answers a payment claim. Every other turn in this
    # stage — a greeting, "send me my confirmation", "my check-in is in a few
    # hours" — used to get it verbatim too, which reads as a bot that is not
    # listening. These variants answer the question that was asked and state
    # the same unconfirmed status; `_status_reply` returns None for a payment
    # claim, so the wording Sir specified is untouched for the case it was
    # written for.
    if get_settings().PAYMENT_CHECK_STATUS_REPLIES_ENABLED and not _CLAIM_RE.search(user_msg):
        tailored = _status_reply(user_msg, booking_id)
        if tailored:
            response_text = tailored

    return {
        **state,
        "booking_stage": "payment_check",
        "lifecycle_stage": "payment_check",
        "bot_response": text_card(response_text),
        "next_action": "awaiting_payment_verification",
    }
