"""
Payment Handler — orchestrates follow-up, receipt validation, status check, and link generation.
"""

import logging
from typing import cast

from app.prompts.templates import text_card
from app.workflows.nodes.payment.constants import user_claims_paid
from app.workflows.nodes.payment.followup import try_payment_followup
from app.workflows.nodes.payment.link import generate_payment_link_flow
from app.workflows.nodes.payment.receipt import validate_payment_receipt
from app.workflows.nodes.payment.status import resolve_payment_status
from app.workflows.state import ConversationState

logger = logging.getLogger(__name__)


async def handle_payment(state: ConversationState) -> ConversationState:
    """
    Handle payment flow:
    1. If no payment link exists → generate one (full or partial amount)
    2. If payment link exists → check status via backend
    3. Enforce 30% minimum advance for partial payments
    4. Track amount_paid and amount_due
    """
    intent = state.get("intent", "")
    existing_link = state.get("payment_link")
    payment_status = state.get("payment_status")
    hotel_id = state.get("hotel_id", 0)
    booking_stage = state.get("booking_stage")

    from app.services.override_resolver import (
        resolve_effective_total,
        resolve_amount_paid,
        resolve_amount_due,
        skip_payment_ask,
    )

    total = resolve_effective_total(state, state.get("total_price") or 0) or 0
    amount_paid = resolve_amount_paid(state, state.get("amount_paid", 0.0))
    amount_due = resolve_amount_due(
        state,
        total=total,
        amount_paid=amount_paid,
        fallback=state.get("amount_due", 0.0),
    )

    # Staff confirmed payment / waived — do not ask again
    if skip_payment_ask(state) and amount_due <= 0:
        return cast(ConversationState, {
            **state,
            "total_price": total,
            "amount_paid": amount_paid,
            "amount_due": 0.0,
            "payment_status": state.get("payment_status") or "paid",
            "next_action": "payment_staff_confirmed",
            "bot_response": text_card(
                "Our team has already confirmed your payment for this stay. "
                "No further payment is needed from my side right now. "
                "How else can I help you?"
            ),
        })

    # If booking is already confirmed, do not process payment again
    if booking_stage == "confirmed":
        booking_id = state.get("booking_id") or (
            state.get("active_booking") or {}
        ).get("booking_id")
        # A stay confirmed by staff in the CRM has no row of ours — see
        # `guest_is_confirmed`. Without it this guard demotes a paying guest to a lead.
        from app.utils.helpers import guest_is_confirmed

        has_real_booking = guest_is_confirmed(state, booking_id)
        if not has_real_booking:
            logger.warning("PAYMENT_HANDLER_GHOST_GUARD | Confirmed stage with no real booking_id; demoting to lead")
            return cast(ConversationState, {
                **state,
                "booking_stage": "lead",
                "lifecycle_stage": "lead",
                "next_action": None,
                "bot_response": text_card(
                    "Dhanyawad! Agar aapko stay booking ya room enquiry ke liye koi jankari chahiye, toh kripya batayein. 🙏"
                ),
            })
        booking_id_str = f" **{booking_id}**" if booking_id and str(booking_id).lower() != "none" else ""
        if amount_due > 0:
            return cast(ConversationState, {
                **state,
                "next_action": "payment_pending",
                "bot_response": text_card(
                    f"Your booking{booking_id_str} is confirmed! ✅\n\n"
                    f"💰 Total: ₹{total:,.0f}\n"
                    f"✅ Paid: ₹{amount_paid:,.0f}\n"
                    f"⏳ Due: ₹{amount_due:,.0f}\n\n"
                    f"Please complete the remaining payment at the time of check-in."
                ),
            })
        return cast(ConversationState, {
            **state,
            "next_action": "confirmed",
            "bot_response": text_card(
                f"Your booking{booking_id_str} is already confirmed! ✅ We look forward to welcoming you."
            ),
        })

    user_msg = state.get("user_message", "").lower()
    claims_paid = user_claims_paid(user_msg, intent)

    payment_details = state.get("payment_details")
    screenshot_valid = False
    invalid_payment_msg = None

    # ── PAYMENT FOLLOW-UP ────────────────────────────────────────
    if payment_details and isinstance(payment_details, dict) and (state.get("payment_missing_fields") or []):
        state, early = try_payment_followup(state, payment_details)
        if early is not None:
            return early
        payment_details = state.get("payment_details")

    # ── RECEIPT VALIDATION ───────────────────────────────────────
    if payment_details and isinstance(payment_details, dict):
        result = await validate_payment_receipt(
            state,
            payment_details,
            hotel_id=hotel_id,
            total=total,
            amount_due=amount_due,
            existing_link=existing_link,
        )
        if result.early_return is not None:
            return result.early_return
        state = result.state
        total = result.total
        amount_due = result.amount_due
        screenshot_valid = result.screenshot_valid
        invalid_payment_msg = result.invalid_payment_msg

    # Just-uploaded valid screenshot → pending manual review
    if screenshot_valid:
        return cast(ConversationState, {
            **state,
            "payment_details": None,
            "next_action": "payment_pending",
            "bot_response": text_card(
                "Thank you for the receipt. We have securely stored your payment details and they "
                "are currently under manual review by our team. We will confirm your booking once it is approved."
            ),
        })

    # ── STATUS / APPROVAL CHECK ──────────────────────────────────
    status_result = await resolve_payment_status(
        state,
        hotel_id=hotel_id,
        total=total,
        amount_paid=amount_paid,
        amount_due=amount_due,
        existing_link=existing_link,
        intent=intent,
        claims_paid=claims_paid,
        payment_details=payment_details,
    )
    if status_result is not None:
        return status_result

    # ── HOLD + LINK GENERATION ───────────────────────────────────
    return await generate_payment_link_flow(
        state,
        hotel_id=hotel_id,
        total=total,
        payment_status=payment_status,
        existing_link=existing_link,
        invalid_payment_msg=invalid_payment_msg,
    )
