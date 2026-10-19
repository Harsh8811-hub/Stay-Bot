"""
Stage 2 Handler — Confirmed Booking & Pre-Arrival Concierge.

Sir's Specification:
- A booking enters Stage 2 only when Razorpay webhook succeeds, hotelier confirms, or manual payment marked.
- ZERO SELLING.
- Never restart booking flow, never ask dates, never show room options or welcome menu.
- Never call confirmation message a bill, receipt, or invoice — MUST be called 'confirmation slip'.
- S53: 'Rate kya hai?' -> Show what guest already paid. No room menu!
- S54: 'Meri booking confirm hai?' -> Show booking ID, dates, room, name, amount paid, balance.
- S55: 'Station se kaise aayein?' -> Route from Varanasi Junction / airport, map pin, front desk phone. No selling.
- S56: 'Hi' -> Personal greeting + booking summary. No welcome menu, no continue/fresh start.
- S57: 'Ek aur room chahiye' -> Ask once if separate booking; if yes, open separate Lead draft.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.config.settings import get_settings
from app.prompts.templates import text_card
from app.repositories import task_repository
from app.services.hotel_identity import HotelIdentity, resolve_hotel_identity
from app.workflows.nodes.stage_handlers.kb_answer import knowledge_base_answer
from app.workflows.state import ConversationState

logger = logging.getLogger(__name__)

# A confirmed guest asking to move their stay. This is a regex, not the
# substring list it replaced, because that list held `"change date"` — singular,
# no words in between — and so missed the most natural phrasing there is.
# Measured on a replay turn 2026-10-10: "can you change the dates to next week?"
# raised no task at all, because "change the dates" is not "change date".
_CHANGE_DATE_RE = re.compile(
    r"\b(?:change|move|shift|resched\w*|postpone|prepone)\b[^.?!]{0,20}?(?:dates?|stay|booking)\b"
    r"|\b(?:date|dates)\s+change\b"
)

# The rest of the amendment cues carry no such plural/word-order trap and stay
# plain substrings.
_AMENDMENT_CUES = (
    "modify", "early check in", "early checkin",
    "late checkout", "late check out", "tarikh badalna",
)

# A question ABOUT the policy is not a request to act on it.
#
# `is_cancel` below is a plain substring test, and "cancellation" contains
# "cancel" — so "what's your cancellation policy?" from a confirmed guest
# raised a BOOKING_CANCELLATION task and woke the front desk for a guest who
# only wanted to read the terms. Same shape as the `"change date"` miss above:
# the cue and the thing it names are not the same word.
#
# The discrimination is adjacency, not a keyword list. A policy noun directly
# after the cancel/refund word ("cancellation policy", "refund rules",
# "cancellation fee") is a question; the same nouns sitting *before* it
# ("policy on cancellation") are caught by the second alternative. A genuine
# request never puts a policy noun next to the verb — "I want to cancel my
# booking, what's your policy?" does not match, because the two are not
# adjacent, and it stays a cancellation.
_POLICY_QUESTION_RE = re.compile(
    r"\b(?:cancel\w*|refund\w*)\s+(?:polic\w*|rules?|terms?|charges?|fees?|process)\b"
    r"|\b(?:polic\w*|rules?|terms?|charges?|fees?)\b[^.?!]{0,20}\b(?:cancel\w*|refund\w*)\b"
)


#: A request to see the rooms, in either of the two shapes it arrives in.
#:
#: The buttons are resolved through `app.utils.button_registry` — the same
#: declaration `button_card` validates against — so the tap, the id the CRM may
#: one day forward, and a guest typing the title all land on one answer. The
#: regex covers the phrasings no button carries ("show me available rooms").
#:
#: Deliberately not "room" on its own: "does the deluxe room have AC" is a
#: factual question the knowledge base answers, and reads nothing like a request
#: to browse.
_ROOMS_BROWSE_RE = re.compile(
    r"\broom\s*(?:types?|options?|list|catalogue|catalog)\b"
    r"|\bavailable\s+rooms?\b"
    r"|\bkambre?\s*(?:aur|ke|ka)\s*(?:rate|daam)\b"
    r"|\brate\s*list\b",
    re.IGNORECASE,
)


def _wants_room_browse(user_msg: str) -> bool:
    """Whether this turn asks to see the rooms, rather than asking about them."""
    from app.utils.button_registry import resolve_button

    tapped = resolve_button(user_msg)
    if tapped is not None and tapped.action in ("SHOW_ROOMS", "BOOK_NOW"):
        return True
    return bool(_ROOMS_BROWSE_RE.search(user_msg or ""))


def _format_confirmation_slip(
    booking_id: str,
    guest_name: str,
    check_in: str,
    check_out: str,
    room_name: str,
    adults: int,
    amount_paid: float,
    total_amount: float,
    ident: HotelIdentity,
    has_payment_record: bool = True,
) -> str:
    balance = max(0.0, total_amount - amount_paid)
    # The caller passes `has_payment_record=False` for a CRM-confirmed guest, who has no
    # row of ours — their `amount_paid` is a default 0.0, not a fact. Printing
    # "Advance Paid: Rs 0" to a guest who has paid is worse than saying nothing, and the
    # CRM is not a payment record to read one from instead.
    if not has_payment_record:
        balance_line = "• *Payment:* Front desk ke paas record hai — woh detail bata denge."
    elif balance > 0:
        balance_line = f"• *Balance Due at Check-in:* Rs {balance:,.0f}"
    else:
        balance_line = "• *Payment Status:* Fully Paid (Nil Balance)"

    # Address and phone come from the property's own record. Each line is
    # dropped entirely when the record has nothing, rather than printed empty
    # or borrowed from another hotel.
    address_line = f"📍 *Hotel Address:* {ident.address}\n" if ident.address else ""
    # Same reason as the caller's fragments: a CRM-confirmed guest has no id of ours,
    # and this slip is printed straight at the guest.
    booking_id_line = f"• *Booking ID:* `{booking_id}`\n" if booking_id else ""
    # Empty for a CRM-confirmed guest the CRM also has no room for — printing the label
    # with nothing after it reads worse than leaving the line out.
    room_line = f"• *Room Category:* {room_name}\n" if room_name else ""
    advance_line = (
        f"• *Advance Paid:* Rs {amount_paid:,.0f}\n" if has_payment_record else ""
    )

    return (
        f"🏨 *{ident.name} — Booking Confirmation Slip*\n\n"
        f"Aapka reservation fully confirmed hai! Swagat ki taiyari hai:\n\n"
        f"{booking_id_line}"
        f"• *Guest Name:* {guest_name}\n"
        f"• *Dates:* {check_in} se {check_out}\n"
        f"{room_line}"
        f"• *Guests:* {adults} Guest(s)\n"
        f"{advance_line}"
        f"{balance_line}\n\n"
        f"{address_line}"
        f"📞 *{ident.front_desk_clause()}*\n\n"
        f"_Kisi bhi madad ke liye aap seedhe is chat par pooch sakte hain._"
    )


async def handle_stage_confirmed(state: ConversationState) -> ConversationState:
    hotel_id = state.get("hotel_id", 111111)
    phone = state.get("customer_identifier") or ""
    user_msg = (state.get("user_message") or "").strip()
    user_msg_lower = user_msg.lower()
    ident = await resolve_hotel_identity(hotel_id, state)
    phone_md = f"**{ident.phone}**" if ident.phone else "**the front desk at reception**"
    # The map pin is only offered when this property actually has one. It used
    # to default to Hotel Sahu's pin for every tenant that was not Sanidhyam.
    map_line = f"🗺️ *Google Maps Pin:* {ident.map_link}\n\n" if ident.map_link else ""
    location_line = f"📍 *Hotel Location:* {ident.address}\n" if ident.address else ""

    active_booking = state.get("active_booking") or {}
    raw_booking_id = (
        active_booking.get("booking_id")
        or state.get("active_booking_id")
        or state.get("booking_id")
    )
    # A booking MUST have a valid, non-placeholder booking ID to be in Stage 2 Confirmed
    # — unless the stay was confirmed by staff in the CRM, which leaves no row of ours.
    from app.utils.helpers import guest_is_confirmed

    has_valid_booking = guest_is_confirmed(state, raw_booking_id)

    if not has_valid_booking:
        logger.warning(
            "STAGE_2_CONFIRMED_GHOST_GUARD | hotel=%s phone=%s has no active booking in DB; demoting to lead",
            hotel_id, phone,
        )
        demoted_state = {
            **state,
            "booking_stage": "lead",
            "lifecycle_stage": "lead",
            "active_booking": {},
            "active_booking_id": None,
            "booking_id": None,
        }
        # Check if guest message is a refusal / decline / negative response to a campaign or offer
        is_refusal = any(
            p in user_msg_lower for p in (
                "no thanks", "no thank you", "nahi chahiye", "nahi", "not interested",
                "not needed", "dont need", "cancel", "stop", "nope", "nhi"
            )
        ) or user_msg_lower in ("no", "na", "n")
        if is_refusal:
            resp = (
                "Dhanyawad! Agar bhavishya mein aapko Varanasi mein stay ki zaroorat ho, "
                "toh humse zaroor sampark karein. Shubh din! 🙏"
            )
            return {
                **demoted_state,
                "next_action": None,
                "bot_response": text_card(resp),
            }

        is_greeting = any(user_msg_lower == g for g in ("hi", "hello", "hey", "namaste", "namaskar"))
        if is_greeting:
            resp = (
                f"🏨 *Namaste! Welcome to {ident.name}.* 🙏\n\n"
                "Main aapki booking mein kya madad kar sakta hoon? "
                "Kripya apni dates aur number of guests batayein."
            )
            return {
                **demoted_state,
                "next_action": "need_dates",
                "bot_response": text_card(resp),
            }

        resp = (
            f"Namaste! {ident.name} mein stay booking ya room enquiry ke liye kripya "
            f"apni check-in dates aur guests ki sankhya batayein. Main aapko best options dikhata hoon! 🏨"
        )
        return {
            **demoted_state,
            "next_action": "need_dates",
            "bot_response": text_card(resp),
        }

    booking_id = raw_booking_id
    # A stay confirmed by staff in the CRM has no booking id of ours, so the strings
    # below would otherwise print "(None)" at the guest. Build the three display
    # fragments once, here, and use them at the guest-facing sites. `booking_id`
    # itself stays raw: the staff-task payloads take it as a field, and a null id
    # there is honest — staff can see this guest has no bot-side row.
    booking_ref = f" ({booking_id})" if booking_id else ""
    booking_ref_bold = f" **{booking_id}**" if booking_id else ""
    booking_id_line = f"• *Booking ID:* `{booking_id}`\n" if booking_id else ""
    guest_name = active_booking.get("guest_name") or state.get("guest_name") or "Valued Guest"
    check_in = str(active_booking.get("check_in") or active_booking.get("check_in_date") or state.get("check_in") or "Upcoming Stay")
    check_out = str(active_booking.get("check_out") or active_booking.get("check_out_date") or state.get("check_out") or "")
    # No hardcoded default. A guest with a booking row reads their room off it; a
    # CRM-confirmed guest has no row, so their room comes from the CRM's own record of
    # the stay. "Deluxe Room" used to stand here as the last resort and was shown to
    # real guests who had booked something else entirely.
    crm_room_type = state.get("crm_confirmed_room_type") if state.get("crm_confirmed_stay") else None
    room_name = str(
        active_booking.get("room_name")
        or active_booking.get("room_type")
        or state.get("room_type_preference")
        or crm_room_type
        or ""
    ).strip()
    room_disp = f" ({room_name})" if room_name else ""
    room_with_dates = (
        f" ({room_name}, {check_in} se {check_out})" if room_name
        else f" ({check_in} se {check_out})"
    )
    adults = int(active_booking.get("adults") or state.get("adults") or 2)
    amount_paid = float(active_booking.get("amount_paid") or state.get("amount_paid") or 0.0)
    total_amount = float(active_booking.get("total_price") or active_booking.get("amount") or state.get("total_price") or amount_paid or 0.0)

    logger.info(
        "STAGE_2_CONFIRMED_DISPATCH | hotel=%s phone=%s booking_id=%s msg='%s'",
        hotel_id, phone, booking_id, user_msg[:60],
    )

    # ── S57: Explicit Request for Another / Second Room ──────────────────────
    # A guest who taps "Room types & rates" or "Book a room" is asking the same
    # thing S57 exists for — they have not said "for my current stay". ZERO
    # SELLING forbids showing the catalogue here (see the tail of this handler),
    # so the answer is the one already designed for a second room: say the stay
    # is safe, ask once whether this is a separate booking. The tap used to fall
    # through to the directions/parking paragraph, answering a question the guest
    # had not asked. Pranav, 2026-10-18.
    wants_rooms = (
        get_settings().CONFIRMED_ROOMS_ASK_SEPARATE_BOOKING
        and _wants_room_browse(user_msg)
    )
    is_separate_booking_trigger = wants_rooms or any(
        p in user_msg_lower for p in (
            "ek aur room", "another room", "one more room", "new booking", "separate booking",
            "naya room", "doosra room", "dusra room", "2nd room", "second room", "additional room"
        )
    )
    clarification_msg = (
        f"Namaste {guest_name}! 🙏 Aapki ek booking{booking_ref} already {check_in} se {check_out} ke liye confirmed hai.\n\n"
        f"Kya aap is confirmed stay ke alawa ek **naya alag booking** karna chahte hain?\n"
        f"👉 Kripya reply karein **'Yes, new booking'** ya **'No'**."
    )
    if is_separate_booking_trigger and not state.get("separate_booking_clarification_asked"):
        # Ask exactly ONCE: Is this a new booking, separate from your current stay?
        return {
            **state,
            "booking_stage": "confirmed",
            "lifecycle_stage": "confirmed",
            "separate_booking_clarification_asked": True,
            "bot_response": text_card(clarification_msg),
            "next_action": "awaiting_separate_booking_confirmation",
        }

    # Asked once, and the guest asked again without answering: repeat the
    # question. Falling through here would send the canned paragraph after a
    # second, explicit request for rooms — the non-answer this branch removes.
    if is_separate_booking_trigger and wants_rooms:
        return {
            **state,
            "booking_stage": "confirmed",
            "lifecycle_stage": "confirmed",
            "bot_response": text_card(clarification_msg),
            "next_action": "awaiting_separate_booking_confirmation",
        }

    # If guest confirmed separate booking creation
    if state.get("separate_booking_clarification_asked") and any(w in user_msg_lower for w in ("yes", "haan", "ha", "new booking", "naya")):
        # Transition to new Lead draft alongside existing confirmed booking
        lead_start_msg = (
            f"Theek hai {guest_name}! Hum naye booking ke liye details note kar rahe hain. "
            f"(Aapki pehli confirmed booking{booking_ref_bold} bilkul safe hai).\n\n"
            f"Naye room ke liye aap kis date par check-in karna chahenge aur kitne guests honge?"
        )
        return {
            **state,
            "booking_stage": "lead",
            "lifecycle_stage": "lead",
            "separate_booking_clarification_asked": False,
            "is_separate_booking_inquiry": True,
            "selected_rooms": [],
            "allocation_options": [],
            "price_breakdown": None,
            "total_price": None,
            "bot_response": text_card(lead_start_msg),
            "next_action": "need_dates",
        }

    # ── S53: "Rate kya hai?" / Price / Tariff Ask ────────────────────────────
    # Must show what guest ALREADY paid for current booking. ZERO selling, NO room menu!
    is_rate_ask = any(
        p in user_msg_lower for p in (
            "rate kya hai", "rate kya h", "price kya hai", "kitna rate", "room rate", "tariff",
            "price list", "charges kya hai", "how much", "what is the rate", "price", "pricing"
        )
    ) and not any(p in user_msg_lower for p in ("station", "airport", "taxi", "cab", "auto"))

    if is_rate_ask:
        # A CRM-confirmed guest has no row of ours, so `amount_paid` is a default 0.0
        # rather than a fact. "Aapne Rs 0 pay kiya hai" to someone who has paid is worse
        # than saying nothing, and the CRM is not a payment record (see
        # `crm_confirmation_repository`), so we must not read an amount from it either.
        # Say only what we can stand behind and hand them to the front desk.
        paid_line = (
            "Payment ka record front desk ke paas hai — woh aapko exact detail bata denge.\n"
            if state.get("crm_confirmed_stay")
            else f"Is booking ke liye aapne **Rs {amount_paid:,.0f}** pay kiya hai"
                 f"{room_with_dates}.\n"
        )
        resp = (
            f"Namaste {guest_name}! 🙏 Aapki booking{booking_ref_bold} already fully confirmed hai.\n\n"
            f"{paid_line}"
            f"Aapka stay secure hai, kisi naye quote ki zaroorat nahi hai. Hum aapka swagat karne ke liye taiyar hain! 🏨"
        )
        return {
            **state,
            "booking_stage": "confirmed",
            "lifecycle_stage": "confirmed",
            "bot_response": text_card(resp),
        }

    # ── S53b: Bill / Invoice / Receipt Request ───────────────────────────────
    is_bill_ask = any(
        w in user_msg_lower for w in ("bill", "invoice", "receipt", "payment slip", "vej dijiye", "bhej dijiye", "hisab")
    ) and not any(w in user_msg_lower for w in ("room chahiye", "booking", "naya room", "rate kya"))
    if is_bill_ask:
        try:
            from app.workflows.nodes.stage_handlers import create_stage_task
            await create_stage_task(
                hotel_id=hotel_id,
                session_id=state.get("session_id") or "",
                title=f"Bill / Invoice Request: Booking {booking_id} ({guest_name})",
                description=f"Confirmed Guest {guest_name} ({phone}) requested bill/invoice: '{user_msg}'. Booking: {booking_id}. Stay dates: {check_in} to {check_out}.",
                task_type="FRONT_DESK",
                subtype="BILLING_INQUIRY",
                department="FRONT_DESK",
                priority="HIGH",
                booking_id=booking_id,
                customer_name=guest_name,
                customer_phone=phone,
                metadata={
                    "guest_message": user_msg,
                    "stage": "confirmed",
                    "task_type": "billing_inquiry",
                },
            )
        except Exception as e:
            logger.warning(f"Error creating confirmed bill task: {e}")

        resp = (
            f"Namaste {guest_name}! Humne aapke bill/receipt ka anurodh front desk team ko bhej diya hai.\n\n"
            f"{booking_id_line}"
            f"• *Stay Dates:* {check_in} se {check_out}\n\n"
            f"Humari front desk team jald hi aapse bill share karegi. Main aapki aur kya madad kar sakta hoon? 🙏"
        )
        return {
            **state,
            "booking_stage": "confirmed",
            "lifecycle_stage": "confirmed",
            "bot_response": text_card(resp),
        }

    # ── S54: "Meri booking confirm hai?" / Status Ask ────────────────────────
    is_confirm_check = any(
        p in user_msg_lower for p in (
            "booking confirm", "confirm hai", "status kya hai", "check booking", "is my booking",
            "meri booking", "slip", "details", "booking details"
        )
    )
    if is_confirm_check:
        slip = _format_confirmation_slip(
            booking_id=booking_id,
            guest_name=guest_name,
            check_in=check_in,
            check_out=check_out,
            room_name=room_name,
            adults=adults,
            amount_paid=amount_paid,
            total_amount=total_amount,
            ident=ident,
            has_payment_record=not state.get("crm_confirmed_stay"),
        )
        return {
            **state,
            "booking_stage": "confirmed",
            "lifecycle_stage": "confirmed",
            "bot_response": text_card(slip),
        }

    # ── S55: "Station se kaise aayein?" / Airport / Directions ───────────────
    # Provide directions, distance, map pin, front desk contact. NO sales pitch!
    is_directions_ask = any(
        p in user_msg_lower for p in (
            "station", "kaise aayein", "kaise pahunchein", "directions", "location", "address",
            "airport", "rasta", "route", "map", "how to reach", "cantt", "banaras"
        )
    )
    if is_directions_ask:
        resp = (
            f"Namaste {guest_name}! 🏨 {ident.name} pahunchna bohot aasan hai:\n\n"
            f"{location_line}"
            f"{map_line}"
            f"🚆 *Varanasi Junction (Cantt Railway Station) Se:* (~3.5 km)\n"
            f"• Station ke bahar se direct Auto / E-Rickshaw Godowlia Chowk ke liye milte hain (approx 15-20 mins, ₹20-30 shared ya ₹100-150 private).\n"
            f"• Godowlia Chowk se Dashashwamedh road par paidal chal kar {ident.name} pahunch sakte hain.\n\n"
            f"✈️ *Lal Bahadur Shastri Airport (Babatpur) Se:* (~26 km)\n"
            f"• Airport se pre-paid taxi ya cab (Ola/Uber) lagbhag 45-50 minutes leti hai.\n\n"
            f"🚗 *Parking:* Hotel ke paas vehicle parking suvidha uplabdh hai.\n"
            f"📞 *Front Desk / Navigation Help:* {phone_md}\n\n"
            f"_Aapki booking{booking_ref} {check_in} ke liye confirmed hai. Happy journey!_"
        )
        return {
            **state,
            "booking_stage": "confirmed",
            "lifecycle_stage": "confirmed",
            "bot_response": text_card(resp),
        }

    # ── S56: "Hi" / Greeting ─────────────────────────────────────────────────
    # Personal greeting + booking summary. NO welcome menu, NO continue/fresh start.
    is_greeting = any(user_msg_lower == g for g in ("hi", "hello", "hey", "namaste", "namaskar", "pranam", "good morning", "good evening"))
    if is_greeting:
        resp = (
            f"Namaste {guest_name}! 🙏 Welcome to {ident.name}.\n\n"
            f"Aapki reservation{booking_ref} {check_in} se {check_out}{room_disp} ke liye confirmed hai.\n\n"
            f"Aapki aane ki taiyari ya Varanasi yatra se judi kisi bhi jankari (directions, Kashi Vishwanath darshan, Aarti timing) ke liye main aapki kya madad kar sakta hoon?"
        )
        return {
            **state,
            "booking_stage": "confirmed",
            "lifecycle_stage": "confirmed",
            "bot_response": text_card(resp),
        }

    # ── Cancel a confirmed booking ───────────────────────────────────────────
    #
    # Split out of a shared change/cancel branch on 2026-10-10 (Pranav). Until
    # then a guest asking to cancel got the same generic acknowledgement as a
    # guest asking to move a date, filed under a CHECKOUT_REQUEST subtype — so an
    # operator could not tell "wants to leave early" from "wants their money
    # back" — and was told nothing at all about their refund.
    #
    # The booking is NOT cancelled here. Cancelling is a money decision the
    # front desk owns, so this raises the task and hands the guest the policy;
    # nothing in this branch writes to the bookings collection.

    # ── A question ABOUT the policy is not a request to act on it ────────────
    #
    # `_POLICY_QUESTION_RE` above carries the reasoning. Measured live against
    # this clone on 2026-10-11: "what is the cancellation policy" from a
    # confirmed guest raised a BOOKING_CANCELLATION task, so the front desk was
    # asked to approve a cancellation nobody had requested.
    #
    # The answer is the same text the cancel branch quotes, from the same
    # service — minus the task, and minus the "we've sent your request" framing,
    # which would be untrue here.
    if _POLICY_QUESTION_RE.search(user_msg_lower):
        from app.services.cancellation_policy_service import get_cancellation_policy_text

        policy = await get_cancellation_policy_text(hotel_id)
        resp = (
            f"Namaste {guest_name}! 🙏 Aapki booking{booking_ref_bold} "
            f"({check_in} se {check_out}) confirmed hai.\n\n"
            f"📋 *Cancellation & Refund Policy*\n{policy}\n\n"
            f"Agar aap cancel karna chahte hain to bas bata dijiye — hum request "
            f"front desk team ko forward kar denge."
        )
        return {
            **state,
            "booking_stage": "confirmed",
            "lifecycle_stage": "confirmed",
            "bot_response": text_card(resp),
        }

    # Amendment is decided first, on this one message, and cancellation defers
    # to it. "change the dates, don't cancel" contains "cancel", so the bare
    # substring test took it to the cancellation branch and raised a task asking
    # the front desk to approve the very thing the guest had ruled out.
    #
    # Deciding both here rather than reordering the two blocks keeps the
    # precedence in one place — the amendment block below reads this variable
    # instead of recomputing it, so the two cannot drift.
    is_amendment = bool(_CHANGE_DATE_RE.search(user_msg_lower)) or any(
        p in user_msg_lower for p in _AMENDMENT_CUES
    )

    is_cancel = (
        any(p in user_msg_lower for p in ("cancel", "cancellation", "refund"))
        and not is_amendment
    )
    if is_cancel:
        try:
            from app.workflows.nodes.stage_handlers import create_stage_task

            await create_stage_task(
                hotel_id=hotel_id,
                title=f"Cancellation Request for Confirmed Booking {booking_id}",
                description=(
                    f"Guest {guest_name} ({phone}) requested cancellation: '{user_msg}'. "
                    f"Stay dates: {check_in} to {check_out}. Booking is still CONFIRMED — "
                    f"cancellation and any refund are for the front desk to approve."
                ),
                task_type="FRONT_DESK",
                subtype="BOOKING_CANCELLATION",
                department="FRONT_DESK",
                priority="HIGH",
                booking_id=booking_id,
                customer_phone=phone,
                metadata={
                    "guest_message": user_msg,
                    "stage": "confirmed",
                    "task_type": "cancellation",
                    "action_required": "approve_or_decline_cancellation",
                    "check_in": str(check_in),
                    "check_out": str(check_out),
                    "room_name": room_name,
                },
            )
        except Exception as e:
            logger.warning(f"Error creating cancellation task: {e}")

        # The policy is the whole point of this reply, so it is fetched and never
        # replaced by a generic line: the service returns a hotel's own bands, or
        # the default wording, and only ever an empty string would be a problem —
        # which it cannot be.
        from app.services.cancellation_policy_service import get_cancellation_policy_text

        policy = await get_cancellation_policy_text(hotel_id)

        resp = (
            f"Humne aapka cancellation request front desk team ko bhej diya hai.\n\n"
            f"{booking_id_line}"
            f"• *Dates:* {check_in} se {check_out}\n\n"
            f"📋 *Cancellation & Refund Policy*\n{policy}\n\n"
            f"Team aapke request ko verify karke refund process aage badhayegi. "
            f"Kisi urgent sahayata ke liye aap reception par call kar sakte hain: {phone_md}."
        )
        return {
            **state,
            "booking_stage": "confirmed",
            "lifecycle_stage": "confirmed",
            "bot_response": text_card(resp),
        }

    # ── Amend a confirmed booking (dates, room, early/late, etc.) ────────────
    #
    # Also split out on 2026-10-10. The amendment itself is not applied here —
    # repricing and re-reserving a confirmed booking is the front desk's call —
    # but the request is now *recorded* on the task in structured form, so an
    # operator sees what the guest asked for against the dates currently held,
    # rather than a free-text sentence they have to parse.
    #
    # `is_amendment` is decided up at the cancellation block, which defers to
    # it. Recomputed here, the two answers could drift apart and a message could
    # be both or neither.
    if is_amendment:
        try:
            from app.workflows.nodes.stage_handlers import create_stage_task

            await create_stage_task(
                hotel_id=hotel_id,
                title=f"Amendment Request for Confirmed Booking {booking_id}",
                description=(
                    f"Guest {guest_name} ({phone}) requested a change: '{user_msg}'. "
                    f"Currently held: {check_in} to {check_out}{room_disp}. "
                    f"Booking is CONFIRMED — the change is for the front desk to "
                    f"reprice and confirm."
                ),
                task_type="FRONT_DESK",
                subtype="BOOKING_AMENDMENT",
                department="FRONT_DESK",
                priority="HIGH",
                booking_id=booking_id,
                customer_phone=phone,
                metadata={
                    "guest_message": user_msg,
                    "stage": "confirmed",
                    "task_type": "amendment",
                    "action_required": "reprice_and_confirm_change",
                    # The dates being changed *from*. Without these the task is a
                    # sentence and the operator has to open the booking to know
                    # what is actually being moved.
                    "current_check_in": str(check_in),
                    "current_check_out": str(check_out),
                    "room_name": room_name,
                },
            )
        except Exception as e:
            logger.warning(f"Error creating amendment task: {e}")

        resp = (
            f"Humne aapka change request front desk team ko note kar diya hai.\n\n"
            f"{booking_id_line}"
            f"• *Current Dates:* {check_in} se {check_out}\n"
            f"• *Request:* {user_msg}\n\n"
            f"Team availability aur rate check karke aapko naye dates aur amount ke saath "
            f"confirm karegi. Kisi urgent sahayata ke liye aap reception par call kar sakte hain: {phone_md}."
        )
        return {
            **state,
            "booking_stage": "confirmed",
            "lifecycle_stage": "confirmed",
            "bot_response": text_card(resp),
        }

    # ── Sightseeing / Temple / Aarti in Advance ───────────────────────────────
    if any(p in user_msg_lower for p in ("aarti", "ganga", "darshan", "vishwanath", "temple", "ghat", "kashi")):
        resp = (
            f"{ident.name} se Kashi Vishwanath Mandir aur Dashashwamedh Ghat bohot kareeb hain:\n\n"
            f"🛕 *Shri Kashi Vishwanath Temple:* Gate No. 4 sirf paidal doori par hai.\n"
            f"🌊 *Ganga Aarti:* Dashashwamedh Ghat par daily sham 6:45 PM – 7:00 PM par shuru hoti hai ({ident.name} se sirf 5-10 minute walk).\n"
            f"🚤 *Boat Rides:* Morning sunrise boat tour aur evening aarti boat tour reception se arrange ho sakte hain.\n\n"
            f"Kisi aur jankari ke liye kripya batayein!"
        )
        return {
            **state,
            "booking_stage": "confirmed",
            "lifecycle_stage": "confirmed",
            "bot_response": text_card(resp),
        }

    # ── S56b: Guest Acknowledgment / Polite Decline ───────────────────────────
    is_declined_ack = any(
        user_msg_lower == p or user_msg_lower.startswith(p + " ") or user_msg_lower.endswith(" " + p)
        for p in ("no thanks", "no thank you", "nahi chahiye", "not needed", "nope", "nahi", "no")
    )
    if is_declined_ack:
        resp = (
            f"Bilkul {guest_name}! Swagat ki poori taiyari hai. "
            f"Kisi bhi sahayata (directions, Kashi darshan, aarti timing) ke liye hum yahan hain. Shubh din! 🙏"
        )
        return {
            **state,
            "booking_stage": "confirmed",
            "lifecycle_stage": "confirmed",
            "bot_response": text_card(resp),
        }

    is_thanks_ack = any(p in user_msg_lower for p in ("thank you", "thanks", "dhanyawad", "shukriya"))
    if is_thanks_ack:
        resp = (
            f"You're most welcome, {guest_name}! Hum {ident.name} mein aapka swagat karne ke liye utsuk hain. "
            f"Kisi aur jankari ke liye batayein. Shubh yatra! 🏨"
        )
        return {
            **state,
            "booking_stage": "confirmed",
            "lifecycle_stage": "confirmed",
            "bot_response": text_card(resp),
        }

    is_ok_ack = any(
        p in user_msg_lower
        for p in ("ok", "okay", "theek hai", "thik hai", "achha", "accha", "theek h", "thik h", "got it", "samajh gaya")
    )
    if is_ok_ack:
        resp = (
            f"Ji {guest_name}! Hum aapke swagat ke liye taiyar hain. "
            f"Front desk se sampark ke liye number hai: {phone_md}. Shubh din! 🙏"
        )
        return {
            **state,
            "booking_stage": "confirmed",
            "lifecycle_stage": "confirmed",
            "bot_response": text_card(resp),
        }

    # ── S58: Everything Else — Answer From the Hotel's Knowledge Base ─────────
    #
    # This tail used to be a fixed paragraph naming directions, check-in time,
    # parking and Kashi darshan — so "what is the wifi password", "do you allow
    # pets" and "is there a restaurant" all got the same non-answer, and the
    # hotel's own knowledge base (14 articles plus structured transit details for
    # Sahu) was never read for a confirmed guest at all.
    #
    # The cause is routing, not this file: `routing.py:40` sends every
    # `lifecycle_stage == "confirmed"` turn here and returns, so the
    # `intent == "FAQ"` branch at `routing.py:99` is unreachable for a confirmed
    # guest. Answering means calling the FAQ node from here.
    #
    # Three things this call must NOT be allowed to do, all of them reverting to
    # the canned paragraph below instead:
    #
    #   * Move the guest out of the confirmed stage. `handle_faq` spreads the
    #     incoming state, so it preserves the stage today — but that is a
    #     property of a different module, and re-asserting it here is what keeps
    #     a confirmed guest confirmed.
    #   * Sell. `handle_faq`'s first branch redirects a room-pricing question to
    #     `handle_room_query`, which shows option cards. That is the right answer
    #     in the booking flow and a contradiction of this stage's ZERO SELLING
    #     contract. `next_action == "faq_answered"` is how an answer is
    #     distinguished from a redirect, and it is the only outcome accepted.
    #   * Raise. `handle_faq` reaches Gemini, ChromaDB, Mongo and the persona
    #     rewriter; a knowledge-base failure must cost the guest a fallback
    #     sentence, never a turn with no reply.
    # `knowledge_base_answer` owns all three of those, and its docstring carries
    # the reasoning. It lives in `kb_answer.py` rather than here because
    # `stage_in_house` now needs the identical call — Pranav, 2026-10-11: *"faq
    # mode till check out"* — and two copies would drift.
    #
    # It also guards a leak this block's own `faq_answered` check did not, and
    # that only showed up once the bot started answering at all: "show me
    # available rooms" came back as a room catalogue with rate plans. The
    # hotel's room list and rate plans are *in* the knowledge base, so it
    # arrives as a plain `faq_answered` and the check above waved it through.
    # Measured 2026-10-11; at HEAD the same message got the canned paragraph
    # below.
    #
    # Everything it declines comes back as `None` and lands on that paragraph —
    # which is why the paragraph is still here.
    kb_state = await knowledge_base_answer(
        state, stage="confirmed", log_tag="STAGE_2_CONFIRMED",
    )
    if kb_state is not None:
        return kb_state

    # ── Default Confirmed Guidance (Reassuring, NO Selling) ───────────────────
    resp = (
        f"Namaste {guest_name}! Aapki booking{booking_ref} {check_in} se {check_out} ke liye confirmed hai.\n\n"
        f"Directions, check-in time (12:00 PM), parking, ya Kashi darshan se judi kisi bhi sahayata ke liye hum yahan hain. "
        f"Emergency ya direct baat karne ke liye front desk number hai: {phone_md}."
    )
    return {
        **state,
        "booking_stage": "confirmed",
        "lifecycle_stage": "confirmed",
        "bot_response": text_card(resp),
    }
