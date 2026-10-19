"""
Understand Node — combined intent classification + entity extraction.

This is the ONLY node that calls Gemini for understanding.
One call extracts BOTH intent and booking entities.
"""

import datetime
import logging
import re
from typing import Any, cast

from app.config.settings import get_settings
from app.models.extraction_contract import validate_updates
from app.services.gemini_service import (
    classify_payment_turn,
    extract_room_request,
    understand_message,
)
from app.utils.ac_text import (
    ac_preference_from_prompt_reply,
    ac_preference_from_text,
    is_bare_ac_class,
    is_plausible_ac_answer,
    mentions_both_classes,
)
from app.utils.button_registry import (
    MEAL_PLAN_BY_ACTION,
    TAP_ROUTED_ACTIONS,
    resolve_button,
)
from app.utils.competitor_mention import mentions_competitor
from app.utils.date_masking import mask_dates
from app.utils.helpers import today_str
from app.utils.intent_overrides import apply_intent_overrides, is_explicit_start_fresh
from app.utils.pax_resolution import resolve_pax_counts
from app.utils.person_name import extract_person_name
from app.utils.room_requirement_parse import is_an_explicit_split
from app.workflows.session_mapper import state_to_session
from app.workflows.state import ConversationState

logger = logging.getLogger(__name__)

# ── OPT-OUT / OPT-IN COMPLIANCE (ZERO-LLM FAST PATH) ─────────────────────────
# A STOP must be the whole message. The first version anchored with `^` but
# tested with `search`, so any message *beginning* with "stop" opted the guest
# out — "stop by the hotel at 5" wiped `selected_rooms`, `allocation_options`,
# `total_price` and `payment_link`, marked the session `opted_out`, and every
# later message got the unsubscribed reply. The tolerated tail is a short fixed
# list of fillers, deliberately not `.*`: "stop sending me photos" and "stop at
# the hotel" must reach the normal flow. Bare "cancel" is absent on purpose —
# cancelling a booking is a different flow, not a marketing opt-out.
STOP_PATTERN = re.compile(
    r"^\s*(?:please\s+)?"
    r"(?:stop(?:\s+all)?|unsubscribe|unsub|opt\s*out|optout|opt-out|halt|quit|cancel\s+subscription)"
    r"(?:\s+(?:please|now|messages?|msg|it|this))?"
    r"[\s.!]*$",
    re.IGNORECASE,
)

START_PATTERN = re.compile(r"^\s*(?:start|unstop|subscribe|opt\s*in)\s*[.!]?\s*$", re.IGNORECASE)


# ── Language cues ─────────────────────────────────────────────────────────
#
# Moved to module level because the node rebuilt all of these on every message,
# and because the legacy and expanded Hinglish sets need to sit side by side so
# the `LANGUAGE_DETECTION_WEIGHTS_HINDI` gate can be read as a diff.
#
# Which set is in play is decided in `understand`; see `settings.py` for why.
_HINGLISH_CUES_LEGACY = {
    "chahiye", "hai", "hain", "kardo", "karna", "bhai", "bhaiya", "batao", "bataye", "batayiye",
    "kitna", "kitne", "mera", "meri", "hoga", "hogi", "shukriya", "dhanyawad", "alag",
    "kamre", "kamra", "kharcha", "aaram", "sasta", "dono", "dikhaye", "dikhaiye",
    "kaisa", "tarah", "zyada", "sabko", "bade", "bacche", "apna", "saath", "seedha", "jaise",
    "theek", "haan", "nahi", "naam", "namaste"
}

#: The legacy set plus the function words Roman Hindi is actually built out of.
#: Their absence is why "room ki photo bhejo" scored zero Hindi cues and fell
#: through to the English default: `ki`, `ka`, `ke`, `kya`, `mujhe`, `aur` and
#: `wala` were simply not in the list.
_HINGLISH_CUES_EXPANDED = _HINGLISH_CUES_LEGACY | {
    "kya", "mujhe", "ka", "ki", "ke", "ko", "mein", "se", "bhi", "toh", "aur", "ya",
    "aap", "hum", "tum", "dena", "karo", "kar", "milega", "paisa", "paise", "din", "raat",
    "aaj", "kal", "abhi", "accha", "achha", "thik", "kuch", "koi", "wala", "wali", "wale",
    "bhejo", "bhej", "dikhao", "bata", "kaise", "log", "liye", "sirf", "phir", "lekin",
    "magar", "kyun", "kab", "kahan", "jaldi", "raha", "rahi", "gaya", "gayi", "kiya",
    "diya", "liya", "hua", "hui", "sakta", "sakte", "chalo", "bahut", "kam",
    "pehle", "baad", "saal", "baje", "band", "khula", "chalu",
}

#: English cues. Note these are *content* words Hinglish routinely borrows —
#: "room", "book", "family", "price". That is fine under a proportional rule,
#: and it is exactly what made the old `e_count > h_count` rule misfire.
_ENGLISH_CUES = {
    "want", "need", "please", "book", "rooms", "adults", "adult", "children", "child", "checkin", "checkout",
    "night", "nights", "hello", "hi", "sorry", "we", "are", "i", "am", "how", "much", "can", "you",
    "available", "cost", "price", "comfortable", "cheapest", "option", "split", "suggest",
    "separate", "family", "decide", "best"
}

#: Words that answer a question without saying anything about language. A bare
#: "thank you" or "ok" mid-Hinglish conversation carried one English cue, won
#: the English branch, and flipped the rest of the conversation to English.
_LANGUAGE_NEUTRAL_WORDS = {
    "thanks", "thank", "thankyou", "thx", "ty", "ok", "okay", "k", "done",
    "yes", "no", "yep", "yeah", "sure", "cool", "great", "fine", "good",
    "nice", "bye", "welcome", "hmm", "oh", "ah", "please",
    "and", "the", "a", "an", "is", "it", "that", "this", "for", "of", "to",
    "my", "me", "you", "we",
}


# ── Confirmed-session side questions ──────────────────────────────────────
#
# A guest whose booking is already confirmed is not shopping any more. The
# phrases below are the ones that should still be allowed through to the
# amenity / location / FAQ branches further down the node; everything else in
# a confirmed session is answered in place and the bot stops there.
#
# Deliberately generous. A false positive here costs a room card shown to a
# guest who did not want one — the bug being fixed. A false negative costs a
# canned reply instead of a real answer, which is the worse trade.
_CONFIRMED_SIDE_QUERIES = (
    # amenities and facilities
    "amenit", "facilit", "wifi", "wi-fi", "internet", "parking", "pool",
    "gym", "laundry", "restaurant", "breakfast", "lunch", "dinner", "food",
    "hot water", "ac ", "air condition", "lift", "elevator", "tv",
    # getting here
    "location", "direction", "address", "map", "how to reach", "distance",
    "landmark", "nearby", "taxi", "airport", "station",
    # arrival admin
    "check in", "check-in", "checkin", "id card", "id proof", "aadhaar",
    "aadhar", "pan card", "passport", "early check", "late check",
    # after-sales
    "cancel", "modify", "refund", "invoice", "receipt", "bill", "gst",
    "parchi", "hisab", "vej", "bhej",
    "payment", "paid", "utr", "screenshot",
    # anything that is plainly a question
    "?", "help", "kya", "kaise", "where", "what", "how",
)


def _is_confirmed_side_query(user_msg_clean: str) -> bool:
    """Whether a message in a confirmed session is a side question rather than
    an attempt to book again. See `_CONFIRMED_SIDE_QUERIES`."""
    low = (user_msg_clean or "").lower()
    return any(cue in low for cue in _CONFIRMED_SIDE_QUERIES)


#: Intents that record something the guest has done, or asked to have done, to a
#: booking. A price phrase must never overwrite one of these — see
#: `_booking_act_intent` and its use in the price-negotiation override below.
_BOOKING_ACT_INTENTS = frozenset({
    "CANCEL_BOOKING",
    "MODIFY_BOOKING",
    "PAYMENT_STATUS",
})


#: Phrases that ask for a better price. Module-level, not a local of `understand`,
#: so a test can assert what they do and do not match — see
#: `tests/test_booking_act_beats_price_phrase.py`. The last alternation is the one
#: that read a cancellation as a discount demand; the test says so in writing.
QUOTE_NEGOTIATION_PATTERNS = [
    r"\b(?:discount|discounts|offer|offers|concession|promo|coupon|price\s*match|cheaper)\b",
    r"\b(?:give\s+(?:me\s+)?discount|need\s+discount|can\s+you\s+reduce|reduce\s+price|less\s+price|price\s+kam|kam\s+price|kam\s+karo|kam\s+kijiye|rate\s+kam|kam\s+hoga|kam\s+nahi\s+hoga)\b",
    r"\b(?:aur\s+kam|thoda\s+aur|aur\s+discount|aur\s+sasta|kuch\s+aur\s+kam|aur\s+kam\s+karo|aur\s+thoda|thoda\s+kam|kamti|sasta)\b",
    r"\b(?:discount\s*milega|kuch\s*discount|koi\s*offer|offer\s*hai|kya\s*offer)\b",
    r"\b(?:in\s+|for\s+|me\s+|mein\s+|pe\s+|to\s+)?(?:rs\.?|inr|₹)\s*\d{3,5}\b",
]


def _stay_check_out(active_booking: Any) -> str | None:
    """The check-out that names the stay holding this turn's confirmed stage.

    Read from the booking row the resolver chose, never from the guest's draft
    `check_out` (which may be a date typed for a second stay). Normalised to
    `YYYY-MM-DD` because it is stored on a session field typed `str` and Mongo
    hands back a `datetime`; `_same_stay_check_out` parses the leading date either
    way, so the date is all the latch needs.
    """
    if not isinstance(active_booking, dict):
        return None
    val = active_booking.get("check_out") or active_booking.get("check_out_date")
    if not val:
        return None
    text = str(val).strip()
    return text[:10] if len(text) >= 10 else text


def _booking_act_intent(updates_intent: str | None, result_intent: str | None) -> str | None:
    """The booking act this turn is, or None if it is not one.

    The set is closed on purpose, and that is the whole design. The false-positive
    space of money phrases is not enumerable — "advance of Rs. 3000", "refund of
    Rs. 3000", a receipt's "₹3,000", "budget is 4000", "3000 mein ho jayega" — so
    no regex can be narrowed far enough to be safe. The set of acts a price phrase
    must not overwrite is small and closed, so this enumerates that instead.

    `updates_intent` is checked first: the word-lists earlier in `understand` are
    deterministic and already claim the turn for a concrete flow (the image
    analysis writes PAYMENT_STATUS, for one). `result_intent` is the model's
    verdict on the same message, which the routing below already reads beside
    `updates` for exactly this reason.
    """
    if updates_intent in _BOOKING_ACT_INTENTS:
        return updates_intent
    if result_intent in _BOOKING_ACT_INTENTS:
        return result_intent
    return None


def get_active_prompt(state: dict[str, Any]) -> str:
    """Return the active waiting prompt action from state or last prompt action."""
    action = state.get("next_action") or state.get("last_prompt_action") or ""
    return str(action).strip()


#: The card's trailing "Tell me how you want to split — I'll quote it" line,
#: spelled out whenever a guest types it rather than tapping the number.
_BUILD_YOUR_OWN_ALIASES = (
    "custom", "build your own", "build own", "custom room",
    "apna banao", "build your own room", "build my own",
)


def _build_your_own_number(state: dict[str, Any]) -> str | None:
    """The number the card printed for its build-your-own line, or None.

    One definition, because there were about to be two that disagreed. The card
    draws this line one past the priced rows (`room_service`), so the number is
    derived from the same live list here. None when no card is on screen, since
    "1" then means anything but a build-your-own pick.
    """
    options = state.get("allocation_options") or []
    if not options:
        return None
    from app.services.room_service import ROOM_CARD_MAX_COMBOS

    return str(min(len(options), ROOM_CARD_MAX_COMBOS) + 1)


def _is_build_your_own_reply(user_msg_clean: str, state: dict[str, Any]) -> bool:
    """Whether this message is the guest picking the card's build-your-own line.

    Every site that reads a digit off the rooms card has to agree on which digit
    this is, and they did not: the fast path below accepted any digit 1-5 and
    claimed this one, so the guest who tapped it was read as choosing an option
    that did not exist. Keycaps are normalised here because the card prints the
    line as "3️⃣" while a guest usually types "3".
    """
    from app.utils.room_card_reply import _normalize_keycaps

    number = _build_your_own_number(state)
    if number is None:
        return False
    # Lowercased here as well as by the caller: this reads two call sites, and
    # one of them must not be the only reason "Option 3" is recognised.
    msg = _normalize_keycaps(user_msg_clean or "").strip().lower()
    return msg in (
        number,
        f"{number}.",
        f"option {number}",
        f"option{number}",
        *_BUILD_YOUR_OWN_ALIASES,
    )


def _apply_split_party(
    updates: dict[str, Any],
    state: dict[str, Any],
    split_items: list[dict],
    split_adults: int,
) -> None:
    """Make the guest's own split the party, and keep it from being undone.

    Three things, and each one is a defect if it is left out.

    **The head count.** A guest who writes "room 1 2 people, room 2 4 people"
    has stated eight adults without writing "8" anywhere, and the model will
    often have read a different number from somewhere else in the message.
    `effective_adults` is the field the search and the validator actually read —
    both take it in preference to `adults` — so writing `adults` alone would be
    inert on any session where a previous turn had set it.

    **The room-type preference.** This branch sets `room_type_preference` to the
    guest's raw sentence. Left as the sentence, `validator` finds no room name
    in it, flags `pref_mismatch`, wipes the rooms we just selected, and searches
    again — forever. "custom" is already one of the values it treats as
    generic, which is exactly what this is.

    **The seats.** `guests` is the adult ceiling per room, already recorded on
    each line by the allocator through `line_seats`.
    """
    updates["adults"] = split_adults
    updates["effective_adults"] = split_adults
    updates["room_type_preference"] = "custom"
    updates["room_selection_resolved"] = True
    logger.info(
        "SPLIT_PARTY | sess=%s | %d lines, %d adults | stage=%s",
        state.get("session_id"),
        sum(int(item["count"]) for item in split_items),
        split_adults,
        updates.get("booking_stage"),
    )


async def understand(state: ConversationState) -> ConversationState:
    """
    Call Gemini to understand the user's message.

    Extracts:
    - Intent (GREETING, BOOK_ROOM, SELECT_ROOM, etc.)
    - Booking entities (adults, children, dates, room type, etc.)

    Then merges extracted entities into state (null = not mentioned = no change).
    """
    # ── 1. PRIVACY SCRUBBER (DPDP Act 2023) ──
    from app.utils.privacy_scrubber import contains_id_number, scrub_id_numbers
    user_msg_raw = state.get("user_message", "")
    id_volunteered = contains_id_number(user_msg_raw)
    if id_volunteered:
        # Scrub message so Aadhaar / Passport number never reaches logs or persistent store
        state["user_message"] = scrub_id_numbers(user_msg_raw)
        logger.info("PRIVACY_SCRUBBER: Inbound Aadhaar/Passport pattern scrubbed from user message.")

    # Build session object for Gemini context (shared mapper)
    session = state_to_session(state, include_messages=True)

    # ── Extract AI config from hotel config (only AI-relevant fields) ──
    ai_config = None
    config_dict = state.get("hotel_config")
    if config_dict and isinstance(config_dict, dict):
        ai_raw = config_dict.get("ai_config")
        if ai_raw:
            from app.config.hotel_config import AIPromptConfig
            ai_config = AIPromptConfig(**ai_raw)

    user_msg_clean = state.get("user_message", "").strip().lower()
    stage = state.get("booking_stage", "new")
    has_options = bool(state.get("allocation_options"))
    active_prompt_action = get_active_prompt(state)
    logger.info(f"DEBUG understand | stage={stage} | booking_id={state.get('booking_id')} | user_msg={user_msg_clean}")

    # ── OPT-OUT / OPT-IN COMPLIANCE (ZERO-LLM FAST PATH) ──
    # Patterns live at module scope (STOP_PATTERN / START_PATTERN) so a test can
    # assert on them directly instead of booting the whole graph.
    if START_PATTERN.match(user_msg_clean):
        logger.info("UNDERSTAND_FAST_PATH | START / Opt-in requested by guest")
        from app.prompts import templates
        return {
            **state,
            "intent": "GREETING",
            "opted_out": False,
            "booking_stage": "new",
            "next_action": None,
            "bot_response": templates.text_card(
                "Welcome back! You have been re-subscribed to StayChat updates. How can I help you with your booking today?"
            ),
            "extracted_data": None,
            "extraction_result": None,
            "updated_fields": [],
        }

    if state.get("opted_out") and not START_PATTERN.match(user_msg_clean):
        from app.prompts import templates
        return {
            **state,
            "intent": "CANCEL_BOOKING",
            "opted_out": True,
            "booking_stage": "opted_out",
            "next_action": "opted_out",
            "bot_response": templates.text_card(
                "You are currently unsubscribed from messages. Reply START to opt back in."
            ),
            "extracted_data": None,
            "extraction_result": None,
            "updated_fields": [],
        }

    # The old bare-tuple check (`user_msg_clean in ("stop", ...)`) is subsumed:
    # every entry it held is a whole-message match for `STOP_PATTERN`.
    if STOP_PATTERN.match(user_msg_clean):
        logger.info("UNDERSTAND_FAST_PATH | STOP / Opt-out requested by guest")
        from app.prompts import templates
        return {
            **state,
            "intent": "CANCEL_BOOKING",
            "opted_out": True,
            "booking_stage": "opted_out",
            "next_action": "opted_out",
            "selected_rooms": [],
            "allocation_options": [],
            "total_price": None,
            "payment_link": None,
            "payment_link_id": None,
            "bot_response": templates.text_card(
                "You have been unsubscribed and will not receive further messages. Reply START to opt back in."
            ),
            "extracted_data": None,
            "extraction_result": None,
            "updated_fields": [],
        }

    # ── EXPLICIT FRESH START, AHEAD OF EVERY STAGE BRANCH ──
    #
    # A guest whose stay is confirmed — usually by the CRM, which leaves no
    # booking row of ours — is answered by the confirmed branch further down, and
    # that branch returns before `apply_intent_overrides` is ever called. So the
    # reset that function implements was unreachable for exactly the guests most
    # likely to ask for it. Measured 2026-10-17: "/start fresh" logged
    # `STAGE_RESOLVED_FROM_CRM | confirmed stay, no booking row of ours`, then
    # `STAGE_2_CONFIRMED_DISPATCH`, with no FRESH_START_WIPED and no "Intent
    # override" line at all — the message was answered as a side question.
    #
    # The reservation is deliberately NOT touched: a reset ends the conversation,
    # not the booking. Only this session's state is dropped, so the guest can
    # start a new enquiry while their stay stands.
    if get_settings().START_FRESH_RESETS_CONFIRMED_STAY and is_explicit_start_fresh(user_msg_clean):
        from app.utils.booking_resume import clear_in_progress_booking_fields

        logger.info(
            "FRESH_START_FAST_PATH | explicit reset ahead of the stage branches | msg='%s'",
            user_msg_clean,
        )
        # Dropping this turn's draft is only half a reset. context_loader resolves
        # the stage from the database before `understand` runs, so on the NEXT turn
        # the stay is found again and the guest is back where they were — the reset
        # would last exactly one message. Recording which stay was dismissed is what
        # makes it stick, and there are two kinds of stay to name:
        #
        #   * one staff confirmed in the CRM, which never lands in our `bookings`
        #     collection. Its identity is the check-out date, so a later
        #     confirmation for a different stay compares unequal and still applies.
        #   * one the bot itself booked, which does have a row — and that row
        #     resolves the stage on every turn regardless of the session. Measured
        #     on prod 2026-10-17: a guest with such a row reset, was saved
        #     `Stage: new`, and was back to `stage=confirmed` on the next message.
        #     Its identity is the booking id.
        #
        # A latch that is already set is kept when this turn has nothing new to
        # dismiss. `crm_confirmed_check_out` reads None for a stay that is *already*
        # dismissed — the resolver reports no opinion about it — so overwriting the
        # latch with it would quietly un-dismiss on a second "start fresh".
        dismissed_stay = state.get("crm_confirmed_check_out") or state.get(
            "crm_confirmation_dismissed_check_out"
        )
        # The stay that set this turn's stage is not always CRM-sourced. For a guest
        # who booked through the bot, `crm_confirmed_check_out` is None by
        # construction — the resolver only asks the CRM when a booking row does not
        # exist — so the latch above stayed empty and dismissing the row un-masked
        # the CRM confirmation for the *same* stay on the next message.
        #
        # Measured on prod 2026-10-17 (the GUNU transcript): `/start fresh` logged
        # `FRESH_START_DISMISSED crm_stay_check_out=None booking_id=BK-…`, and the
        # reply after it logged `STAGE_BOOKING_ROW_DISMISSED rows=1→0` followed by
        # `STAGE_RESOLVED_FROM_CRM check_out=2026-10-19`. One stay, two sources; the
        # reset has to name the stay, not the source it was found in.
        #
        # `active_booking` is the confirmed row itself — never the draft `check_out`,
        # which may be a date the guest typed for a second stay.
        if not dismissed_stay:
            dismissed_stay = _stay_check_out(state.get("active_booking"))
        dismissed_booking = None
        if get_settings().START_FRESH_DISMISSES_BOOKING_ROW:
            dismissed_booking = state.get("active_booking_id") or state.get("booking_id")
        dismissed_booking = dismissed_booking or state.get("dismissed_booking_id")
        logger.info(
            "FRESH_START_DISMISSED | crm_stay_check_out=%s booking_id=%s | "
            "the reset now outlives this turn",
            dismissed_stay,
            dismissed_booking,
        )
        return {
            **state,
            **clear_in_progress_booking_fields(state),
            "intent": "GREETING",
            "booking_stage": "new",
            "lifecycle_stage": "lead",
            "next_action": None,
            "last_prompt_action": None,
            "extracted_data": None,
            "extraction_result": None,
            "updated_fields": [],
            # This turn must not read as confirmed either, or the reply is built
            # from the stage we just told the guest we were dropping.
            "crm_confirmed_stay": False,
            "crm_confirmed_room_type": None,
            "crm_confirmation_dismissed_check_out": dismissed_stay,
            "dismissed_booking_id": dismissed_booking,
        }

    # ── ZERO-LLM FAST PATH: USER ACKNOWLEDGMENT & FILLER MESSAGES ("Oo", "Ok", "Coming", "Wait") ──
    _ACKNOWLEDGMENT_FILLER = re.compile(
        r"^(?:"
        r"o+|ok+|okay|coming|wait|wait\s+a\s+(?:sec|minute)|wait\s+karo|hold\s+on|"
        r"acha+|achha+|cool"
        r")[.!?\s]*$",
        re.IGNORECASE,
    )
    if _ACKNOWLEDGMENT_FILLER.match(user_msg_clean) and not re.search(r"\d", user_msg_clean):
        is_hi = str(state.get("preferred_language") or "").lower() in ("hinglish", "hindi", "hi") or any(w in user_msg_clean for w in ("ji", "haan", "acha", "achha", "thik", "theek", "karo"))
        txt = (
            "Ji bilkul! Aap aaram se details share kar dijiye, main yahi hoon. 😊"
            if is_hi else
            "Sure! Take your time. Whenever you're ready, please share your details. 😊"
        )
        logger.info("FAST_PATH_ACKNOWLEDGMENT | msg='%s' -> USER_ACKNOWLEDGMENT", user_msg_clean)
        from app.prompts import templates
        return {
            **state,
            "intent": "USER_ACKNOWLEDGMENT",
            "bot_response": templates.text_card(txt),
            "extracted_data": None,
            "extraction_result": None,
        }

    # ── ZERO-LLM FAST PATH: GUEST CLAIMS EXISTING BOOKING ("Already booked hai") ──
    _ALREADY_BOOKED_PATTERN = re.compile(
        r"\b("
        r"already\s+booked|booking\s+already\s+done|already\s+have\s+a\s+booking|"
        r"already\s+confirm|pehle\s+se\s+book\s*(?:hai|tha)?|pehle\s+se\s+booking\s*(?:hai|kar\s+rakhi)?|"
        r"booking\s+ho\s+chuki\s+hai|book\s+ho\s+gaya\s+hai|booking\s+hai\s+meri|"
        r"humari\s+booking\s+hai|meri\s+booking\s+hai|kitni\s+baa?r\s+b(?:a|ta)taya"
        r")\b",
        re.IGNORECASE,
    )
    if _ALREADY_BOOKED_PATTERN.search(user_msg_clean):
        logger.info("FAST_PATH_ALREADY_BOOKED | msg='%s' -> CHECK_EXISTING_BOOKING", user_msg_clean)
        return {
            **state,
            "intent": "CHECK_EXISTING_BOOKING",
            "claimed_existing_booking": True,
            "extracted_data": None,
            "extraction_result": None,
        }

    # ── Extra Bed Preference ──
    # Three answers, not two: refuse, request, or say nothing about a bed. The
    # classifier changes nothing on "nothing", and that is the half that was
    # missing — until 2026-09-24 this block only ever wrote "NO", so a guest who
    # declined the bed once could never buy one back: `quote_generator.py:59`
    # waived the charge for the rest of the session, silently. The pattern for
    # "no" moved to `app/utils/extra_bed_intent.py` unchanged.
    from app.utils.extra_bed_intent import apply_to_state as apply_extra_bed
    apply_extra_bed(state, user_msg_clean)

    # ── Pre-extract Child Ages if present in message ──
    from app.utils.helpers import extract_child_ages_from_text
    early_child_ages = extract_child_ages_from_text(user_msg_clean)
    if early_child_ages:
        existing_ca = list(state.get("child_ages") or [])
        merged_ca = list(dict.fromkeys(existing_ca + early_child_ages))
        state["child_ages"] = merged_ca

    # ── ZERO-LLM FAST PATH FOR EXPLICIT BANK DETAILS REQUEST ──
    from app.utils.intent_overrides import _BANK_DETAILS_PATTERN
    if _BANK_DETAILS_PATTERN.search(user_msg_clean):
        logger.info("UNDERSTAND_FAST_PATH | Bank details inquiry detected -> BANK_DETAILS")
        return {
            **state,
            "intent": "BANK_DETAILS",
            "extracted_data": None,
            "extraction_result": None,
        }

    # ── ZERO-LLM FAST PATH: INTERACTIVE BUTTON TAPS ──
    #
    # A tap arrives as the button's *title*. The CRM forwards it as plain text and drops
    # `button_reply.id` (`combot_crm_v2/webhook_receiver/views.py:115-128`), so matching on
    # the title is the only thing that works on the live channel, and matching on the id is
    # the free upgrade if that ever changes. `resolve_button` tries both.
    #
    # WHICH titles mean what is NOT decided here. It is declared once in
    # `app.utils.button_registry`, which is also what `button_card` validates against — so a
    # template cannot send a button whose title this ladder would fail to recognise. That
    # pairing is the point: before the registry those two halves were maintained separately,
    # and four buttons had already drifted into doing nothing when tapped.
    # Set by the meal arm below, read ~2,700 lines down where the LLM's extraction is folded
    # into the turn. A tap on a meal button is a definite answer — it is the button we sent —
    # so the extractor must not be allowed to overwrite it. See the comment at the read site.
    _meal_plan_from_button = False

    _tapped = resolve_button(user_msg_clean)
    if _tapped is not None and _tapped.action in TAP_ROUTED_ACTIONS:
        action = _tapped.action
        logger.info(
            "UNDERSTAND_FAST_PATH | Button %s (id=%s) -> %s",
            user_msg_clean, _tapped.id, action,
        )

        if action == "SHOW_ROOMS":
            has_dates = bool(state.get("check_in") and state.get("check_out"))
            intent = "CHECK_AVAILABILITY" if has_dates else "SHOW_PRICES"
            return {
                **state,
                "intent": intent,
                "next_action": "menu_handled" if intent == "SHOW_PRICES" else state.get("next_action"),
                "extracted_data": None,
                "extraction_result": None,
            }

        if action == "BOOK_NOW":
            # Straight into the booking flow. `next_action` is left empty on
            # purpose: the validator is what decides which slot to ask for first,
            # and pre-setting one here would skip its ordering.
            return {
                **state,
                "intent": "BOOK_ROOM",
                "next_action": None,
                "extracted_data": None,
                "extraction_result": None,
            }

        if action == "CHECK_EXISTING_BOOKING":
            return {
                **state,
                "intent": "CHECK_EXISTING_BOOKING",
                "next_action": "menu_handled",
                "extracted_data": None,
                "extraction_result": None,
            }

        if action == "HUMAN_REQUEST":
            return {
                **state,
                "intent": "HUMAN_REQUEST",
                "escalation_required": True,
                "extracted_data": None,
                "extraction_result": None,
            }

        if action == "PAYMENT_PAID":
            return {
                **state,
                "intent": "PAYMENT_STATUS",
                "payment_proof_submitted": True,
                "extracted_data": None,
                "extraction_result": None,
            }

        if action == "CONFIRM_BOOKING":
            return {
                **state,
                "intent": "CONFIRM_BOOKING",
                "extracted_data": None,
                "extraction_result": None,
            }

        if action == "CANCEL_BOOKING":
            return {
                **state,
                "intent": "CANCEL_BOOKING",
                "extracted_data": None,
                "extraction_result": None,
            }

        if action == "CANCEL_ABORT":
            from app.prompts import templates
            return {
                **state,
                "intent": "USER_ACKNOWLEDGMENT",
                "booking_stage": "confirmed" if state.get("booking_id") else "new",
                "bot_response": templates.text_card("Your booking remains active! Let me know if you need anything else."),
                "extracted_data": None,
                "extraction_result": None,
            }

        if action == "NUDGE_CONTINUE":
            from app.prompts import templates
            return {
                **state,
                "intent": "USER_ACKNOWLEDGMENT",
                "bot_response": templates.text_card("Great! Please share your details or questions and we'll continue."),
                "extracted_data": None,
                "extraction_result": None,
            }

        if action == "NUDGE_CANCEL":
            from app.prompts import templates
            return {
                **state,
                "intent": "USER_ACKNOWLEDGMENT",
                "bot_response": templates.text_card("No problem! Whenever you're ready, feel free to message us again. Have a great day!"),
                "extracted_data": None,
                "extraction_result": None,
            }

        plan_code = MEAL_PLAN_BY_ACTION.get(action)
        if plan_code is not None:
            state["meal_plan_preference"] = plan_code
            _meal_plan_from_button = True
            logger.info("UNDERSTAND_FAST_PATH | Meal button %s -> %s", user_msg_clean, plan_code)

            # The tap answered the meal prompt, so there is nothing left to interpret and
            # the turn is complete: re-price with the new plan. This arm used to fall
            # through for every case, on the argument that the AC qualifier further down
            # still had to run on the same message. It cannot: `active_prompt_action` holds
            # one value, and while the meal prompt is the active one the AC block (guarded
            # on `need_ac_preference`) never fires. What the fall-through did do is run the
            # message on into the room-selection ladder, where `validator.py:423` could not
            # read "Breakfast (CP)" as a room and returned `invalid_room_selection`,
            # overwriting the answer and re-quoting the room card. Measured 2026-10-15 —
            # tapping a meal plan stored `meal_plan_preference='CP'` and then asked the
            # guest for a room.
            #
            # Only taken when the meal prompt is genuinely the active one. A meal button
            # arriving against some other prompt still falls through exactly as before.
            if active_prompt_action == "need_meal_plan_preference":
                return {
                    **state,
                    "meal_plan_preference": plan_code,
                    "intent": "BOOK_ROOM",
                    "next_action": "search",
                    "force_new_search": True,
                    "selected_rooms": [],
                    "allocation_options": [],
                    "price_breakdown": None,
                    "total_price": None,
                    "booking_stage": "collecting",
                    "updated_fields": ["meal_plan_preference"],
                }

    # ── S3: ZERO-LLM FAST PATH FOR AC PREFERENCE QUALIFIER ──
    #
    # This path RETURNs, so it is the only handler that runs on the live channel.
    # The `expected_action == "need_ac_preference"` block further down catches
    # the turns this one lets through — and lets through is the point: a miss
    # here is not "assume AC", it is "ask again", which is why the two blocks
    # have to agree on what counts as an answer.
    #
    # The two are now written in terms of the same reader
    # (`app.utils.ac_text`), which is what stops them drifting apart again.
    #
    # All three answers set `ac_preference` and `ac_qualified` and go straight to
    # `search`; the priced card is what the guest sees next.
    #
    # The ordinals and the words are read by the same matcher, so "non - ac",
    # "non_ac", "a/c" and "bina ac" arrive here exactly as "2" does. An answer
    # it cannot read does not become AC by default — control falls through to
    # the `expected_action` block below, which asks again.
    if active_prompt_action == "need_ac_preference" and not state.get("ac_qualified"):
        resolved = ac_preference_from_prompt_reply(user_msg_clean)
        if resolved == "AC":
            feat_list = list(set((state.get("feature_preferences") or []) + ["AC"]))
            return {
                **state,
                "ac_preference": "AC",
                "ac_qualified": True,
                "feature_preferences": feat_list,
                "intent": "BOOK_ROOM",
                "booking_stage": "collecting",
                "next_action": "search",
                "extracted_data": None,
                "extraction_result": None,
            }
        elif resolved == "NON_AC":
            feats = [f for f in (state.get("feature_preferences") or []) if f != "AC"] + ["Non-AC"]
            return {
                **state,
                "ac_preference": "NON_AC",
                "ac_qualified": True,
                "feature_preferences": list(set(feats)),
                "intent": "BOOK_ROOM",
                "booking_stage": "collecting",
                "next_action": "search",
                "extracted_data": None,
                "extraction_result": None,
            }
        elif resolved == "NO_PREFERENCE":
            return {
                **state,
                "ac_preference": "NO_PREFERENCE",
                "ac_qualified": True,
                "intent": "BOOK_ROOM",
                "booking_stage": "collecting",
                "next_action": "search",
                "extracted_data": None,
                "extraction_result": None,
            }

    # ── The mattress question, answered ──
    #
    # Asked by `validator` after the guest picks a card row whose rooms can take
    # a mattress. This was a disclosure until 2026-09-25 — the bed was already in
    # the row's price and the question existed to let the guest take it off. It
    # is an offer now, because the row no longer contains one: `distribute_guests`
    # seats adults to the room's `max_adults` and prices a mattress only when the
    # guest asks. So both answers re-price the same set, and only the Rs 600 line
    # moves.
    #
    #   "no"  → the same rooms, without the mattress line.
    #   "yes" → the same rooms, with it.
    #   None  → unreadable. Keep the row, no mattress, no charge: the cheap
    #           direction, and the one that cannot bill a guest for something
    #           they did not ask for. Silence is not consent to being charged.
    #
    # "The same set either way" is the fix. On "no" this used to re-search
    # without mattresses, which also removed the berth the row was using, so a
    # party of three in a Deluxe King was answered with two rooms Rs 2,250 more
    # expensive. A mattress is not a berth — see `distribute_guests`.
    if active_prompt_action == "need_extra_bed":
        from app.utils.extra_bed_intent import REFUSE, answer_bed_question

        answer = answer_bed_question(user_msg_clean)
        if answer is None:
            # Unreadable — "Final", an emoji, a question about something else.
            # This used to drop `next_action` and fall through with no bed, which
            # let the *next* node re-derive the guest's room type from nothing:
            # measured on prod 2026-10-17, a guest who had just picked PREMIER
            # ROOM KING BED said "Final" and was told "we don't have an AC king
            # bed room". Ask the question again instead — it is still the one
            # thing standing between them and a quote, and no bed is charged
            # while it goes unanswered.
            logger.info(
                "EXTRA_BED_UNREADABLE | input=%r — re-asking rather than guessing",
                user_msg_clean,
            )
            return {
                **state,
                "wants_bed": False,
                "next_action": "need_extra_bed",
                "last_prompt_action": "need_extra_bed",
                "extracted_data": None,
                "extraction_result": None,
            }
        wants = answer != REFUSE
        logger.info(
            "EXTRA_BED_ANSWERED | input=%r wants_bed=%s",
            user_msg_clean,
            wants,
        )

        if wants:
            # Accepting has to go back through the allocator, and only for the
            # reason that the charge is not a field this node can write: it is
            # what `distribute_guests(..., wants_bed=True)` puts on the option
            # the guest then picks, read off the room's own `extra_adult_rate`
            # (allocation_engine.py:656). The rooms the guest is holding were
            # allocated without a bed, so adding one here would mean
            # reimplementing that pricing rule in a second place — the drift
            # that put two copies of the calendar lookup in the codebase once
            # already. Re-searching with `wants_bed=True` returns the same set
            # with the bed priced on its rows, which is what the guest picks
            # from. This is unchanged from before: the accept path is the one
            # way the mattress has ever reached a bill.
            return {
                **state,
                "no_extra_bed": False,
                "extra_bed_preference": answer,
                "wants_bed": True,
                "selected_rooms": [],
                "allocation_options": [],
                "intent": "BOOK_ROOM",
                "booking_stage": "collecting",
                "next_action": "search",
                "extracted_data": None,
                "extraction_result": None,
            }

        # Declining needs no allocation at all — `quote_generator` zeroes the
        # mattress line straight off `no_extra_bed` (quote_generator.py:82), so
        # the guest's own pick can be priced exactly as it stands. Keeping it is
        # the fix. Clearing `selected_rooms` / `allocation_options` and setting
        # `next_action: "search"` sent the turn back through `room_searcher`,
        # which re-rendered the whole room card the guest had just chosen from:
        # they answered a yes/no and got the menu again. `extra_bed_asked` is
        # already True — `validator` set it on the turn it asked — so returning
        # to `validator` with the pick intact takes the `else` arm at
        # validator.py:448 and goes straight to `show_quote`.
        return {
            **state,
            "no_extra_bed": True,
            "extra_bed_preference": answer,
            "wants_bed": False,
            "intent": "BOOK_ROOM",
            "booking_stage": "rooms_shown",
            "next_action": "show_quote",
            "extracted_data": None,
            "extraction_result": None,
        }

    # ── S5: ZERO-LLM FAST PATH FOR COMPOSITION PREFERENCE QUALIFIER ──
    if active_prompt_action == "need_composition_preference":
        room_match = re.search(r'\b(\d+)\s*(?:rooms?|kamre?|kamra)\b', user_msg_clean)
        budget_match = re.search(r'\b(?:budget|within|under|tak)?\s*(\d{3,6})\s*(?:budget|tak|rs|inr)?\b', user_msg_clean)
        if room_match:
            return {
                **state,
                "composition_asked": True,
                "num_rooms": int(room_match.group(1)),
                "allocation_options": [],
                "selected_rooms": [],
                "intent": "BOOK_ROOM",
                "next_action": "search",
                "extracted_data": None,
                "extraction_result": None,
            }
        elif "budget" in user_msg_clean and budget_match:
            return {
                **state,
                "composition_asked": True,
                "budget": float(budget_match.group(1)),
                "allocation_options": [],
                "selected_rooms": [],
                "intent": "BOOK_ROOM",
                "next_action": "search",
                "extracted_data": None,
                "extraction_result": None,
            }
        elif user_msg_clean in ("1", "1.", "option 1", "option1", "opt 1") or any(w in user_msg_clean for w in ("kam kharcha", "best price", "cheapest", "sasta", "extra bed", "kam room", "fewer room", "low budget")):
            return {
                **state,
                "composition_asked": True,
                "occupancy_preference": "MAX",
                "allocation_options": [],
                "selected_rooms": [],
                "intent": "BOOK_ROOM",
                "next_action": "search",
                "extracted_data": None,
                "extraction_result": None,
            }
        elif user_msg_clean in ("2", "2.", "option 2", "option2", "opt 2") or any(w in user_msg_clean for w in ("aaram se", "most comfortable", "comfortable", "comfort", "zyada room", "more room", "apna bed", "own bed", "dedicated bed", "no extra bed")):
            return {
                **state,
                "composition_asked": True,
                "occupancy_preference": "BASE",
                "allocation_options": [],
                "selected_rooms": [],
                "intent": "BOOK_ROOM",
                "next_action": "search",
                "extracted_data": None,
                "extraction_result": None,
            }
        # "Split by family" is no longer on the menu (2026-09-19), so the bare
        # numbers "3" / "option 3" that used to select it are gone — we do not
        # honour a number we do not print. The free-text wording stays, because
        # the intent is real and a guest who types it is asking for something we
        # can still do; they simply have to say it rather than number it.
        elif any(w in user_msg_clean for w in ("family alag", "split", "separate", "alag", "bade alag", "subgroups")):
            from app.prompts import templates
            lang = state.get("preferred_language") or "Hinglish"
            return {
                **state,
                "composition_asked": True,
                "grouping_mode": "SEGMENTED",
                "allocation_options": [],
                "selected_rooms": [],
                "intent": "BOOK_ROOM",
                "next_action": "need_subgroups",
                "bot_response": templates.ask_family_subgroups_followup(lang=lang),
                "extracted_data": None,
                "extraction_result": None,
            }
        # The old option 4 ("You decide") is gone from the menu, and the branch
        # is gone with it: it set `occupancy_preference = None` and searched,
        # which is exactly what an unrecognised reply does by falling through.
        # Keeping it would only have re-routed "suggest" / "options" / "aap
        # decide karo" into a branch that does the same thing as no branch.

    # ── S6: ZERO-LLM FAST PATH FOR ID CHECK-IN VS NEW BOOKING ──
    if active_prompt_action == "ask_booking_or_checkin":
        import datetime
        guest_name = state.get("guest_name") or "Guest"
        now_dt = datetime.datetime.now()
        now_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
        date_display = now_dt.strftime("%d %b %Y")

        # 1. Check if user provides stay details (dates, guest count, nights)
        has_booking_criteria = bool(re.search(r'\b(?:adults?|children|child|sep|oct|nov|dec|jan|feb|mar|apr|may|jun|jul|aug|\d{1,2}[/-]\d{1,2}|night|nights|room\s*only|with\s*breakfast)\b', user_msg_clean))
        is_new_booking_phrase = any(w in user_msg_clean for w in (
            "new booking", "make a booking", "want to book", "book a room",
            "new reservation", "booking", "book", "reserve", "reservation"
        ))
        room_match = (
            re.search(r'\b(?:room|room\s*no\.?|check\s*in|checking\s*in|checkin|kamra|kamre)\s*(?:is\s*|no\.?\s*|number\s*)?([0-9]{1,4}[a-zA-Z]?)\b', user_msg_clean)
            or re.search(r'^\s*([0-9]{1,4}[a-zA-Z]?)\s*[.,!]?\s*$', user_msg_clean)
        )
        hotel_name = state.get("hotel_name") or (state.get("hotel_config") or {}).get("hotel_name") or "Hotel Sahu"
        
        if not has_booking_criteria and not is_new_booking_phrase and room_match and (room_match.group(1).isdigit() or len(room_match.group(1)) <= 4):
            room_num = room_match.group(1).strip().upper()
            # Double check it's not "1" or "2" meant as option choice unless preceded by explicit room keyword
            has_room_word = bool(re.search(r'\b(room|kamra|kamre|check\s*in|checking\s*in|checkin)\b', user_msg_clean))
            if room_num in ("1", "2") and not has_room_word:
                pass  # let standard logic evaluate
            else:
                formatted_room = f"Room {room_num}" if not room_num.lower().startswith("room") else room_num
                verified_ids = state.get("verified_ids") or []
                if len(verified_ids) > 1:
                    cards_bullets = "\n".join([
                        f"• **Guest {i+1}:** {c.get('guest_name', 'Guest')} — {c.get('doc_label', 'Government ID')}" + (f" ({c.get('id_masked_number')})" if c.get('id_masked_number') else "")
                        for i, c in enumerate(verified_ids)
                    ])
                    msg = (
                        f"✅ **Check-in Recorded!**\n\n"
                        f"Thank you, **{guest_name}**! {len(verified_ids)} IDs have been linked to **{formatted_room}** on {date_display}:\n\n"
                        f"{cards_bullets}\n\n"
                        f"Have a comfortable and pleasant stay at {hotel_name}! 🏨\n"
                        f"If you need room service, WiFi info, or anything else, feel free to ask."
                    )
                else:
                    msg = (
                        f"✅ **Check-in Recorded!**\n\n"
                        f"Thank you, **{guest_name}**! Your ID has been linked to **{formatted_room}** on {date_display}.\n\n"
                        f"Have a comfortable and pleasant stay at {hotel_name}! 🏨\n"
                        f"If you need room service, WiFi info, or anything else, feel free to ask."
                    )
                logger.info("CHECKIN_RECORDED | name=%s | room=%s | date=%s | total_ids=%d", guest_name, formatted_room, now_str, len(verified_ids))
                return {
                    **state,
                    "intent": "GREETING",
                    "room_number": formatted_room,
                    "id_received_date": now_str,
                    "booking_stage": "checked_in",
                    "next_action": None,
                    "bot_response": msg,
                    "extracted_data": None,
                    "extraction_result": None,
                }

        # 2. User says "checking in" / "check in" without room number
        if any(w in user_msg_clean for w in ("check in", "checking in", "checkin", "check-in")) and not has_booking_criteria:
            return {
                **state,
                "intent": "GREETING",
                "next_action": "ask_booking_or_checkin",
                "bot_response": f"Welcome, **{guest_name}**! Please share your **Booking ID** so I can link your ID for check-in. 🏨",
                "extracted_data": None,
                "extraction_result": None,
            }

        # 3. User says "new booking" / "booking" without dates yet
        is_new_booking_phrase = any(w in user_msg_clean for w in ("new booking", "make a booking", "want to book", "book a room", "new reservation", "booking", "book"))
        if is_new_booking_phrase and not has_booking_criteria:
            return {
                **state,
                "intent": "BOOK_ROOM",
                "booking_stage": "collecting",
                "next_action": "need_dates",
                "bot_response": "Great! Please share your **stay dates** and **number of guests** (e.g., *22-23 Sep, 2 adults*) so I can show you our best room options! 🏨",
                "extracted_data": None,
                "extraction_result": None,
            }

    # ── S7: ZERO-LLM FAST PATH FOR PAYMENT RECEIPT FOLLOWUP ──
    if active_prompt_action == "payment_followup":
        declined_followup = any(w in user_msg_clean for w in ("not now", "no", "nahi", "skip", "none", "no thanks", "later", "nhi", "bad me", "baad me", "not needed", "dont need"))
        
        # Extract meal plan if guest specified one
        meal_plan_val = None
        if "breakfast" in user_msg_clean:
            meal_plan_val = "CP"
        elif any(w in user_msg_clean for w in ("dinner", "lunch", "map")):
            meal_plan_val = "MAP"
        elif any(w in user_msg_clean for w in ("all meals", "ap", "full board")):
            meal_plan_val = "AP"
        elif any(w in user_msg_clean for w in ("room only", "ep", "no food")):
            meal_plan_val = "EP"
        elif declined_followup:
            meal_plan_val = "EP"

        booking_id = state.get("booking_id") or state.get("active_booking_id")
        from app.utils.helpers import guest_is_confirmed

        if not guest_is_confirmed(state, booking_id):
            # Ghost booking guard
            return {
                **state,
                "booking_stage": "lead",
                "lifecycle_stage": "lead",
                "next_action": None,
                "bot_response": "Thank you! If you need any assistance with a room booking or stay in Varanasi, please let us know. 🙏",
                "extracted_data": None,
                "extraction_result": None,
            }
        guest_name = state.get("guest_name") or "Guest"
        check_in = state.get("check_in") or "Your stay"
        check_out = state.get("check_out") or ""
        room_name = state.get("room_type_preference") or "Standard Room"
        selected_rooms = state.get("selected_rooms") or []
        if selected_rooms and isinstance(selected_rooms[0], dict):
            room_name = selected_rooms[0].get("room_name", room_name)

        dates_disp = f"{check_in} to {check_out}" if check_out else check_in
        # NOT `or "Hotel Sahu"`. This used to name Hotel Sahu for every tenant
        # whose session had not stamped `hotel_name` yet, so a Hotel Paradise
        # guest was congratulated on booking somewhere else. A missing name is
        # smaller than a wrong one — the copy below reads correctly without it.
        h_name = str(state.get("hotel_name") or "").strip()
        welcome_line = f"We look forward to welcoming you to {h_name}! 🏨" if h_name else "We look forward to welcoming you! 🏨"
        msg = (
            f"✅ **Booking Confirmed!**\n\n"
            f"*Booking ID:* `{booking_id}`\n"
            f"*Guest:* **{guest_name}**\n"
            f"*Dates:* {dates_disp}\n"
            f"*Room:* {room_name}\n"
            f"*Meal Plan:* {'Room Only' if meal_plan_val in ('EP', None) else meal_plan_val}\n\n"
            f"{welcome_line}\n"
            f"You can upload your ID card anytime for express check-in, or ask me for directions, amenities, or stay assistance."
        )

        return {
            **state,
            "intent": "PAYMENT_STATUS",
            "booking_stage": "confirmed",
            "next_action": None,
            "payment_missing_fields": [],
            "meal_plan_preference": meal_plan_val,
            "bot_response": msg,
            "extracted_data": None,
            "extraction_result": None,
        }

    # ── S7.2: ZERO-LLM FAST PATH FOR ISOLATED MONTH FOLLOW-UP ──
    _MONTH_NAMES = {
        "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
        "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
        "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10, "october": 10,
        "nov": 11, "november": 11, "dec": 12, "december": 12
    }
    if user_msg_clean in _MONTH_NAMES:
        target_m = _MONTH_NAMES[user_msg_clean]
        ci = state.get("check_in")
        co = state.get("check_out")
        updates_month: dict[str, Any] = {"intent": "BOOK_ROOM"}
        if ci:
            try:
                dt_ci = datetime.datetime.strptime(ci, "%Y-%m-%d")
                updates_month["check_in"] = dt_ci.replace(month=target_m).strftime("%Y-%m-%d")
            except Exception:
                pass
        if co:
            try:
                dt_co = datetime.datetime.strptime(co, "%Y-%m-%d")
                updates_month["check_out"] = dt_co.replace(month=target_m).strftime("%Y-%m-%d")
            except Exception:
                pass
        if "check_in" in updates_month:
            logger.info("UNDERSTAND_FAST_PATH | Month follow-up applied (%s) -> check_in=%s, check_out=%s", user_msg_clean, updates_month.get("check_in"), updates_month.get("check_out"))
            return {
                **state,
                **updates_month,
                "extracted_data": None,
                "extraction_result": None,
            }

    # ── S7.5: ZERO-LLM FAST PATH FOR PAYMENT RECEIPT / CLAIMS ──
    _PAYMENT_CLAIM_RE = re.compile(
        r"\b(?:payment\s*(?:done|ho\s*gaya|completed?|successful?|bhej\s*diya?|kar\s*diya?)"
        r"|paid|i\s*have\s*paid|maine\s*pay(?:ment)?\s*(?:bhi\s*)?(?:kar\s*diya|bhej\s*diya)|money\s*sent|advance\s*paid"
        r"|screenshot\s*(?:sent|attached|bheja|dekho|receipt)|check\s*payment|payment\s*receipt"
        r"|here\s*is\s*(?:the\s*)?(?:payment|receipt|screenshot)|transferred|utr\s*[:\s#]?[a-zA-Z0-9]+)\b",
        re.IGNORECASE,
    )
    if _PAYMENT_CLAIM_RE.search(user_msg_clean) and stage in ("awaiting_payment", "ready_for_payment", "quote_shown", "rooms_shown", "collecting"):
        logger.info("UNDERSTAND_FAST_PATH | Payment claim detected -> PAYMENT_STATUS")
        return {
            **state,
            "intent": "PAYMENT_STATUS",
            "booking_stage": stage,
            "next_action": "payment_followup",
            "bot_response": "Thank you! 🙏 If you've completed the payment, please share a screenshot of your payment receipt or UTR / reference number here so we can instantly verify and confirm your booking! 📲",
            "extracted_data": None,
            "extraction_result": None,
        }

    # ── S7.8: ZERO-LLM FAST PATH FOR ROOM OPTION SELECTION ──
    # When options are on screen, digits (1, 2, 3), letters (A, B, C), or "Option X"
    # are selections, NEVER unknown messages.
    #
    # But the card's LAST line is a number too — "N. Tell me how you want to
    # split — I'll quote it" — and it is not a room option. This path claimed it
    # anyway, because the pattern below accepts any digit 1-5 and this block
    # runs before the build-your-own block that owns that line. A guest who
    # tapped it sent "option 3", which matched neither of the card's two
    # options, so the card came back with "reply with the option number (1 to
    # 2)" — a list that excludes the number they had just sent — and nothing
    # they could reply would get them out. Measured on a replay turn 2026-10-10.
    if (has_options or stage in ("rooms_shown", "quote_shown")) and not (
        _is_build_your_own_reply(user_msg_clean, state)
    ):
        from app.utils.room_card_reply import _normalize_keycaps
        user_msg_norm = _normalize_keycaps(user_msg_clean).strip()
        opt_choice_match = re.match(
            r'^\s*(?:(?:i\s+)?(?:want|choose|select|book|take)|give\s+me|option|opt|choice|selection|room|no\.?)?\s*([1-5]|[a-dA-D]|1[a-b]|2[a-b]|3[a-b]|4[a-b])\s*(?:please|pls|wala|wali)?\s*[.,!]?\s*$',
            user_msg_norm,
            re.IGNORECASE,
        )
        is_ordinal_choice = user_msg_norm in (
            "first", "second", "third", "first option", "second option", "third option",
            "1st", "2nd", "3rd", "last option", "pehla", "doosra", "teesra",
            "1st option", "2nd option", "3rd option", "pehla option", "doosra option",
        )
        is_superlative = any(w in user_msg_norm for w in ("cheapest", "sasta", "sabse sasta", "lowest price", "most comfortable", "luxury", "custom", "build your own"))

        is_room_name_match = False
        if not (opt_choice_match or is_ordinal_choice or is_superlative):
            opts = state.get("allocation_options") or []
            for o in opts[:3]:
                for r in o.get("rooms", []):
                    r_nm = str(r.get("room_name") or "").strip().lower()
                    if r_nm and (user_msg_norm == r_nm or (len(user_msg_norm) >= 4 and user_msg_norm in r_nm)):
                        is_room_name_match = True
                        break
                if is_room_name_match:
                    break

        if (opt_choice_match or is_ordinal_choice or is_superlative or is_room_name_match) and not any(w in user_msg_norm for w in ("photo", "discount", "offer", "coupon", "cancel")):
            logger.info("UNDERSTAND_FAST_PATH | Room option choice detected (%r) -> SELECT_ROOM", user_msg_clean)
            pref_val = state.get("user_message", "").strip()
            if opt_choice_match and opt_choice_match.group(1):
                pref_val = f"option {opt_choice_match.group(1)}"
            return {
                **state,
                "intent": "SELECT_ROOM",
                "room_type_preference": pref_val,
                "extracted_data": None,
                "extraction_result": None,
            }

    # ── S8: CONFIRMED RESERVATION ROUTER & MULTI-BOOKING MEMORY ──
    if stage == "confirmed" or (state.get("booking_id") and stage in ("confirmed", "checked_in")):
        # 1. If user explicitly requests a new booking
        if any(w in user_msg_clean for w in ("make a new booking", "new booking", "another booking", "book another room", "book one more room", "1 make a new booking", "book another")):
            completed = list(state.get("completed_bookings") or [])
            if state.get("booking_id") and state.get("booking_id") not in [b.get("booking_id") if isinstance(b, dict) else b for b in completed]:
                completed.append({
                    "booking_id": state.get("booking_id"),
                    "guest_name": state.get("guest_name"),
                    "check_in": state.get("check_in"),
                    "check_out": state.get("check_out"),
                    "room_name": state.get("room_type_preference"),
                    "total_price": state.get("total_price"),
                })
            return {
                **state,
                "completed_bookings": completed,
                "booking_id": None,
                "check_in": None,
                "check_out": None,
                "adults": None,
                "children": None,
                "selected_rooms": [],
                "allocation_options": [],
                "price_breakdown": None,
                "total_price": None,
                "payment_details": None,
                "payment_status": None,
                "intent": "BOOK_ROOM",
                "booking_stage": "collecting",
                "next_action": "need_dates",
                "bot_response": "Great! Let's arrange your new booking. Please share your **stay dates** and **number of guests**! 🏨",
                "extracted_data": None,
                "extraction_result": None,
            }

        # 2. General conversation / greetings / payment confirmations in confirmed state
        is_greeting_or_ack = user_msg_clean in ("hi", "hello", "namaste", "hey", "thanks", "thank you", "ok", "okay", "done", "not now", "no thanks")
        is_payment_conf = bool(_PAYMENT_CLAIM_RE.search(user_msg_clean))
        raw_b_id = state.get("booking_id") or state.get("active_booking_id")
        # A stay confirmed by staff in the CRM has no booking row of ours and so no
        # `booking_id`. Without this the ghost guard below would demote a guest who has
        # already paid back to a lead and start selling to them again — which is the
        # whole bug the stage resolver's CRM branch exists to fix.
        from app.utils.helpers import guest_is_confirmed

        has_real_confirmed_booking = guest_is_confirmed(state, raw_b_id)

        if not has_real_confirmed_booking:
            logger.warning("UNDERSTAND_GHOST_CONFIRMED_GUARD | Confirmed stage with no real booking_id; demoting to lead")
            demoted_state = {
                **state,
                "booking_stage": "lead",
                "lifecycle_stage": "lead",
                "booking_id": None,
                "active_booking_id": None,
            }
            if user_msg_clean in ("no thanks", "no thank you", "nahi", "not now", "cancel", "not interested"):
                return {
                    **demoted_state,
                    "intent": "UNKNOWN",
                    "next_action": None,
                    "bot_response": "Thank you! If you plan a trip to Varanasi in the future, feel free to reach out. Have a wonderful day! 🙏",
                    "extracted_data": None,
                    "extraction_result": None,
                }
            # Fall through to general flow for inquiry/dates
            state = demoted_state
        elif is_greeting_or_ack or is_payment_conf:
            b_id = raw_b_id
            g_name = state.get("guest_name") or "Guest"
            ci = state.get("check_in") or ""
            co = state.get("check_out") or ""
            dates_disp = f" ({ci} to {co})" if ci and co else ""
            # A CRM-confirmed guest has no booking id of ours. Naming no reference
            # reads better than printing the literal "(`None`)" at them.
            id_disp = f" (`{b_id}`)" if b_id else ""
            if user_msg_clean in ("no thanks", "no thank you", "not now", "ok", "okay", "thanks", "thank you"):
                ack_reply = f"Bilkul {g_name}! Swagat ki poori taiyari hai. Kisi bhi sahayata ke liye hum yahan hain. Shubh din! 🙏"
            else:
                ack_reply = f"Hello **{g_name}**! Your reservation{id_disp}{dates_disp} is confirmed with us! 🏨\n\nHow can I help you today? You can share your ID card for express check-in, ask for directions, WiFi info, or anything else."
            return {
                **state,
                "intent": "PAYMENT_STATUS" if is_payment_conf else "GREETING",
                "booking_stage": "confirmed",
                "next_action": None,
                "bot_response": ack_reply,
                "extracted_data": None,
                "extraction_result": None,
            }
        elif (
            get_settings().CONFIRMED_BOOKING_STOPS_THE_BOT
            # Deliberately still `next_action`, which is None by the time a turn runs — so
            # this clause never blocks the suppression. It is a "is anything pending at
            # all" check, not a "which prompt am I waiting on" dispatch, and the
            # confirmed-booking suppression around it (2026-10-13) is the delicate one
            # with the ghost guards. Left as found rather than widened to
            # `active_prompt_action` on the back of a pointer fix; flagged, not changed.
            and not state.get("next_action")
            and not _is_confirmed_side_query(user_msg_clean)
        ):
            b_id = raw_b_id
            g_name = state.get("guest_name") or "Guest"
            id_disp = f" (`{b_id}`)" if b_id else ""
            return {
                **state,
                "intent": "PAYMENT_STATUS",
                "booking_stage": "confirmed",
                "next_action": None,
                "bot_response": (
                    f"Your booking{id_disp} is confirmed, **{g_name}** — you're all set! ✅\n\n"
                    "I'll leave you to it. If you need anything, just ask: "
                    "*WiFi, directions, check-in time, amenities, or your invoice*.\n\n"
                    "To book another room, say **\"new booking\"**."
                ),
                "extracted_data": None,
                "extraction_result": None,
            }

    # ── C1: ZERO-LLM FAST PATH FOR THE ROOT MENU'S TYPED WORDS ──
    # ONLY applies at root new session when no prompt or booking details are active
    #
    # The bare digits are gone as of 2026-10-15, and the words stay. Those digits numbered a
    # 1-6 text menu (`journey_templates/inquiry_templates.py:inq_greet`) that is registered
    # as `INQ_GREET` and called by nothing — the live greeting is the three-button
    # `greeting_response`, so no guest has been sent that menu for a long time. A cold "1"
    # was therefore being answered from a menu the bot had stopped sending, and it is not
    # even a settled answer: the same digit meant `SHOW_PRICES` here and `BOOK_ROOM` at the
    # second copy of this table further down, and a third copy in
    # `app/utils/intent_overrides.py` agreed with the first one.
    #
    # The words are real intents and are kept, because a guest who types "location" or
    # "photos" is asking for something the bot can do. What is dropped is only the shorthand
    # that existed to answer a menu nobody is shown; a guest typing "1" now reaches the model,
    # which already resolves "book a room" and "make a new booking" to the same intent.
    active_prompt = active_prompt_action in (
        "need_ac_preference",
        "need_extra_bed",
        "need_composition_preference",
        "need_subgroups",
        "need_child_ages",
        "need_composite_slots",
        "need_custom_rooms",
        "need_guest_name_for_payment",
        "need_change_dates",
        "need_change_guests",
        "need_meal_plan_preference",
        "change_menu_shown",
        "ask_booking_or_checkin",
        "payment_followup",
    )
    has_dates_or_guests = bool(state.get("check_in") or state.get("adults"))
    has_existing_booking = bool(state.get("booking_id") or (state.get("completed_bookings") and stage == "confirmed"))
    if stage == "new" and not has_options and not active_prompt and not has_dates_or_guests and not has_existing_booking:
        if user_msg_clean in ("hotel location & directions", "location", "directions"):
            return {
                **state,
                "intent": "SHOW_LOCATION",
                "next_action": "menu_handled",
                "extracted_data": None,
                "extraction_result": None,
            }
        elif user_msg_clean in ("room photos & amenities", "photos", "amenities"):
            return {
                **state,
                "intent": "SHOW_AMENITIES",
                "next_action": "menu_handled",
                "extracted_data": None,
                "extraction_result": None,
            }
        elif user_msg_clean in ("make a new booking", "new booking"):
            return {
                **state,
                "intent": "BOOK_ROOM",
                "next_action": "collecting",
                "extracted_data": None,
                "extraction_result": None,
            }
        elif user_msg_clean in ("check my existing booking", "my booking"):
            return {
                **state,
                "intent": "CHECK_EXISTING_BOOKING",
                "next_action": "menu_handled",
                "extracted_data": None,
                "extraction_result": None,
            }
        elif user_msg_clean in ("room prices & availability", "price and availability"):
            return {
                **state,
                "intent": "SHOW_PRICES",
                "next_action": "menu_handled",
                "extracted_data": None,
                "extraction_result": None,
            }
        elif user_msg_clean in ("something else", "other"):
            return {
                **state,
                "intent": "SOMETHING_ELSE",
                "next_action": "menu_handled",
                "extracted_data": None,
                "extraction_result": None,
            }

    # ── M3: 'change' KEYWORD NUMBERED MENU ──
    if user_msg_clean in ("change", "modify", "badlo", "update", "change booking", "modify booking", "change details", "update details") and (stage in ("rooms_shown", "quote_shown") or has_options):
        from app.prompts import templates
        return {
            **state,
            "intent": "SHOW_CHANGE_MENU",
            "next_action": "change_menu_shown",
            "updated_fields": [],
            # `lang` was missing here while every other template in this block passes it, so
            # the menu's Hinglish half was unreachable and a Hinglish guest got the English
            # menu. Passing it is the whole fix; the branch itself was always written.
            "bot_response": templates.change_options_menu(
                lang=state.get("preferred_language") or "English"
            ),
        }

    # ── M3: RESPONSES TO 'change' MENU ──
    #
    # Four rows, so the menu is a list and a tap comes back as the row's **title**
    # ("Stay Dates"), not a number. `resolve_button` is the one place that says what that
    # title means, so this reader no longer keeps its own copy of the wording — the copy
    # that used to be here accepted four numbers and a handful of typed words, and the
    # template's labels were a third list that nothing checked against either of them.
    #
    # The digits are not kept as an alias, though the rows were numbered until today. They
    # would not have worked: a bare "1" at `rooms_shown` is claimed by the SELECT_ROOM
    # handler in `apply_intent_overrides`, which runs first and reads any bare number as a
    # room option. So a guest tapping "1" on the old change menu was selecting a room, not
    # changing dates — measured 2026-10-15, and it is the reason this menu became a list
    # rather than a fourth place to keep numbers in step.
    #
    # A guest with the old numbered menu still on screen is no worse off than they were
    # yesterday, because that reply never reached this branch either.
    #
    # This guard read `state["next_action"]`, which `load_context` resets to None at the
    # start of every turn — so the branch was unreachable in production for as long as it
    # has existed (`9282250`), and every row and every typed word on the menu fell through
    # to the model. Measured 2026-10-15: tapping "Meal Plan" came back as a re-quoted room
    # card, and `next_action` on disk said "change_menu_shown" while this expression was
    # None. It reads `active_prompt_action` now — the same next_action-or-last_prompt_action
    # fallback the working prompt readers (AC, extra bed, composition) have always used.
    if active_prompt_action == "change_menu_shown":
        from app.prompts import templates
        _change_tap = resolve_button(user_msg_clean)
        _change_action = _change_tap.action if _change_tap else None
        if _change_action == "CHANGE_DATES" or user_msg_clean in ("dates", "date", "tarikh", "change dates", "check in", "check out", "checkin", "checkout"):
            return {
                **state,
                "intent": "BOOK_ROOM",
                "next_action": "need_change_dates",
                "updated_fields": [],
                "bot_response": templates.ask_change_dates(lang=state.get("preferred_language") or "English"),
            }
        elif _change_action == "CHANGE_GUESTS" or user_msg_clean in ("guests", "guest", "room count", "rooms", "guests / room count", "log", "kamre", "change guests", "change rooms"):
            return {
                **state,
                "intent": "BOOK_ROOM",
                "next_action": "need_change_guests",
                "updated_fields": [],
                "bot_response": templates.ask_change_guests(lang=state.get("preferred_language") or "English"),
            }
        elif _change_action == "CHANGE_AC" or user_msg_clean in ("ac", "non-ac", "non ac", "ac / non-ac", "ac/non-ac", "change ac"):
            return {
                **state,
                "intent": "BOOK_ROOM",
                "next_action": "need_ac_preference",
                "updated_fields": [],
                "bot_response": templates.ask_ac_preference(lang=state.get("preferred_language") or "English"),
            }
        elif _change_action == "CHANGE_MEAL" or user_msg_clean in ("meal plan", "meals", "meal", "khana", "breakfast", "change meal plan"):
            return {
                **state,
                "intent": "BOOK_ROOM",
                "next_action": "need_meal_plan_preference",
                "updated_fields": [],
                "bot_response": templates.ask_meal_plan_preference(lang=state.get("preferred_language") or "English"),
            }

    # ── M3: MEAL PLAN PREFERENCE QUALIFIER RESPONSE ──
    # Reads the surviving pointer, not `next_action` — see the note on the change-menu
    # reader above. This is where the change menu's "Meal Plan" row lands, so it was dead
    # for the same reason and by the same mechanism.
    if active_prompt_action == "need_meal_plan_preference":
        if user_msg_clean in ("1", "1.", "room only", "ep", "only room", "without breakfast", "no meals"):
            return {
                **state,
                "meal_plan_preference": "EP",
                "intent": "BOOK_ROOM",
                "next_action": "search",
                "force_new_search": True,
                "selected_rooms": [],
                "allocation_options": [],
                "price_breakdown": None,
                "total_price": None,
                "booking_stage": "collecting",
                "updated_fields": ["meal_plan_preference"],
            }
        elif user_msg_clean in ("2", "2.", "with breakfast", "breakfast", "cp", "nashta", "bed and breakfast"):
            return {
                **state,
                "meal_plan_preference": "CP",
                "intent": "BOOK_ROOM",
                "next_action": "search",
                "force_new_search": True,
                "selected_rooms": [],
                "allocation_options": [],
                "price_breakdown": None,
                "total_price": None,
                "booking_stage": "collecting",
                "updated_fields": ["meal_plan_preference"],
            }
        elif user_msg_clean in ("3", "3.", "breakfast & dinner", "breakfast and dinner", "map", "dinner", "all meals", "ap"):
            return {
                **state,
                "meal_plan_preference": "MAP",
                "intent": "BOOK_ROOM",
                "next_action": "search",
                "force_new_search": True,
                "selected_rooms": [],
                "allocation_options": [],
                "price_breakdown": None,
                "total_price": None,
                "booking_stage": "collecting",
                "updated_fields": ["meal_plan_preference"],
            }

    # ── S4 CUSTOM / BUILD YOUR OWN OPTION SELECTION ──
    avail_opts = state.get("allocation_options", [])
    is_custom_selected = _is_build_your_own_reply(user_msg_clean, state)
    if is_custom_selected:
        from app.prompts import templates
        avail_names = []
        for opt in avail_opts:
            for r in opt.get("rooms", []):
                rn = r.get("room_name", "")
                short = rn.replace("ROOM", "").replace("Room", "").strip()
                if short and short not in avail_names:
                    avail_names.append(short)
        return {
            **state,
            "intent": "BOOK_ROOM",
            "next_action": "need_custom_rooms",
            "booking_stage": "collecting_custom_room",
            "room_type_not_available": False,
            "updated_fields": [],
            "bot_response": templates.ask_split_one_liner(avail_names),
        }

    # Call Gemini (SINGLE call for both intent + entities + escalation)
    # Only ai_config (tone, language, system prompt) is passed — never business config
    result = state.get("extraction_result")
    if not result:
        result = await understand_message(
            user_message=state.get("user_message", ""),
            session=session,
            today_str=today_str(),
            ai_config=ai_config,
            persona_prompt=state.get("persona_prompt"),
            hotel_rules=state.get("hotel_rules"),
        )

    logger.info(f"Intent: {result.intent} | Extracted: {result.extracted_data.model_dump(exclude_none=True)}")

    resolved_intent, override_faq, override_extras = apply_intent_overrides(
        user_message=state.get("user_message", ""),
        intent=result.intent,
        state=state,
        reset_booking=bool(getattr(result, "reset_booking", False)),
    )
    if resolved_intent != result.intent:
        logger.info(f"Intent override: {result.intent} -> {resolved_intent}")
    if override_extras.get("abandoned_draft"):
        logger.info("Draft booking cleared (reset_booking / decline continue)")

    updates: dict[str, Any] = {
        "intent": resolved_intent,
        "faq_question": override_faq or result.faq_question,
        **override_extras,
    }

    # ── PAYMENT-TURN VERDICT (LLM, with the regex as fallback) ───────────────
    #
    # Computed only on the turns where a payment gate will actually read it —
    # `route_by_intent` consults it under `intent == "CONFIRM_BOOKING"`, and
    # `validator` consults it in the `quote_shown` stage. Any other turn skips
    # the call entirely, so this costs one extra model round-trip on the handful
    # of turns that can reach the payment step and nothing on the rest.
    #
    # Always written, including as `None`. The key is declared in
    # `ConversationState` (an undeclared key would be silently dropped at the
    # node boundary) and classified EPHEMERAL in `session_contract` — writing it
    # unconditionally means a turn that did not classify cannot inherit the
    # previous turn's verdict from graph state.
    payment_verdict: str | None = None
    _stage_now = str(state.get("booking_stage") or "")
    if get_settings().PAYMENT_TURN_LLM_VERDICT and (
        resolved_intent == "CONFIRM_BOOKING" or _stage_now == "quote_shown"
    ):
        payment_verdict = await classify_payment_turn(
            state.get("user_message", ""),
            context_hint=f"stage={_stage_now or 'unknown'}",
        )
        logger.info(
            f"PAYMENT_TURN_VERDICT | intent={resolved_intent} stage={_stage_now or '-'} "
            f"-> {payment_verdict}"
        )
    updates["payment_turn_verdict"] = payment_verdict

    # ── EXPECTED PROMPT ANSWERS OVERRIDE (V3) ──
    active_prompt_action = get_active_prompt(state)
    expected_action = None
    if active_prompt_action == "need_ac_preference":
        expected_action = "need_ac_preference"
    elif active_prompt_action == "need_composition_preference":
        expected_action = "need_composition_preference"
    elif active_prompt_action == "need_subgroups":
        expected_action = "need_subgroups"
    elif active_prompt_action == "need_child_ages" or ((state.get("children") or 0) > 0 and len(state.get("child_ages") or []) < int(state.get("children") or 0)):
        expected_action = "need_child_ages"
    elif state.get("booking_stage") == "collecting_custom_room" or active_prompt_action == "need_custom_rooms":
        expected_action = "need_custom_rooms"
    elif state.get("booking_stage") == "quote_shown" and not state.get("guest_name"):
        expected_action = "need_guest_name_for_payment"
    elif active_prompt_action == "need_guest_name_for_payment":
        expected_action = "need_guest_name_for_payment"
    elif active_prompt_action in ("group_booking_lead", "lead_created") or state.get("booking_stage") == "lead_created":
        expected_action = "group_booking_lead"
    elif active_prompt_action == "need_meal_plan_preference":
        expected_action = "need_meal_plan_preference"
    elif not state.get("check_in") or not state.get("check_out") or state.get("adults") is None:
        expected_action = "need_composite_slots"

    user_msg_clean = state.get("user_message", "").strip().lower()

    GREETING_WORDS = {
        "hi", "hello", "hey", "namaste", "hola", "good morning", 
        "good evening", "good afternoon", "good night", "hii", "helo", "hiii", 
        "start", "restart", "menu", "help", "sup", "yo",
        "good evening sir", "good morning sir", "good afternoon sir",
        "good evening madam", "good morning madam", "good afternoon madam",
        "good evening sir/madam", "good morning sir/madam", "good afternoon sir/madam",
        "good evening sir / madam", "good morning sir / madam", "good afternoon sir / madam",
        "hello sir", "hi sir", "namaste sir", "hello madam", "hi madam", "dear sir", "dear team",
        "sir", "madam", "sir/madam", "sir / madam"
    }
    NON_NAME_WORDS = GREETING_WORDS.union({
        "confirm", "yes", "no", "ok", "book", "book room", "1", "2", "3", "4", "5", "6",
        "ac", "non ac", "non-ac", "english", "hindi", "hinglish",
        "vacancy", "job", "jobs", "hiring", "resume", "cv", "apply", "interview",
        "manager", "front office", "receptionist", "staff", "salary", "document",
        "photo", "photos", "photoes", "foto", "pic", "pics", "pocs", "picture", "pictures",
        "image", "images", "img", "gallery", "dikhao", "bhejo", "dekhna"
    })

    _PHOTO_KW_LIST = ("photo", "photos", "photoes", "foto", "pic", "pics", "pocs", "picture", "pictures", "image", "images", "img", "gallery", "tasveer")
    is_photo_query = (
        updates.get("intent") == "SHOW_AMENITIES"
        or any(pw in user_msg_clean for pw in _PHOTO_KW_LIST)
        or any(w in user_msg_clean for w in ("see rooms", "view rooms"))
    )

    is_inquiry_or_question = (
        "?" in state.get("user_message", "")
        or is_photo_query
        or any(user_msg_clean.startswith(q) for q in (
            "is there", "are there", "do you have", "can i", "can you", "what is", "how much", 
            "where is", "when is", "tell me", "i want to apply", "any vacancy", "vacancy for",
            "show me", "send me", "send", "show"
        ))
        or any(k in user_msg_clean for k in ("vacancy", "hiring", "resume", "curriculum vitae", "job application", "front office manager"))
    )

    if user_msg_clean in GREETING_WORDS:
        updates["intent"] = "GREETING"
        updates.pop("guest_name", None)

    # ── V3 Language Switch at any stage (Non-blocking) ──
    if user_msg_clean in ("hindi", "hinglish", "hindi me", "hindi mein", "hindi please"):
        updates["preferred_language"] = "Hinglish"
        updates["conversation_language"] = "Hindi"
    elif user_msg_clean in ("english", "english please", "in english"):
        updates["preferred_language"] = "English"
        updates["conversation_language"] = "English"

    # ── V3 Root Menu Words — zero model spend ──
    # ONLY applies at root new session when no prompt or booking details are active
    #
    # The bare digits are gone, for the reason the table at C1 gives: they answered a 1-6
    # menu nothing sends. The words are kept.
    #
    # The "3" arm is the one to look at before changing anything here. It read
    # `BOOK_ROOM` with a comment asserting "MENU_03 routes to the same S2_SLOTS flow as
    # MENU_01", while the C1 table above and the third copy in `intent_overrides.py` both
    # read the same word as `SHOW_PRICES`. Two of the three, and the printed label — "Room
    # prices & availability" — say prices. It is NOT changed here: which of the two is
    # right is a behaviour call, not a refactor, and this commit only takes the digits away.
    stage = state.get("booking_stage", "new")
    has_options = bool(state.get("allocation_options"))
    has_dates_or_guests = bool(state.get("check_in") or state.get("adults"))
    if stage == "new" and not has_options and not expected_action and not has_dates_or_guests:
        if user_msg_clean in ("make a new booking", "new booking"):
            updates["intent"] = "BOOK_ROOM"
        elif user_msg_clean in ("check my existing booking", "my booking"):
            updates["intent"] = "CHECK_EXISTING_BOOKING"
        elif user_msg_clean in ("room prices & availability", "price and availability"):
            updates["intent"] = "BOOK_ROOM"
        elif user_msg_clean in ("hotel location & directions", "location", "directions"):
            updates["intent"] = "SHOW_LOCATION"
        elif user_msg_clean in ("room photos & amenities", "photos", "amenities"):
            updates["intent"] = "SHOW_AMENITIES"
        elif user_msg_clean in ("something else", "other"):
            updates["intent"] = "SOMETHING_ELSE"

    # ── S7: Lead guest name capture at quote confirmation ──
    is_side_intent = updates.get("intent") in (
        "SHOW_AMENITIES", "AMENITY_QUERY", "FAQ", "PRICE_MATCH", "SHOW_PRICES",
        "SHOW_LOCATION", "CHECK_EXISTING_BOOKING", "CANCEL_BOOKING", "MODIFY_BOOKING"
    )
    if expected_action == "need_guest_name_for_payment" and not is_inquiry_or_question and not is_side_intent and user_msg_clean not in GREETING_WORDS:
        if user_msg_clean in ("confirm", "confirm booking", "confirm kar do", "kar do", "yes", "haan", "ha", "y", "correct", "theek hai", "ji", "proceed", "pay", "ok", "okay", "done"):
            existing_name = state.get("guest_name") or state.get("prior_guest_name") or state.get("customer_name")
            if existing_name:
                updates["guest_name"] = existing_name
            updates["intent"] = "CONFIRM_BOOKING"
        elif (
            get_settings().NAME_CAPTURE_READS_ONLY_THE_NAME
            and not updates.get("guest_name")
            and user_msg_clean not in NON_NAME_WORDS
            and not any(user_msg_clean.startswith(w) for w in ("book ", "i want", "need room", "room from", "can i book"))
        ):
            # ── Read the name, not the message ───────────────────────────
            # Pranav, 2026-09-30: the name prompt "takes the whole thing no
            # matter what the customer sent". `extract_person_name` scans for a
            # name and returns `None` when there is not one, where the branch
            # below accepted any short digit-free message — which is how
            # "hi i am rahul" was saved as "Hi I Am Rahul".
            #
            # `None` is left as `None`: no name is captured and the prompt
            # stands, so the guest is asked again. That is the safe direction —
            # a guest called "Looking" is not recoverable by asking again.
            _name = extract_person_name(state.get("user_message", "") or "")
            if _name:
                updates["guest_name"] = _name
                updates["intent"] = "CONFIRM_BOOKING"
        elif not updates.get("guest_name") and user_msg_clean not in NON_NAME_WORDS and not any(user_msg_clean.startswith(w) for w in ("book ", "i want", "need room", "room from", "can i book")):
            clean_name = state.get("user_message", "").strip()
            for prefix in ("my name is", "i am", "i'm", "naam hai", "name is", "name:"):
                if clean_name.lower().startswith(prefix):
                    clean_name = clean_name[len(prefix):].strip(" :,-")
            
            # Strict human name validator (Sir's Spec / D11)
            # Rejects sentences, payment phrases, numbers, and inquiry words
            cand_lower = clean_name.lower()
            payment_or_sentence_words = (
                "payment", "paid", "diya", "kar diya", "bhej diya", "transfer", "screenshot", 
                "gpay", "phonepe", "paytm", "upi", "qr", "receipt", "advance", "rupaye", "rs", "inr",
                "hai", "hain", "karna", "karein", "chahiye", "chahta", "room", "hotel", "stay", 
                "booking", "please", "help", "batao", "batayein", "cancel", "change", "what", "where",
                "vacancy", "job", "jobs", "hiring", "resume", "cv", "?"
            )
            words = cand_lower.split()
            is_valid_name = (
                bool(clean_name)
                and 1 <= len(words) <= 4
                and not any(ch.isdigit() for ch in clean_name)
                and not any(w in words for w in payment_or_sentence_words)
                and not any(pw in cand_lower for pw in ("kar diya", "payment", "bhej diya", "hotel sahu"))
                and bool(re.match(r"^[A-Za-z.\s\'-]{2,50}$", clean_name))
            )
            if is_valid_name and cand_lower not in NON_NAME_WORDS:
                updates["guest_name"] = clean_name.title()
                updates["intent"] = "CONFIRM_BOOKING"

    elif expected_action == "need_child_ages":
        # Check if user replied with child ages (e.g. "8", "13", "4 and 10", "child is 6", "1.5 saal", "4 years", "1 child 4 years")
        from app.utils.helpers import extract_child_ages_from_text
        found_ages = extract_child_ages_from_text(user_msg_clean)
        if not found_ages:
            nums = [int(float(n)) for n in re.findall(r'\b\d+(?:\.\d+)?\b', user_msg_clean)]
            # If user wrote "1 child 4 years", nums is [1, 4]. If state already knows children=1, the age is 4!
            total_kids = int(state.get("children") or 1)
            if len(nums) > total_kids and nums[0] == total_kids:
                nums = nums[1:]
            found_ages = [n for n in nums if 0 <= n <= 17]

        total_kids = int(state.get("children") or len(found_ages) or 1)
        if found_ages:
            existing_ages = list(state.get("child_ages") or [])
            if len(found_ages) >= total_kids:
                updates["child_ages"] = found_ages[:total_kids]
            elif existing_ages and (len(existing_ages) + len(found_ages) <= total_kids):
                updates["child_ages"] = existing_ages + found_ages
            else:
                updates["child_ages"] = found_ages
            updates["intent"] = "BOOK_ROOM"
        elif user_msg_clean not in GREETING_WORDS and not any(w in user_msg_clean for w in ("job", "vacancy", "hiring", "resume")):
            # Guest replied with something else or ignored age -> assume 4 (free) to avoid endless loop
            updates["child_ages"] = [4] * total_kids
            updates["intent"] = "BOOK_ROOM"

    elif expected_action == "need_ac_preference":
        discount_kws = ("discount", "kam karo", "sasta", "offer", "coupon", "bargain", "best rate", "reduce", "price match", "less price", "kam price", "price kam")
        resolved_ac = ac_preference_from_prompt_reply(user_msg_clean)
        if any(w in user_msg_clean for w in discount_kws):
            updates["intent"] = "PRICE_MATCH"
            updates["next_action"] = None
            updates["ac_preference"] = "AC"
            updates["ac_qualified"] = True
            feat_list = list(set((updates.get("feature_preferences") or state.get("feature_preferences") or []) + ["AC"]))
            updates["feature_preferences"] = feat_list
        elif resolved_ac == "AC":
            updates["ac_preference"] = "AC"
            updates["ac_qualified"] = True
            feat_list = list(set((updates.get("feature_preferences") or state.get("feature_preferences") or []) + ["AC"]))
            updates["feature_preferences"] = feat_list
            updates["intent"] = "BOOK_ROOM"
            updates["next_action"] = "search"
        elif resolved_ac == "NON_AC":
            updates["ac_preference"] = "NON_AC"
            updates["ac_qualified"] = True
            feats = [f for f in (updates.get("feature_preferences") or state.get("feature_preferences") or []) if f != "AC"] + ["Non-AC"]
            updates["feature_preferences"] = list(set(feats))
            updates["intent"] = "BOOK_ROOM"
            updates["next_action"] = "search"
        elif resolved_ac == "NO_PREFERENCE" or user_msg_clean.strip() == "no":
            updates["ac_preference"] = "NO_PREFERENCE"
            updates["ac_qualified"] = True
            updates["intent"] = "BOOK_ROOM"
            updates["next_action"] = "search"
        elif not updates.get("ac_preference") and is_plausible_ac_answer(user_msg_clean):
            # The guest answered and we could not read it. Only `next_action` is
            # pinned — deliberately not `intent`, so a message that was really
            # an escalation is still allowed to escalate, and deliberately not
            # `ac_qualified`, which means "the guest has answered" and belongs
            # to the branches above. Leaving both alone is what lets
            # `room_searcher` put the question again instead of quietly
            # assuming AC and selling the class the guest ruled out.
            logger.info(
                "AC_REASK | input=%r resolved to no AC preference — asking again",
                user_msg_clean,
            )
            updates["next_action"] = "need_ac_preference"

    elif expected_action == "need_composition_preference":

        updates["composition_asked"] = True

        # Check free-text shortcuts first:
        # 1. "3 rooms", "3 room", "3 rooms chahiye"
        room_match = re.search(r'\b(\d+)\s*(?:rooms?|kamre?|kamra)\b', user_msg_clean)
        # 2. "budget 6000", "budget 6000 tak", "6000 budget"
        budget_match = re.search(r'\b(?:budget|within|under|tak)?\s*(\d{3,6})\s*(?:budget|tak|rs|inr)?\b', user_msg_clean)

        if room_match:
            updates["num_rooms"] = int(room_match.group(1))
            updates["intent"] = "BOOK_ROOM"
            updates["next_action"] = "search"
        elif "budget" in user_msg_clean and budget_match:
            try:
                updates["budget"] = float(budget_match.group(1))
                updates["intent"] = "BOOK_ROOM"
                updates["next_action"] = "search"
            except ValueError:
                updates["intent"] = "BOOK_ROOM"
                updates["next_action"] = "search"
        elif user_msg_clean in ("1", "1.", "option 1", "option1", "opt 1") or any(w in user_msg_clean for w in ("kam kharcha", "best price", "cheapest", "sasta", "extra bed", "kam room", "fewer room", "low budget")):
            # Option 1: Kam kharcha / Best price / cheapest / extra bed
            updates["occupancy_preference"] = "MAX"
            updates["intent"] = "BOOK_ROOM"
            updates["next_action"] = "search"
        elif user_msg_clean in ("2", "2.", "option 2", "option2", "opt 2") or any(w in user_msg_clean for w in ("aaram se", "most comfortable", "comfortable", "comfort", "zyada room", "more room", "apna bed", "own bed", "dedicated bed")):
            # Option 2: Aaram se / Most comfortable / own bed / more rooms / dedicated bed
            updates["occupancy_preference"] = "BASE"
            updates["intent"] = "BOOK_ROOM"
            updates["next_action"] = "search"
        elif any(w in user_msg_clean for w in ("family alag", "split", "separate", "alag", "bade alag", "subgroups")):
            # "Split by family" is off the menu as of 2026-09-19, so the bare
            # numbers are gone — but the intent is real and still routed, for a
            # guest who says it in words. Mirrors the zero-LLM fast path above;
            # the two must agree or the same reply would mean two things
            # depending on whether the fast path caught it.
            # (`composition_asked` is already set at the top of this handler.)
            updates["grouping_mode"] = "SEGMENTED"
            updates["intent"] = "BOOK_ROOM"
            updates["next_action"] = "need_subgroups"
            from app.prompts import templates
            lang = state.get("preferred_language") or "Hinglish"
            updates["bot_response"] = templates.ask_family_subgroups_followup(lang=lang)
        else:
            # Anything else — including the "3" and "4" numbers the old
            # four-option menu used to offer, and "you decide" — falls through
            # to the standard ranking. That is the same outcome the old option 4
            # produced, and it is why cutting the menu to two choices needed no
            # new branch here: the guest can no longer ask for something we do
            # not offer, and if they type it anyway they get the default rather
            # than a dead end.
            #
            # "Never ask twice" still holds: whatever they said, we search now.
            updates["intent"] = "BOOK_ROOM"
            updates["next_action"] = "search"

    elif expected_action == "need_subgroups":
        updates["intent"] = "BOOK_ROOM"
        updates["next_action"] = "search"
        updates["grouping_mode"] = "SEGMENTED"
        updates["subgroups"] = [{"description": state.get("user_message", "").strip()}]

    elif expected_action == "need_custom_rooms":

        updates["room_type_preference"] = user_msg_clean
        updates["intent"] = "BOOK_ROOM"
        updates["booking_stage"] = "collecting"
        updates["next_action"] = "search"

        from app.services.room_service import room_names_from_options
        from app.utils.room_requirement_parse import (
            parse_room_requirements,
            validate_room_requirements,
        )

        raw_message = state.get("user_message", "")

        # The model reads the request; the deterministic scan is the net under
        # it. This branch is the one place in the funnel where the guest writes
        # without a template — the card asks "how many rooms, which type" and
        # takes the answer in their own words, in their own language, usually
        # mixed into the rest of the booking ("Cheack in 24/10/26 at 5pm ...
        # TOTAL HEAD 10+ 2 Child ... Room category- 3 kashi classic Room in 4th
        # floor"). Reading that is what the model is for. The scan reads the
        # same message by rule and has to be taught, guard by guard, not to take
        # 26 rooms out of a date; told which numbers are the dates, the model
        # does not have the problem.
        #
        # So `parse_room_requirements` runs only if the call itself fails, and
        # that is deliberate: an outage degrades to the previous behaviour
        # rather than to "no room request". A clean empty read is not an outage
        # — it means the message was read and holds no rooms — and running the
        # scan over that would hand back exactly the misparse the guards above
        # exist to stop.
        extracted = await extract_room_request(
            user_message=raw_message,
            catalogue=room_names_from_options(state.get("allocation_options")),
            check_in=state.get("check_in"),
            check_out=state.get("check_out"),
        )
        if extracted is None:
            logger.warning(
                "CUSTOM_ROOM_READ_FALLBACK | sess=%s | the normaliser did not "
                "answer; reading with the deterministic scan",
                state.get("session_id"),
            )
            extracted = parse_room_requirements(raw_message)

        # A per-room occupancy split — "2 people 1 room" — arrives from the
        # normaliser as lines carrying `guests` and naming no room type. It
        # states the party as well as the rooms: the adults are the sum of the
        # per-room counts, so the guest does not have to have said "9" anywhere
        # for the request to be answerable.
        is_split = is_an_explicit_split(extracted)
        split_adults = sum(
            int(item.get("count") or 0) * int(item.get("guests") or 0)
            for item in (extracted or [])
        ) if is_split else 0
        party_size = split_adults or (
            int(state.get("adults") or 0) + int(state.get("children") or 0)
        )

        # Whatever read it, one bound applies: a request for more rooms than
        # there are guests is a misparse, not a booking, and is refused rather
        # than priced.
        reqs, problem = validate_room_requirements(extracted, party_size=party_size)
        if reqs:
            updates["room_requirements"] = reqs
            updates["booking_stage"] = "custom_rooms_selected"
            updates["custom_room_problem"] = None
            if is_split:
                _apply_split_party(updates, state, extracted, split_adults)
        else:
            # Ask again, and say which part did not land. This branch used to
            # leave `booking_stage` as "collecting" and search anyway, so an
            # answer it could not read was silently replaced by whatever the
            # allocator would have offered regardless — the guest's request
            # disappeared with nothing said about it.
            updates["booking_stage"] = "collecting_custom_room"
            updates["next_action"] = "need_custom_rooms"
            updates["custom_room_problem"] = problem or (
                "I couldn't find a room count in that — could you write it the "
                "way the example does?"
            )

    elif expected_action == "group_booking_lead":

        raw_msg = state.get("user_message", "").strip()
        msg_clean = raw_msg.lower()

        # 1. Check if user is clarifying that this is NOT a group booking (Correction)
        not_group_keywords = (
            "group requirement nahi", "group requirement nehi", "group nahi", "group nehi", "not a group",
            "not group", "no group", "individual", "alag alag", "alag-alag", "separate room", "personal", "individual room"
        )
        is_clarifying_not_group = any(k in msg_clean for k in not_group_keywords)

        # Check if user mentioned room count (e.g. "Three alag alag room", "3 rooms", "2 rooms")
        word_to_num = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "ek": 1, "do": 2, "teen": 3, "char": 4, "paanch": 5, "chhe": 6
        }
        room_match = re.search(r'\b(one|two|three|four|five|six|ek|do|teen|char|paanch|chhe|\d+)\s*(?:alag\s*alag\s*)?(?:rooms?|kamre?|kamra)\b', msg_clean)

        if is_clarifying_not_group or (room_match and int(word_to_num.get(room_match.group(1), int(room_match.group(1)) if room_match.group(1).isdigit() else 1)) < 5):
            new_rooms = None
            if room_match:
                val = room_match.group(1)
                new_rooms = word_to_num.get(val, int(val) if val.isdigit() else None)

            if new_rooms:
                updates["num_rooms"] = new_rooms
            else:
                new_rooms = int(state.get("num_rooms") or 1)

            cur_adults = int(state.get("adults") or 2)
            if cur_adults >= 10:
                updates["adults"] = min(cur_adults, max(2, new_rooms * 2))
                updates["effective_adults"] = updates["adults"]

            updates["group_lead_submitted"] = False
            updates["next_action"] = "search"
            updates["booking_stage"] = "new"
            updates["intent"] = "BOOK_ROOM"

        else:
            # 2. Check if user sent contact details (Phone / Name)
            phone_match = re.search(r'(?:\+?91[\s-]?)?([6-9]\d{9})\b', raw_msg)
            detected_phone = phone_match.group(1) if phone_match else None

            # Extract name from lines or message
            lines = [l.strip() for l in raw_msg.split("\n") if l.strip()]
            detected_name = None
            for line in lines:
                clean_l = re.sub(r'[^\w\s]', '', line).strip()
                if clean_l and not re.search(r'\d', clean_l):
                    l_lower = clean_l.lower()
                    if not any(w in l_lower for w in ("contact", "detail", "details", "mera", "meri", "number", "phone", "hai", "yehi", "ye", "this", "my")):
                        if len(clean_l.split()) <= 4:
                            detected_name = clean_l.title()
                            break

            if detected_phone or detected_name or any(w in msg_clean for w in ("contact", "details", "phone", "number", "mera", "meri")):
                existing_name = state.get("guest_name") or state.get("customer_name")
                existing_phone = state.get("customer_identifier") or ""
                
                final_name = detected_name or (existing_name if existing_name != "Guest" else None) or ""
                final_phone = detected_phone or (existing_phone if (len(existing_phone) == 10 or existing_phone.startswith("+91")) else "")

                if final_name:
                    updates["guest_name"] = final_name
                if detected_phone:
                    updates["customer_identifier"] = detected_phone
                    final_phone = detected_phone

                try:
                    from app.repositories.lead_repository import create_lead, generate_lead_id
                    hotel_id = state.get("hotel_id", 0)
                    lead_id = await generate_lead_id(hotel_id)
                    lead_data = {
                        "lead_id": lead_id,
                        "hotel_id": hotel_id,
                        "session_id": state.get("session_id", ""),
                        "customer_phone": final_phone or state.get("customer_identifier") or "",
                        "guest_name": final_name or "Guest",
                        "check_in": state.get("check_in"),
                        "check_out": state.get("check_out"),
                        "adults": state.get("adults"),
                        "children": state.get("children") or 0,
                        "num_rooms": state.get("num_rooms") or 5,
                        "lead_type": "GROUP_BOOKING",
                        "notes": raw_msg,
                    }
                    await create_lead(lead_data)
                except Exception as e:
                    logger.warning(f"Failed to save group lead: {e}")

                name_greeting = f" {final_name}" if final_name and final_name != "Guest" else ""
                phone_mention = f" on **+91 {final_phone}**" if final_phone else ""

                updates["group_lead_submitted"] = True
                updates["booking_stage"] = "lead_created"
                updates["next_action"] = "lead_created"
                from app.prompts import templates
                
                if not detected_phone and not detected_name and any(w in msg_clean for w in ("contact", "details", "mera", "meri")):
                    updates["bot_response"] = templates.text_card(
                        "Ji bilkul, kripya apna **naam** aur **contact number** share kar dijiye."
                    )
                else:
                    updates["bot_response"] = templates.text_card(
                        f"Thank you{name_greeting}! 🙏\n\n"
                        f"We have received your group requirements and contact details. "
                        f"Our reservations team will review special group pricing and connect with you directly{phone_mention} shortly.\n\n"
                        f"Is there anything else we can assist you with in the meantime?"
                    )

    # ── Extract escalation classification from Gemini ──────────
    esc = result.escalation.model_dump() if result.escalation else {}
    updates["escalation_required"] = esc.get("escalation_required", False)
    updates["escalation_issue_type"] = esc.get("issue_type", "NORMAL")
    updates["escalation_reason_code"] = esc.get("reason_code", "OTHER")
    updates["escalation_urgency"] = esc.get("urgency", "LOW")
    updates["escalation_sentiment"] = esc.get("sentiment", "NEUTRAL")
    updates["escalation_reason"] = esc.get("reason", "normal_query")
    updates["escalation_confidence"] = esc.get("confidence", 0.9)

    if esc.get("escalation_required"):
        logger.info(
            f"Escalation flagged: type={esc.get('issue_type')}, "
            f"reason_code={esc.get('reason_code')}, urgency={esc.get('urgency')}"
        )

    # ── Extract task classification from Gemini ────────────────
    task = result.task_classification or {}
    updates["task_is_request"] = task.get("is_task_request", False)
    updates["task_type"] = task.get("task_type", "OTHER")
    updates["task_subtype"] = task.get("subtype", "OTHER")
    updates["task_urgency"] = task.get("urgency", "NORMAL")
    updates["task_confidence"] = task.get("confidence", 0.0)
    updates["task_summary"] = task.get("summary", "")

    if task.get("is_task_request"):
        logger.info(
            f"Task request detected: type={task.get('task_type')}, "
            f"subtype={task.get('subtype')}, urgency={task.get('urgency')}"
        )

    # ── Extract room service extraction from Gemini ────────────
    updates["room_service_extraction"] = getattr(result, "room_service_extraction", None)

    # ── Extract amenity query from Gemini ─────────────────────
    if not updates.get("amenity_query"):
        updates["amenity_query"] = getattr(result, "amenity_query", None)

    # ── Extract explicit preferences unconditionally ─────────────
    ext = getattr(result, "extracted_data", None)
    if ext:
        if getattr(ext, "ac_preference", None) is not None:
            updates["ac_preference"] = ext.ac_preference
            feat_list = list(updates.get("feature_preferences") or state.get("feature_preferences") or [])
            if ext.ac_preference == "AC" and "AC" not in feat_list:
                feat_list.append("AC")
            elif ext.ac_preference == "NON_AC" and "Non-AC" not in feat_list:
                feat_list.append("Non-AC")
            updates["feature_preferences"] = feat_list
        if getattr(ext, "occupancy_preference", None) is not None:
            updates["occupancy_preference"] = ext.occupancy_preference
        if getattr(ext, "photo_tag_preference", None) is not None:
            updates["photo_tag_preference"] = ext.photo_tag_preference

    # ── Extract language tracking from Gemini ──────────────────
    # ── Continuous Language Detection & Adaptation ──────────────────

    user_msg_text = state.get("user_message", "").strip()
    user_msg_lower = user_msg_text.lower()
    
    # Deterministic lexical cues. The sets are module-level constants now — the
    # node used to rebuild both on every single message.
    _full_weights = get_settings().LANGUAGE_DETECTION_WEIGHTS_HINDI
    hinglish_cues = _HINGLISH_CUES_EXPANDED if _full_weights else _HINGLISH_CUES_LEGACY
    english_cues = _ENGLISH_CUES

    words = set(re.findall(r'\b\w+\b', user_msg_lower))
    h_count = len(words.intersection(hinglish_cues))
    e_count = len(words.intersection(english_cues))

    detected_lang = getattr(result, "detected_language", None)
    base_lang = getattr(result, "base_language", None)
    requested_lang = getattr(result, "requested_language", None)
    # The reply vocabulary of the room card is 1-4 and A-D, so "d" belongs in
    # this set with "a"/"b"/"c". Without it a bare "d" — the card's
    # "tell me how you want to split" — failed the subtraction below, was not
    # recognised as a short choice, and was read as a language signal instead.
    _SHORT_CHOICES = {
        "option", "opt", "choice", "selection", "room", "rooms",
        "1", "2", "3", "4", "5", "a", "b", "c", "d",
        "1a", "1b", "2a", "2b", "3a", "3b", "4a", "4b",
    }
    is_numeric_or_short_choice = bool(re.match(r'^(?:option|opt|choice|selection)?\s*[0-9a-zA-Z\s.,]*$', user_msg_lower.strip())) and len(words - _SHORT_CHOICES) == 0

    # ── Which way the cues point ─────────────────────────────────────
    #
    # Pranav, 2026-09-30: *"detection of Hindi, Hinglish (Roman Hindi) is not
    # consistent, it still replies in english most of the time."*
    #
    # The old rule was `h_count >= e_count` for Hinglish and `e_count >
    # h_count` for English — a straight head-count. It misfired because the two
    # sides are not comparable: Hinglish borrows English *content* words
    # ("room", "book", "price", "family") while English borrows no Hindi, so a
    # single borrowed noun could outvote real Hindi. "2 adults aur 1 child" was
    # read as English; "room ki photo bhejo" scored zero and fell through to
    # the English default.
    #
    # The proportional rule encodes the asymmetry: a Hindi cue is strong
    # evidence, an English cue is weak evidence, so Hindi wins ties and
    # near-ties and English must clearly dominate to take the turn.
    #
    # The acknowledgement guard is separate and deliberate. "ok", "thank you"
    # and "done" carry one English cue each and no language intent at all —
    # under either rule they flipped an established Hinglish session to
    # English, which is the "keeps replying in English" complaint in one line.
    # A message made only of those words now changes nothing.
    _has_content = bool(words - _LANGUAGE_NEUTRAL_WORDS - _SHORT_CHOICES)
    _neutral_ack = _full_weights and not _has_content

    _has_devanagari = bool(re.search(r'[\u0900-\u097F]', user_msg_text))

    if _full_weights:
        _hindi_wins = _has_devanagari or (h_count > 0 and e_count <= h_count * 2)
        _english_wins = not _has_devanagari and (e_count > 0 and e_count > h_count * 2 and _has_content)
    else:
        _hindi_wins = _has_devanagari or (h_count > 0 and h_count >= e_count)
        _english_wins = not _has_devanagari and (e_count > 0 and e_count > h_count)

    if requested_lang or getattr(result, "language_switch_requested", False):
        target_style = requested_lang or detected_lang or "English"
        updates["preferred_language"] = "English" if "english" in target_style.lower() else "Hinglish"
        updates["conversation_language"] = "English" if "english" in target_style.lower() else "Hindi"
    elif is_numeric_or_short_choice and state.get("preferred_language"):
        updates["preferred_language"] = state["preferred_language"]
        updates["conversation_language"] = state.get("conversation_language", "English" if state["preferred_language"] == "English" else "Hindi")
    elif _hindi_wins:
        updates["preferred_language"] = "Hinglish"
        updates["conversation_language"] = "Hindi"
    elif _english_wins:
        updates["preferred_language"] = "English"
        updates["conversation_language"] = "English"
    elif detected_lang and not _neutral_ack:
        # `_neutral_ack` blocks the model's read as well as the cue counts: a
        # bare "thank you" is not evidence of a language change, whichever
        # signal is asking. Guarded so that an explicit switch (branch 1) and
        # real Devanagari still get through — those carry content.
        if "hinglish" in detected_lang.lower() or "hindi" in detected_lang.lower() or (base_lang and "hindi" in base_lang.lower()):
            updates["preferred_language"] = "Hinglish"
            updates["conversation_language"] = "Hindi"
        elif "english" in detected_lang.lower() or (base_lang and "english" in base_lang.lower()):
            updates["preferred_language"] = "English"
            updates["conversation_language"] = "English"
    elif not state.get("preferred_language"):
        updates["preferred_language"] = "English"
        updates["conversation_language"] = "English"


    # ── Progress and repetition failure counter management (Issue 3 Root Cause C) ──
    from app.services.repetition_guard import count_trailing_user_repeats

    user_msg = state.get("user_message", "")
    current_failures = state.get("escalation_failure_count", 0)
    trailing_user_repeats = count_trailing_user_repeats(
        user_msg,
        state.get("messages_history", []),
    )

    PROGRESS_SLOTS = (
        "check_in",
        "check_out",
        "adults",
        "children",
        "num_rooms",
        "selected_rooms",
        "guest_name",
        "customer_identifier",
        "room_type_preference",
        "occupancy_preference",
        "special_requests",
        "subgroups",
    )
    slot_progress = any(
        updates.get(slot) is not None and updates.get(slot) != state.get(slot)
        for slot in PROGRESS_SLOTS
    )
    stage_progress = bool(
        updates.get("booking_stage")
        and updates.get("booking_stage") != state.get("booking_stage")
    )
    action_progress = bool(
        updates.get("group_lead_submitted")
        or updates.get("amenity_query")
        or resolved_intent in ("FAQ", "POLICY", "HOTEL_TASK", "ROOM_SERVICE_ORDER")
    )

    # When the guest is repeating themselves (trailing_repeats >= 2), the conversation
    # is looping; re-extracting slots from the same text does not constitute progress.
    is_progress = (slot_progress or stage_progress or action_progress) and trailing_user_repeats < 2

    if trailing_user_repeats >= 3:
        # A guest saying the same thing 3 times is the clearest possible signal of failure.
        # Immediately set to >= 3 to trigger escalation in routing.py.
        updates["escalation_failure_count"] = max(current_failures + 1, 3)
        logger.info(
            "Guest repetition threshold reached (%s times) | msg=%r | failure_count=%s",
            trailing_user_repeats,
            user_msg,
            updates["escalation_failure_count"],
        )
    elif trailing_user_repeats == 2:
        # Guest repeated for the 2nd time -> increment failure count
        updates["escalation_failure_count"] = current_failures + 1
        logger.info(
            "Guest repetition detected (2nd time) | msg=%r | failure_count=%s",
            user_msg,
            updates["escalation_failure_count"],
        )
    elif resolved_intent == "UNKNOWN":
        updates["escalation_failure_count"] = current_failures + 1
        logger.info(f"Comprehension failure (UNKNOWN intent): counter incremented to {current_failures + 1}")
    elif is_progress:
        updates["escalation_failure_count"] = 0
        logger.info("Progress detected in conversation: failure counter reset to 0")
    else:
        # Stalled turn without progress or repetition: preserve existing counter
        updates["escalation_failure_count"] = current_failures

    # Force intent to PAYMENT_STATUS if a payment screenshot is detected in the image analysis
    is_silent_upload = user_msg.startswith("[Analysis from ")

    if "[IMAGE_ANALYSIS]" in user_msg:

        import json

        if "Type: Payment Screenshot" in user_msg or "[Image Analysis: Payment Proof]" in user_msg:
            logger.info("Payment screenshot detected in user message. Forcing intent to PAYMENT_STATUS.")
            updates["intent"] = "PAYMENT_STATUS"
                
        elif "Type: Room Image" in user_msg:
            if not is_silent_upload:
                logger.info("Room image detected. Forcing intent to ROOM_FEATURE_LOOKUP.")
                updates["intent"] = "ROOM_FEATURE_LOOKUP"
                updates["escalation_required"] = False
            else:
                updates["intent"] = "SILENT_DOCUMENT_UPLOAD"
                
        elif "Type: Competitor Price Match" in user_msg:
            if not is_silent_upload:
                logger.info("Competitor price match screenshot detected. Forcing intent to PRICE_MATCH.")
                updates["intent"] = "PRICE_MATCH"
            else:
                updates["intent"] = "SILENT_DOCUMENT_UPLOAD"
                
            details_match = re.search(r"Details:\s*(.*?)\s*\[/IMAGE_ANALYSIS\]", user_msg, re.DOTALL)
            if details_match:
                try:
                    updates["price_match_data"] = json.loads(details_match.group(1))
                except json.JSONDecodeError:
                    logger.warning("Failed to decode JSON from price match details")
                    
        elif "Type: Invalid Price Match" in user_msg:
            if not is_silent_upload:
                logger.info("Invalid price match document detected.")
                updates["intent"] = "PRICE_MATCH"
            else:
                updates["intent"] = "SILENT_DOCUMENT_UPLOAD"
                
            details = "Uploaded image is not related to a hotel booking."
            details_match = re.search(r"Details:\s*(.*)", user_msg)
            if details_match:
                details = details_match.group(1).strip()
            updates["price_match_data"] = {"invalid": True, "reason": details}
            
        else:
            if is_silent_upload:
                updates["intent"] = "SILENT_DOCUMENT_UPLOAD"

    # Detect in-chat discount / price negotiation requests
    user_lower = (state.get("user_message") or "").lower()
    # The pattern moved to `app/utils/competitor_mention.py` on 2026-10-14, when
    # `price_match_handler` needed the same answer to decide whether it may
    # compare our rate to a portal's. Moved verbatim: the behaviour here is
    # unchanged, and a test pins each alternation.
    explicit_competitor = mentions_competitor(user_lower)

    user_booking_stage = state.get("booking_stage", "new")
    has_active_quote = (
        user_booking_stage in ("quote_shown", "rooms_shown", "rooms_selected", "awaiting_payment")
        or float(state.get("total_price") or 0.0) > 0
        or bool(state.get("allocation_options"))
        or bool(state.get("selected_rooms"))
    )

    # ── A booking act outranks a price phrase ─────────────────────────────────
    # The money alternation in `QUOTE_NEGOTIATION_PATTERNS` matches a *mention* of
    # an amount, and a guest who is cancelling, amending or paying almost always
    # mentions one — "advance of Rs. 3000", "refund of Rs. 3000", the "₹3,000" on a
    # receipt. On 2026-10-16 that read a cancellation as a discount demand and
    # answered it with the direct-booking pitch, so the guest was never asked
    # whether he wanted to cancel. The veto is on the act, not on the phrase,
    # because the phrase side cannot be enumerated — see `_booking_act_intent`.
    _booking_act = _booking_act_intent(updates.get("intent"), result.intent)

    if _booking_act:
        logger.info(
            f"Price/discount override skipped: turn is a booking act ({_booking_act})."
        )
        # Only when nothing has claimed the turn yet. If a word-list above already
        # wrote an intent, the veto stands aside rather than adjudicating between
        # two answers neither of which it can see the message for. `next_action` is
        # deliberately left alone: this guard exists to stop the override, not to
        # re-drive the flow.
        if updates.get("intent") is None:
            updates["intent"] = _booking_act
    elif explicit_competitor:
        logger.info("Explicit competitor price match inquiry detected. Mapping to PRICE_MATCH.")
        updates["intent"] = "PRICE_MATCH"
        updates["next_action"] = None
    elif has_active_quote and any(re.search(pat, user_lower) for pat in QUOTE_NEGOTIATION_PATTERNS):
        if user_booking_stage not in ("confirmed", "cancelled"):
            logger.info("Direct quote negotiation / discount inquiry detected on active quote/options. Mapping to PRICE_MATCH.")
            updates["intent"] = "PRICE_MATCH"
            updates["next_action"] = None
            updates["payment_missing_fields"] = []
    elif not has_active_quote and (updates.get("intent") == "PRICE_MATCH" or result.intent == "PRICE_MATCH"):
        # If user explicitly asked about discount/offers without dates yet, keep as PRICE_MATCH
        # so price_match_handler can explain direct booking benefits!
        is_discount_inquiry = any(re.search(pat, user_lower) for pat in QUOTE_NEGOTIATION_PATTERNS)
        if is_discount_inquiry:
            logger.info("General discount/offer inquiry without active quote -> keeping PRICE_MATCH to show benefits.")
            updates["intent"] = "PRICE_MATCH"
            updates["next_action"] = None
        elif state.get("check_in") or state.get("adults"):
            logger.info("General inquiry with dates misclassified as PRICE_MATCH -> routing to BOOK_ROOM.")
            updates["intent"] = "BOOK_ROOM"
            updates["next_action"] = "search"
        else:
            logger.info("General rate inquiry misclassified as PRICE_MATCH -> routing to SHOW_PRICES.")
            updates["intent"] = "SHOW_PRICES"
            updates["next_action"] = None

    # Force intent to PAYMENT_STATUS if we are in a payment follow-up flow and user is answering payment questions
    active_missing = state.get("payment_missing_fields") or []
    current_intent = updates.get("intent", result.intent)
    non_payment_intents = (
        "GREETING", "CANCEL_BOOKING", "FAQ", "HOTEL_TASK", "COMPLAINT",
        "HUMAN_REQUEST", "PRICE_MATCH", "MODIFY_BOOKING", "BOOK_ROOM",
        "CHECK_AVAILABILITY", "LIFECYCLE_RESET"
    )
    is_opt_out = bool(re.search(r"^\s*(stop|halt|quit|unsubscribe|cancel|ruko|ruk\s*jao|chhod\s*do|rahne\s*do|rehne\s*do)\s*[.!]?$", user_lower))
    if is_opt_out:
        updates["payment_missing_fields"] = []
    elif active_missing and current_intent not in non_payment_intents:
        logger.info(f"Payment follow-up active (missing: {active_missing}). Forcing intent to PAYMENT_STATUS.")
        updates["intent"] = "PAYMENT_STATUS"

        # Merge follow-up answers into existing payment_details
        extracted = result.extracted_data
        existing_pd = dict(state.get("payment_details") or {})

        # Map top-level extracted fields to payment_details fields
        field_mapping = {
            "guest_name": "customer_name",
            "guest_phone": "customer_phone",
            "guest_email": "customer_email",
            "adults": "num_adults",
            "children": "num_children",
            "num_rooms": "num_rooms",
            "check_in": "check_in",
            "check_out": "check_out",
        }

        # Pull from payment_details sub-object first
        if extracted.payment_details:
            pd_dict = extracted.payment_details.model_dump(exclude_none=True)
            for k, v in pd_dict.items():
                if v is not None:
                    existing_pd[k] = v

        # Also pull from top-level extracted fields
        extracted_dict = extracted.model_dump(exclude_none=True)
        for top_field, pd_field in field_mapping.items():
            if top_field in extracted_dict and extracted_dict[top_field] is not None:
                existing_pd[pd_field] = extracted_dict[top_field]

        updates["payment_details"] = existing_pd

        # Recalculate which fields are still missing
        from app.services.image_vision import CRITICAL_FIELDS, IMPORTANT_FIELDS
        still_missing = []
        for f in active_missing:
            val = existing_pd.get(f)
            if val is None or (isinstance(val, str) and val.strip() == ""):
                # Check if we can fill from session context
                session_map = {"check_in": "check_in", "check_out": "check_out",
                               "num_adults": "adults", "num_children": "children",
                               "num_rooms": "num_rooms", "customer_name": "guest_name",
                               "customer_phone": "guest_phone", "customer_email": "guest_email"}
                session_field = session_map.get(f)
                session_val = state.get(session_field) if session_field else None
                if session_val is not None:
                    existing_pd[f] = session_val
                else:
                    still_missing.append(f)
        updates["payment_missing_fields"] = still_missing if still_missing else None
        updates["payment_details"] = existing_pd

    # If the user is in the middle of a cancellation request, do not parse booking parameter changes
    is_cancellation = state.get("booking_stage") == "cancellation_pending"

    # ── CRITICAL: Only modify booking state for BOOKING-RELATED intents ──
    # Non-booking intents (FAQ, ROOM_FEATURE_LOOKUP, SHOW_AMENITIES, etc.)
    # should NEVER wipe dates, guests, rooms, or booking progress.
    # Also skip when we just cleared the draft (reset / decline continue).
    booking_intents = {
        "BOOK_ROOM", "MODIFY_BOOKING", "SELECT_ROOM", "CONFIRM_BOOKING",
        "CHECK_AVAILABILITY", "CANCEL_BOOKING", "PAYMENT_STATUS", "PRICE_MATCH", "BOOKING_SUMMARY"
    }
    
    draft_cleared = bool(override_extras.get("abandoned_draft"))
    if draft_cleared:
        # ── MULTI-BOOKING QUEUE CONTINUATION (ABANDONED) ──
        from app.utils.queue_manager import advance_booking_queue
        q_updates = advance_booking_queue(state, "abandoned")
        if q_updates.get("queue_advanced"):
            # Merge the new active booking into override_extras so it gets loaded instead of staying cleared
            for k, v in q_updates.items():
                override_extras[k] = v
            override_extras["abandoned_draft"] = False # Un-abandon it since we are continuing!
            draft_cleared = False
            updates["multi_booking_detected"] = True # Force response_generator to say something
            
    is_booking_intent = (
        not draft_cleared
        and (resolved_intent in booking_intents or result.intent in booking_intents)
    )

    if not is_cancellation and is_booking_intent:
        if state.get("booking_stage") == "lead_created" and not updates.get("group_lead_submitted"):
            updates["booking_stage"] = "new"
            updates["group_lead_submitted"] = False
            if updates.get("next_action") == "lead_created":
                updates["next_action"] = None

        extracted = result.extracted_data
        
        # ── MULTI-BOOKING QUEUE LOGIC ──
        if getattr(result, "extracted_bookings", None):
            queue_len = len(result.extracted_bookings)
            if queue_len > 1:
                from app.observability.metrics import metrics
                import random
                from datetime import datetime, timezone, timedelta
                
                _IST = timezone(timedelta(hours=5, minutes=30))
                now_iso = datetime.now(_IST).strftime("%Y-%m-%dT%H:%M:%S+05:30")
                queue_id = f"QUEUE-{datetime.now(_IST).strftime('%Y%m%d')}-{random.randint(10000, 99999)}"
                logger.info(f"MULTI_BOOKING_DETECTED: {queue_len} bookings. QUEUE_CREATED: {queue_id}")
                metrics.incr("queue.created")
                
                queue_data = []
                for b in result.extracted_bookings:
                    d = b.model_dump(exclude_none=True)
                    d["queue_status"] = "pending"
                    d["queue_id"] = queue_id
                    d["created_at"] = now_iso
                    d["last_updated"] = now_iso
                    d["completed_at"] = None
                    queue_data.append(d)
                    
                # Mark first as immediately processing
                queue_data[0]["queue_status"] = "processing"
                queue_data[0]["last_updated"] = now_iso
                
                updates["booking_queue"] = queue_data
                updates["active_booking_index"] = 0
                updates["multi_booking_detected"] = True
                updates["queue_id"] = queue_id
            # Pull the first booking to process right now
            extracted = result.extracted_bookings[0]
            
        # ── TARGETED QUEUE MODIFICATIONS ──
        if getattr(result, "target_queue_index", None) is not None:
            t_idx = result.target_queue_index
            active_idx = updates.get("active_booking_index", state.get("active_booking_index", 0))
            queue = updates.get("booking_queue", state.get("booking_queue", []))
            
            if 0 <= t_idx < len(queue) and t_idx != active_idx:
                logger.info(f"Targeted queue modification for index {t_idx} (active is {active_idx})")
                target_booking = dict(queue[t_idx])
                
                if resolved_intent == "CANCEL_BOOKING" or result.intent == "CANCEL_BOOKING":
                    target_booking["queue_status"] = "cancelled"
                    target_booking["terminal_status"] = "cancelled"
                    updates["intent"] = "QUEUE_MODIFICATION"
                    updates["bot_response"] = f"✅ The booking for {target_booking.get('guest_name', 'that guest')} has been cancelled."
                else:
                    # It's a modification
                    ex_dict = extracted.model_dump(exclude_none=True)
                    for k, v in ex_dict.items():
                        target_booking[k] = v
                    updates["intent"] = "QUEUE_MODIFICATION"
                    updates["bot_response"] = f"✅ The details for {target_booking.get('guest_name', 'that guest')}'s booking have been updated."
                    
                queue[t_idx] = target_booking
                updates["booking_queue"] = queue
                
                # Prevent these extracted fields from polluting the active booking
                extracted = result.extracted_data.__class__() # Empty it out so no changed_fields are detected
                resolved_intent = updates["intent"]
            
        # Clean feature preferences early to prevent false change detection
        if extracted.feature_preferences is not None:
            cleaned_features = [
                f for f in extracted.feature_preferences 
                if f and f.lower().strip() not in ["none", "null", "n/a", ""]
            ]
            extracted.feature_preferences = cleaned_features if cleaned_features else None

        core_fields = ["adults", "children", "child_ages", "num_rooms", "check_in", "check_out", "room_type_preference", "ambiguous_guests", "budget", "feature_preferences", "meal_plan_preference"]
        changed_fields = []
        extracted_dict = extracted.model_dump(exclude_none=True)

        # ── HARD DATE VALIDATION & ORDER ENFORCEMENT ──
        if "check_in" in extracted_dict and "check_out" not in extracted_dict:
            new_ci = extracted_dict["check_in"]
            old_ci = state.get("check_in")
            old_co = state.get("check_out")
            if old_ci and old_co:
                try:
                    from datetime import datetime, timedelta
                    d_ci = datetime.strptime(old_ci, "%Y-%m-%d")
                    d_co = datetime.strptime(old_co, "%Y-%m-%d")
                    duration = max((d_co - d_ci).days, 1)
                    new_ci_dt = datetime.strptime(new_ci, "%Y-%m-%d")
                    new_co = (new_ci_dt + timedelta(days=duration)).strftime("%Y-%m-%d")
                    extracted_dict["check_out"] = new_co
                    extracted.check_out = new_co
                except Exception:
                    pass
            elif not old_co:
                try:
                    from datetime import datetime, timedelta
                    new_ci_dt = datetime.strptime(new_ci, "%Y-%m-%d")
                    new_co = (new_ci_dt + timedelta(days=1)).strftime("%Y-%m-%d")
                    extracted_dict["check_out"] = new_co
                    extracted.check_out = new_co
                except Exception:
                    pass

        if "check_in" in extracted_dict or "check_out" in extracted_dict:
            from app.utils.helpers import normalize_date, validate_proposed_date_range
            
            raw_ci = extracted_dict.get("check_in")
            raw_co = extracted_dict.get("check_out")
            
            norm_ci = normalize_date(raw_ci) if raw_ci else None
            norm_co = normalize_date(raw_co) if raw_co else None
            
            if norm_ci:
                extracted_dict["check_in"] = norm_ci
                extracted.check_in = norm_ci
            if norm_co:
                extracted_dict["check_out"] = norm_co
                extracted.check_out = norm_co

            proposed_ci = extracted_dict.get("check_in") or state.get("check_in")
            proposed_co = extracted_dict.get("check_out") or state.get("check_out")

            is_valid_dates, date_err_msg = validate_proposed_date_range(proposed_ci, proposed_co)
            if not is_valid_dates:
                logger.warning(f"Invalid date proposal detected: check_in={proposed_ci}, check_out={proposed_co}. Error: {date_err_msg}")
                # Do NOT mutate state with invalid dates
                extracted_dict.pop("check_in", None)
                extracted_dict.pop("check_out", None)
                extracted.check_in = None
                extracted.check_out = None
                updates["bot_response"] = date_err_msg
                updates["next_action"] = "invalid_dates"


        # ── BOOKING MODIFICATION INTERCEPTION ──
        booking_status = state.get("booking_status", "new")
        stage = state.get("booking_stage", "new")
        is_confirmed_or_checked_in = (
            booking_status in ("confirmed", "checked_in")
            or stage in ("confirmed", "cancellation_pending")
        )

        if updates.get("intent") == "PRICE_MATCH":
            pass
        elif result.intent == "MODIFY_BOOKING" and not is_confirmed_or_checked_in:
            # Pre-confirmation draft modification: update active search parameters directly
            updates["intent"] = "BOOK_ROOM"
            resolved_intent = "BOOK_ROOM"
            result.intent = "BOOK_ROOM"
        elif result.intent == "MODIFY_BOOKING":
            # Load existing pending modification state for confirmed bookings
            mod_fields = list(state.get("modified_fields") or [])
            prev_vals = dict(state.get("previous_values") or {})
            req_vals = dict(state.get("requested_values") or {})
            
            # We check all core fields that can be modified
            mod_track_fields = ["check_in", "check_out", "adults", "children", "num_rooms", "room_type_preference", "meal_plan_preference", "special_requests"]
            
            new_changes_detected = False
            for field in mod_track_fields:
                if field in extracted_dict:
                    new_val = extracted_dict[field]
                    old_val = state.get(field)
                    
                    # Ignore no-op updates (same as current active booking)
                    if old_val == new_val:
                        continue
                    if isinstance(old_val, str) and isinstance(new_val, str) and old_val.lower().strip() == new_val.lower().strip():
                        continue
                        
                    # Ignore if this exact change is already pending
                    if req_vals.get(field) == new_val:
                        continue

                    if field not in mod_fields:
                        mod_fields.append(field)
                        prev_vals[field] = old_val
                    req_vals[field] = new_val
                    new_changes_detected = True
                        
            if mod_fields:
                lines = ["**Booking Changes Detected**"]
                for f in mod_fields:
                    formatted_field = f.replace("_", " ").title()
                    old_str = prev_vals[f] if prev_vals[f] is not None else "None"
                    lines.append(f"• {formatted_field}: {old_str} → {req_vals[f]}")
                
                # Maintain Modification Metadata
                from datetime import datetime, timezone
                now_str = datetime.now(timezone.utc).isoformat()
                meta = dict(state.get("modification_metadata") or {})
                if not meta.get("modification_id"):
                    import random
                    meta["modification_id"] = f"MOD-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{random.randint(100, 999)}"
                    meta["created_at"] = now_str
                meta["updated_at"] = now_str
                meta["status"] = "pending"

                updates["modified_fields"] = mod_fields
                updates["previous_values"] = prev_vals
                updates["requested_values"] = req_vals
                updates["booking_change_summary"] = "\n".join(lines)
                updates["modification_metadata"] = meta
                
                logger.info(f"Booking Modification Intercepted/Updated: {mod_fields} | Meta: {meta['modification_id']}")
                # We skip applying these directly to state to protect the current booking!
                extracted_dict = {} # Clear it so normal apply logic does nothing
                extracted = result.extracted_data.__class__() # Empty out extracted
            else:
                if updates.get("intent") != "PRICE_MATCH":
                    # If no actual changes detected, fallback to BOOK_ROOM so the flow continues smoothly
                    updates["intent"] = "BOOK_ROOM"
                    result.intent = "BOOK_ROOM"
                
        # ── PENDING MODIFICATION SELECTION INTERCEPTION ──
        if result.intent == "SELECT_ROOM" and state.get("requested_values"):
            logger.info("User selected a room while modification is pending. Flushing requested values into active state.")
            req_vals = state.get("requested_values")
            for k, v in req_vals.items():
                # We put them directly into extracted_dict so they trigger the normal change detection!
                extracted_dict[k] = v
                
            # Clear pending modification state so it doesn't pollute future turns
            updates["requested_values"] = None
            updates["modified_fields"] = None
            updates["booking_change_summary"] = None
            updates["modification_metadata"] = None

        # Guard against option number / menu selection / continue phrases being extracted as num_rooms / adults
        #
        # Every count regex in this block reads `countable_text` — this message
        # with its dates blanked — and never `user_msg_clean`. A date puts
        # numbers in exactly the position a count would occupy, and on
        # 2026-09-24 "…check out 15/11/26 person 5+1" had its year read as the
        # party size: the regex below matched "26 person" and a guest who asked
        # for six people was told the bot had 26, then that no room could seat
        # them. See app/utils/date_masking.py, and the second, independent
        # guard in extraction_contract Rule 1.
        countable_text = mask_dates(user_msg_clean)

        is_comp_option = bool(
            expected_action == "need_composition_preference" 
            and not re.search(r'\b\d+\s*(?:rooms?|kamre?|kamra)\b', countable_text)
        )
        is_menu_option = bool(
            expected_action in ("need_ac_preference", "need_meal_plan_preference", "change_menu_shown", "need_composition_preference")
            and not re.search(r'\b\d+\s*(?:adults?|persons?|people|pax|guests?|rooms?|kamre?|bachche?|children)\b', countable_text)
        )
        is_continue_msg = bool(re.search(r'\b(?:continue|resume|pick\s+up|aage\s+badho)\b', user_msg_clean))
        has_explicit_capacity_phrase = bool(re.search(r'\b\d+\s*(?:adults?|persons?|people|pax|guests?|members?|rooms?|kamre?|kamra|bachche?|children)\b', countable_text))

        is_option_number_selection = bool(
            not has_explicit_capacity_phrase
            and (
                is_comp_option
                or is_menu_option
                or is_continue_msg
                or (
                    (updates.get("intent") == "SELECT_ROOM" or result.intent == "SELECT_ROOM")
                    and (
                        updates.get("selected_option_number")
                        or updates.get("selected_option_label")
                        or updates.get("room_type_preference") in ("cheapest", "expensive")
                        or re.search(r'^\s*(?:option|opt|no\.?|choice)?\s*([1-9][a-zA-Z]?|[a-zA-Z][1-9])\.?\s*$', (state.get("user_message") or "").strip(), re.I)
                        or re.search(r'\b(cheapest|chepest|sasta|lowest\s*price|best\s*value|most\s*affordable|budget\s*option)\b', (state.get("user_message") or "").strip(), re.I)
                    )
                )
            )
        )
        if has_explicit_capacity_phrase and (updates.get("intent") == "SELECT_ROOM" or result.intent == "SELECT_ROOM"):
            updates["intent"] = "BOOK_ROOM"

        # Perform change detection before updating state for NORMAL bookings
        for field in core_fields:
            if field in extracted_dict:
                new_val = extracted_dict[field]
                old_val = state.get(field)
                
                # If child_ages was extracted as empty/falsy but old_val exists, default extraction didn't specify ages; keep old
                if field == "child_ages" and not new_val and old_val:
                    continue

                old_val_clean = old_val if old_val else None
                new_val_clean = new_val if new_val else None
                
                if old_val_clean != new_val_clean:
                    # If they are actively selecting a room or confirming a booking, don't flag capacity or preference fields as changes!
                    if (
                        result.intent in ("SELECT_ROOM", "CONFIRM_BOOKING")
                        or updates.get("intent") in ("SELECT_ROOM", "CONFIRM_BOOKING")
                        or is_option_number_selection
                    ):
                        if field in ("room_type_preference", "recommendation_preference", "selected_option_label", "selected_option_number", "num_rooms", "adults", "children", "child_ages"):
                            continue
                    changed_fields.append(field)

        # Deterministic extraction of multi-family / person counts (e.g. "6 person(3/3)-2 family", "6 people 2 rooms")

        m_fam = re.search(r'(\d+)\s*(?:person|people|pax|members?|guests?)?\s*\(\s*(\d+)\s*/\s*(\d+)\s*\)\s*[- ]*(\d+)\s*famil', countable_text)
        if m_fam:
            total_pax = int(m_fam.group(1))
            num_fams = int(m_fam.group(4))
            updates["adults"] = total_pax
            updates["effective_adults"] = total_pax
            updates["children"] = 0
            updates["effective_children"] = 0
            updates["child_ages"] = []
            updates["child_age_ask_count"] = 0
            updates["num_rooms"] = num_fams
        else:
            has_child_kw = bool(re.search(r'\b(?:child|children|kids?|bachche?|bacha|baby|infant)\b', user_msg_clean))

            # A guest-stated split is the party as well as the rooms: nine
            # guests is "2 people 1 room, 3 people 1 room, 4 people 1 room",
            # and the split has already written that sum by the time we get
            # here — on the custom branch. The regex below would read the "2"
            # out of the first line and shrink the party to two. The main
            # branch writes its split *after* this point rather than before, so
            # without this guard the two branches answer the same sentence
            # differently. Requirements carrying `guests` are the better
            # reader; see `is_an_explicit_split`.
            if not is_an_explicit_split(updates.get("room_requirements")):
                m_pax = re.search(r'\b(\d+)\s*(?:persons?|people|pax|members?|guests?|adults?)\b', countable_text)
                if not m_pax and re.search(r'^\s*(\d+)\s*$', user_msg_clean):
                    if (
                        "adults" in (state.get("missing_fields") or [])
                        or expected_action in ("need_composite_slots", "need_guests")
                        # `expected_action` already covers `need_composite_slots`; only
                        # `need_guests` is reachable through this clause, and it was dead
                        # for the same reason as the two readers above — `next_action` is
                        # None by the time a turn runs. It reads the surviving pointer.
                        or active_prompt_action in ("need_composite_slots", "need_guests")
                    ):
                        m_pax = re.search(r'^\s*(\d+)\s*$', user_msg_clean)

                # The model reads first; the regex is the fallback, not a rival.
                # The decision is a pure function so the ordering is testable
                # without a model call — see app/utils/pax_resolution.py, which
                # carries the 2026-09-24 defect and the rule it produced.
                resolved, used_regex = resolve_pax_counts(
                    model_adults=extracted.adults,
                    model_children=extracted.children,
                    regex_pax=int(m_pax.group(1)) if m_pax else None,
                    has_child_kw=has_child_kw,
                    is_option_number_selection=is_option_number_selection,
                    expected_action=expected_action,
                )
                updates.update(resolved)
                if used_regex:
                    # The room count beside the pax phrase belongs to the regex
                    # branch only; the model's own `num_rooms` is applied later.
                    m_rms = re.search(r'\b(\d+)\s*(?:rooms?|kamre?|famil(?:y|ies))\b', countable_text)
                    if m_rms:
                        updates["num_rooms"] = int(m_rms.group(1))
        if not updates.get("child_ages"):
            if getattr(extracted, "child_ages", None):
                updates["child_ages"] = extracted.child_ages
            else:
                # Deterministic extraction of child ages from patterns like "Age 9 Yrs and 3 Yrs", "1 child (age 8)", "age 3 and 14", "1.5 saal"
                age_matches = re.findall(r'\b(\d+(?:\.\d+)?)\s*(?:saal|years?|yrs?)\b', user_msg_clean)
                if age_matches:
                    updates["child_ages"] = [int(float(a)) for a in age_matches if int(float(a)) <= 17]

                if not updates.get("child_ages"):
                    age_group_match = re.search(r'(?:age|aged|saal|umar)\s*([\d\s,and&yrolds]+)', user_msg_clean)
                    if age_group_match:
                        found_nums = [int(float(n)) for n in re.findall(r'\b\d+(?:\.\d+)?\b', age_group_match.group(1)) if int(float(n)) <= 17]
                        if found_nums:
                            updates["child_ages"] = found_nums
                if not updates.get("child_ages"):
                    paren_matches = re.findall(r'\((?:age\s*)?([\d\s,and&a-zA-Z]+)\)', user_msg_clean)
                    if paren_matches:
                        for pm in paren_matches:
                            if any(w in pm.lower() for w in ("age", "yr", "year", "saal", "child", "kid")):
                                found_nums = [int(float(n)) for n in re.findall(r'\b\d+(?:\.\d+)?\b', pm) if int(float(n)) <= 17]
                                if found_nums:
                                    updates["child_ages"] = found_nums
                                    break

        # Deterministic extraction of AC preference from user message.
        #
        # The comparison clause asks whether the guest named BOTH classes, and
        # it asks that with `mentions_both_classes` rather than by checking the
        # two vocabularies side by side — because "without ac" contains the
        # word AC without mentioning a second class, and read carelessly it
        # looks like a guest weighing the two up.
        #
        # `has_both_keyword` stays narrower than the shared no-preference
        # vocabulary on purpose: "any ac?" is a question about AC, and only the
        # words below mean the guest wants to see both classes. The class
        # decision itself is the shared resolver's, which is what makes
        # "non - ac" and "a/c" resolve here as they do everywhere else.
        has_both_keyword = bool(re.search(r'\b(?:show\s*both|both|dono|all|sab)\b', user_msg_clean))
        is_ac_comparison_or_question = (
            (mentions_both_classes(user_msg_clean) and user_msg_clean not in ("non ac", "non-ac", "non_ac"))
            or " or " in user_msg_clean
            or " ya " in user_msg_clean
            or "how do" in user_msg_clean
            or "difference" in user_msg_clean
            or (user_msg_clean.endswith("?") and not has_both_keyword)
            or "kaise" in user_msg_clean
            or "kya" in user_msg_clean
        )

        if has_both_keyword and not any(w in user_msg_clean for w in ("how do", "difference", "kaise", "kya")):
            updates["ac_preference"] = "SHOW_BOTH"
            updates["ac_qualified"] = True
        elif not is_ac_comparison_or_question:
            resolved_ac = ac_preference_from_text(user_msg_clean)
            if resolved_ac:
                updates["ac_preference"] = "SHOW_BOTH" if resolved_ac == "NO_PREFERENCE" else resolved_ac
                updates["ac_qualified"] = True
        if extracted.ambiguous_guests is not None and not is_option_number_selection:
            updates["ambiguous_guests"] = extracted.ambiguous_guests
        if extracted.num_rooms is not None and not is_option_number_selection:
            # Guard against date digits (e.g. 5 from "5 to 7 sep") or child ages (e.g. 9 from "age 4 and 9") being wrongly parsed as num_rooms
            has_room_kw = bool(re.search(r'\b\d+\s*(?:rooms?|kamre?|kamra|famil(?:y|ies))\b', user_msg_clean))
            is_date_digit = bool(re.search(rf'\b{extracted.num_rooms}\s*(?:to|-|se|\.|\/)\b', user_msg_clean))
            is_child_age = extracted.num_rooms in (updates.get("child_ages") or []) or bool(re.search(rf'\b(?:age|aged|saal|umar)\s*.*?\b{extracted.num_rooms}\b', user_msg_clean))
            if has_room_kw:
                updates["num_rooms"] = extracted.num_rooms
            elif not is_date_digit and not is_child_age:
                updates["num_rooms"] = extracted.num_rooms
        if getattr(extracted, "room_requirements", None) and not updates.get("room_requirements"):
            reqs = [r.model_dump() if hasattr(r, "model_dump") else dict(r) for r in extracted.room_requirements]
            updates["room_requirements"] = reqs
            updates["force_new_search"] = True
            updates["booking_stage"] = "collecting"
            updates["next_action"] = "search"
            updates["selected_rooms"] = []
            updates["room_type_not_available"] = False
            changed_fields.append("room_requirements")
            # A guest-stated split is the party as well as the rooms. Note the
            # `booking_stage = "collecting"` two lines up and the hard reset
            # further down: this stage does NOT survive to `room_searcher`, so
            # the routing there keys off the requirements themselves rather
            # than off a stage that gets overwritten. See
            # `room_searcher.search_rooms`.
            if is_an_explicit_split(reqs):
                _apply_split_party(
                    updates,
                    state,
                    reqs,
                    sum(
                        int(r.get("count") or 0) * int(r.get("guests") or 0)
                        for r in reqs
                    ),
                )
        if extracted.budget is not None:
            updates["budget"] = extracted.budget
            
        if extracted.feature_preferences is not None:
            updates["feature_preferences"] = extracted.feature_preferences
        if extracted.check_in is not None:
            updates["check_in"] = extracted.check_in
        if extracted.meal_plan_preference is not None and not _meal_plan_from_button:
            # A tap on a meal button must not be overwritten here. Measured 2026-10-15: the
            # fast path stores the clean code ("MAP"), then — because the meal arm does not
            # return — the turn reaches this extraction, and the model is free to answer with
            # the button's own label echoed back ("MAP (Bkfast+Din)"). The merge at the end of
            # this function lets `updates` win, so the raw label landed in state instead.
            #
            # It was harmless only by luck: `get_preferred_rate_plan` keyword-matches, so
            # `"map" in "map (bkfast+din)"` still picked the MAP plan. It is a trap for
            # anything that exact-matches, and the lifecycle summary prints the raw string to
            # staff. Typed text is unaffected — the flag is only set by a resolved tap.
            updates["meal_plan_preference"] = extracted.meal_plan_preference
        if getattr(extracted, "recommendation_preference", None) is not None:
            updates["recommendation_preference"] = extracted.recommendation_preference
        if getattr(extracted, "display_preference", None) is not None:
            updates["display_preference"] = extracted.display_preference
        if getattr(extracted, "room_count_preference", None) is not None:
            updates["room_count_preference"] = extracted.room_count_preference
        if getattr(extracted, "comfort_preference", None) is not None:
            updates["comfort_preference"] = extracted.comfort_preference
        if extracted.check_out is not None:
            updates["check_out"] = extracted.check_out
        if extracted.room_type_preference is not None:
            clean_rt = extracted.room_type_preference.strip().lower()
            resolved_ac = ac_preference_from_text(clean_rt)
            if resolved_ac:
                updates["ac_preference"] = resolved_ac
                updates["ac_qualified"] = True
            # "AC Deluxe" is a class AND a room name: the class is banked above,
            # and the name still has to go through or the alias resolver never
            # sees the room the guest actually asked for. A bare class
            # ("non - ac", "a/c room") is not a name and must not become one.
            if (
                not is_bare_ac_class(clean_rt)
                and not clean_rt.startswith("option ")
                and not clean_rt.isdigit()
                and clean_rt not in ("custom", "build your own")
            ):
                updates["room_type_preference"] = extracted.room_type_preference
        if extracted.guest_name is not None:
            g_name = str(extracted.guest_name).strip()
            inquiry_words = ("vacancy", "job", "jobs", "manager", "hiring", "interview", "receptionist", "resume", "cv", "?")
            if (
                g_name.lower() not in NON_NAME_WORDS
                and len(g_name.split()) <= 4
                and not any(w in g_name.lower().split() for w in inquiry_words)
                and g_name.lower() not in ("hotel", "hotel sahu")
                and not is_inquiry_or_question
            ):
                updates["guest_name"] = g_name.title()
        if extracted.guest_email is not None:
            updates["guest_email"] = extracted.guest_email
        if extracted.guest_phone is not None:
            updates["guest_phone"] = extracted.guest_phone

        # Dynamic Language Switching / Detection (English is Primary Default)
        #
        # ── There are two language blocks in this function, and this one runs
        #    LAST, so whatever it writes wins ──────────────────────────────
        #
        # The newer block sits ~800 lines above and does the real lexical work.
        # This is the older one, never removed when the newer one was added,
        # and it carried a *second*, smaller copy of the cue logic plus a
        # *second* English default. On the turn a language is first
        # established `state["preferred_language"]` is still unset — so that
        # default overwrote the Hinglish the block above had just chosen.
        #
        # That is the mechanism behind "it still replies in english most of the
        # time": the newer block was doing its job and being clobbered here.
        # "mera naam Rahul hai" and "mujhe room book karna hai" matched nothing
        # in the regex below, fell through to the English default, and threw
        # away the Hinglish decision made 800 lines earlier.
        #
        # Under `LANGUAGE_DETECTION_WEIGHTS_HINDI` this block keeps only what
        # it alone can do — an explicit request, and Devanagari — and leaves
        # the cues and the default to the block that owns them.
        clean_raw_msg = (state.get("user_message") or "").strip().lower()
        if any(w in clean_raw_msg for w in ("in english", "english please", "speak in english", "english mein", "talk in english")) or clean_raw_msg == "english":
            updates["preferred_language"] = "English"
            updates["conversation_language"] = "English"
        elif clean_raw_msg in ("hindi", "in hindi", "hindi please", "hindi mein", "hindi me", "हिंदी"):
            updates["preferred_language"] = "Hindi"
            updates["conversation_language"] = "Hindi"
        elif clean_raw_msg in ("hinglish", "in hinglish", "hinglish please"):
            updates["preferred_language"] = "Hinglish"
            updates["conversation_language"] = "Hindi"
        elif re.search(r'[\u0980-\u09FF]', state.get("user_message", "")) or clean_raw_msg in ("bengali", "in bengali", "bangla", "বাংলা"):
            updates["preferred_language"] = "Bengali"
            updates["conversation_language"] = "Bengali"
        elif re.search(r'[\u0900-\u097F]', state.get("user_message", "")):
            updates["preferred_language"] = "Hindi"
            updates["conversation_language"] = "Hindi"
        elif not _full_weights and re.search(r'\b(chahiye|kripya|bataiye|dijiye|hoga|rahega|kitna|kitne|kamra|kamre|dono|sasta|mehenga|batao|karo|karein|hain|nahi|haan)\b', clean_raw_msg):
            updates["preferred_language"] = "Hinglish"
            updates["conversation_language"] = "Hindi"
        elif not _full_weights and not state.get("preferred_language"):
            updates["preferred_language"] = "English"
            updates["conversation_language"] = "English"
        if extracted.special_requests is not None:
            updates["special_requests"] = extracted.special_requests
        if extracted.payment_details is not None:
            pd_dict = extracted.payment_details.model_dump(exclude_none=True)
            if pd_dict:
                updates["payment_details"] = pd_dict

                # Initialize missing fields only for actual payment receipt uploads
                user_msg = state.get("user_message", "")
                is_payment_image = (
                    ("[IMAGE_ANALYSIS]" in user_msg and "Type: Payment Screenshot" in user_msg)
                    or "[Image Analysis: Payment Proof]" in user_msg
                    or "Type: Payment Screenshot" in user_msg
                )
                if is_payment_image:
                    # Downstream Payment Verification Logic (Prompt 2 Task 4)
                    p_amount = float(pd_dict.get("amount") or 0.0)
                    p_status = str(pd_dict.get("status") or "").strip().lower()
                    tot_price = float(state.get("total_price") or 0.0)
                    exp_advance = round(tot_price * 0.30, 2) if tot_price > 0 else 0.0
                    logger.info("PAYMENT_VERIFY_START | amount=%s | status=%s | expected_advance=%s | sess=%s", p_amount, p_status, exp_advance, state.get("session_id"))

                    gap_adv = abs(p_amount - exp_advance) if exp_advance > 0 else 0.0
                    gap_full = abs(p_amount - tot_price) if tot_price > 0 else 0.0
                    min_gap = min(gap_adv, gap_full)

                    is_success_status = p_status in ("success", "completed", "paid", "successful")

                    if is_success_status and (min_gap <= 10.0 or p_amount >= (exp_advance - 10.0 if exp_advance > 0 else 1.0)):
                        logger.info("PAYMENT_VERIFY_RESULT | outcome=auto_confirmed | p_amount=%s | exp_advance=%s | gap=%s", p_amount, exp_advance, min_gap)
                        updates["booking_stage"] = "confirmed"
                        updates["payment_status"] = "paid"
                        updates["amount_paid"] = p_amount
                        updates["intent"] = "PAYMENT_STATUS"
                    elif 10.0 < min_gap <= 500.0 and is_success_status:
                        logger.info("PAYMENT_VERIFY_RESULT | outcome=staff_review | p_amount=%s | exp_advance=%s | gap=%s", p_amount, exp_advance, min_gap)
                        updates["intent"] = "PAYMENT_STATUS"
                        updates["payment_status"] = "review_needed"
                    elif p_status in ("pending", "processing", "failed"):
                        logger.info("PAYMENT_VERIFY_RESULT | outcome=pending_or_failed | p_status=%s", p_status)

                    if not state.get("payment_missing_fields"):
                        from app.services.image_vision import _compute_missing_fields, CRITICAL_FIELDS, IMPORTANT_FIELDS
                        all_missing = _compute_missing_fields(pd_dict)
                        trackable = [f for f in all_missing if f in CRITICAL_FIELDS or f in IMPORTANT_FIELDS]
                        if trackable:
                            updates["payment_missing_fields"] = trackable
                            updates["payment_followup_index"] = 0
                            logger.info(f"Payment receipt missing fields: {trackable}")
        if extracted.partial_payment_amount is not None:
            updates["partial_payment_amount"] = extracted.partial_payment_amount
        if extracted.coupon_code is not None:
            updates["coupon_code"] = extracted.coupon_code
            changed_fields.append("coupon_code")
            
        if extracted.extra_services is not None:
            # Append newly extracted services to existing
            current_extras = state.get("extra_services", [])
            new_extras = [s for s in extracted.extra_services if s not in current_extras]
            if new_extras:
                updates["extra_services"] = current_extras + new_extras
                changed_fields.append("extra_services")

        # Process ignored/declined upsells
        last_upsell = state.get("last_upsell_offered")
        if last_upsell and (updates.get("intent") in ["CONFIRM_BOOKING", "PAYMENT_STATUS", "BOOK_ROOM"]):
            # If they didn't add it in this turn, consider it declined/ignored
            if last_upsell not in (extracted.extra_services or []):
                declined = state.get("declined_upsells", [])
                if last_upsell not in declined:
                    updates["declined_upsells"] = declined + [last_upsell]
                    
                    from app.repositories.activity_repository import log_activity, ActivityLog
                    import asyncio
                    asyncio.create_task(log_activity(ActivityLog(
                        hotel_id=str(state.get("hotel_id", 0)),
                        session_id=state.get("session_id", ""),
                        booking_id=state.get("booking_id"),
                        activity_type="SYSTEM_EVENT",
                        description=f"UPSELL_DECLINED: {last_upsell}"
                    )))
            updates["last_upsell_offered"] = None # consume

        # If any core search parameter changed, invalidate downstream context
        is_payment_receipt = (
            ("[IMAGE_ANALYSIS]" in state.get("user_message", "") and "Type: Payment Screenshot" in state.get("user_message", ""))
            or "[Image Analysis: Payment Proof]" in state.get("user_message", "")
            or "Type: Payment Screenshot" in state.get("user_message", "")
        )
        current_stage = state.get("booking_stage")
        ignore_changes = current_stage == "awaiting_payment" and (is_payment_receipt or updates.get("intent") == "PAYMENT_STATUS")

        if changed_fields and not ignore_changes:
            logger.info(f"Booking query change detected in: {changed_fields}.")
            updates["updated_fields"] = changed_fields

            # True structural/capacity changes that invalidate existing pricing & selected room
            hard_reset_fields = {"check_in", "check_out", "adults", "children", "num_rooms", "child_ages"}
            is_confirming = (updates.get("intent") in ("SELECT_ROOM", "CONFIRM_BOOKING") or result.intent in ("SELECT_ROOM", "CONFIRM_BOOKING"))
            requires_hard_reset = any(f in changed_fields for f in hard_reset_fields) and not is_option_number_selection and not is_confirming

            # Preference/taste changes (room_type_preference, budget, feature_preferences, meal_plan_preference, etc.)
            preference_only_changed = not requires_hard_reset and any(
                f in changed_fields for f in ["room_type_preference", "budget", "feature_preferences", "meal_plan_preference", "recommendation_preference", "display_preference", "room_count_preference", "comfort_preference"]
            )

            # If user changed room preference, AC preference, or budget, wipe selected rooms and trigger fresh search
            if preference_only_changed and not is_option_number_selection and not is_confirming:
                logger.info("Room preference change detected. Triggering fresh room search and wiping stale selections.")
                updates["selected_rooms"] = []
                updates["allocation_options"] = []
                updates["price_breakdown"] = None
                updates["total_price"] = None
                updates["payment_link"] = None
                updates["payment_link_id"] = None
                updates["payment_link_amount"] = None
                updates["force_new_search"] = True
                updates["booking_stage"] = "collecting"
                updates["next_action"] = "search"
            elif requires_hard_reset:
                # Always invalidate pricing and payment link when dates or capacity change
                updates["price_breakdown"] = None
                updates["total_price"] = None
                updates["payment_link"] = None
                updates["payment_status"] = None
                updates["booking_id"] = None
                updates["room_type_not_available"] = False
                updates["allocation_options"] = []
                updates["force_new_search"] = True

                # Check if TRUE capacity parameters changed (adults, children, num_rooms, child_ages)
                capacity_fields = {"adults", "children", "num_rooms", "child_ages"}
                capacity_changed = any(f in changed_fields for f in capacity_fields) and not is_option_number_selection and not is_confirming

                if capacity_changed:
                    logger.info("Capacity parameters changed. Wiping selected rooms and resetting upsell.")
                    updates["selected_rooms"] = []
                    updates["available_rooms"] = []
                    # A room preference is pinned against a party size, so a
                    # change to the party makes it stale: "a deluxe room" names a
                    # different room for two people than for six, and one the
                    # guest chose for five says nothing about six. Left in place
                    # it kept narrowing the search that followed — `room_searcher`
                    # reads it and `filter_rooms_by_reference` allocates over
                    # only the rooms it matched — which is how a party that grew
                    # by one child went on being quoted the room it had outgrown.
                    #
                    # Not when the guest restated it in this same message: then
                    # it is in `changed_fields` too, and it is current.
                    if (
                        get_settings().PARTY_CHANGE_DROPS_ROOM_PREFERENCE
                        and state.get("room_type_preference")
                        and "room_type_preference" not in changed_fields
                    ):
                        logger.info(
                            "Capacity changed; dropping the carried-over room_type_preference=%r.",
                            state.get("room_type_preference"),
                        )
                        updates["room_type_preference"] = None
                    updates["booking_stage"] = "collecting"
                    if updates.get("next_action") != "need_subgroups":
                        updates["next_action"] = "search"
                    updates["ambiguous_guests"] = None
                    updates["upsell_shown"] = False

                    # If room_requirements existed from a previous turn, but user changed guests/rooms without providing new requirements
                    if getattr(extracted, "room_requirements", None) is None:
                        current_num_rooms = state.get("num_rooms") or 0
                        current_adults = updates.get("adults", state.get("adults", 0))
                        if current_adults and current_num_rooms and current_adults < current_num_rooms:
                            logger.info(f"Guest count ({current_adults}) is less than previous room count ({current_num_rooms}). Resetting room_requirements & num_rooms.")
                            updates["room_requirements"] = None
                            updates["num_rooms"] = None
                else:
                    # Only dates changed! Keep selected rooms.
                    logger.info("Only date parameters changed. Keeping selected rooms and showing quote.")
                    if state.get("selected_rooms"):
                        updates["booking_stage"] = "rooms_shown"
                        updates["next_action"] = "show_quote"
                    else:
                        updates["booking_stage"] = "collecting"
                        updates["next_action"] = "search"

            if updates.get("intent") not in ("PRICE_MATCH", "CONFIRM_BOOKING", "CANCEL_BOOKING") and (updates.get("intent") in ("BOOKING_SUMMARY", "UNKNOWN", "GREETING") or resolved_intent in ("BOOKING_SUMMARY", "UNKNOWN", "GREETING")):
                logger.info(f"Changed intent from {resolved_intent} to BOOK_ROOM because core booking parameters were modified.")
                updates["intent"] = "BOOK_ROOM"
        else:
            if ignore_changes and changed_fields:
                logger.info(f"Ignoring booking query changes {changed_fields} because user uploaded a payment receipt.")
            updates["updated_fields"] = []
    else:
        updates["updated_fields"] = []
        # For non-booking intents, still extract photo_tag_preference if present
        if not is_cancellation:
            extracted = result.extracted_data
            if getattr(extracted, "photo_tag_preference", None) is not None:
                updates["photo_tag_preference"] = extracted.photo_tag_preference
            # Keep room type + nights context for pricing summaries
            if resolved_intent == "BOOKING_SUMMARY":
                if extracted.room_type_preference is not None:
                    updates["room_type_preference"] = extracted.room_type_preference
                if extracted.num_rooms is not None:
                    updates["num_rooms"] = extracted.num_rooms
                    updates["pricing_query_num_rooms"] = extracted.num_rooms
                if not updates.get("pricing_query_nights"):
                    from app.utils.intent_overrides import extract_nights_from_message
                    nights = extract_nights_from_message(state.get("user_message", ""))
                    if nights:
                        updates["pricing_query_nights"] = nights
                if not updates.get("pricing_query_num_rooms"):
                    from app.utils.intent_overrides import extract_num_rooms_from_message
                    num_rooms = extract_num_rooms_from_message(state.get("user_message", ""))
                    if num_rooms:
                        updates["pricing_query_num_rooms"] = num_rooms
                        updates["num_rooms"] = num_rooms
    if updates.get("wants_photos") and not state.get("selected_rooms") and not state.get("allocation_options"):
        updates["force_new_search"] = True
        updates["next_action"] = "search"
        updates["booking_stage"] = "collecting"
        updates["allocation_options"] = []
        updates["selected_rooms"] = []

    # ── The guest's own clock ──
    # The arrival or departure time they volunteered, if they volunteered one.
    # Deliberately read from `user_message` here and then *persisted* onto the
    # session: `user_message` is turn-scoped, so a guest who says "I'll reach at
    # 7am" and picks a room two turns later must still be priced for the 7am.
    #
    # Never asked for. If a guest never states a time, no early/late fee is ever
    # quoted — which is the whole point, because a misread time is a wrong
    # number on someone's bill.
    #
    # Written after every extraction branch and before the funnel, so it lands
    # in `updates` like any other field and survives `validate_updates` (these
    # two keys are not governed fields, so the contract leaves them alone).
    from app.utils.guest_time import extract_requested_times

    _guest_times = extract_requested_times(state.get("user_message"))
    if _guest_times.get("arrival"):
        updates["requested_arrival_time"] = _guest_times["arrival"]
    if _guest_times.get("departure"):
        updates["requested_departure_time"] = _guest_times["departure"]

    # ── THE FUNNEL ──
    # Every extraction branch above writes into `updates`, and this line is the
    # only place they all pass through. A value that cannot be what it claims —
    # the year in "21.10.2026" as a headcount, a guest's whole sentence as a
    # room type — is nulled here, once, instead of at each of the seventeen
    # call sites that would each have to remember. See
    # app/models/extraction_contract.py.
    #
    # Validated over the merged result, not over `updates` alone. A value that
    # is wrong does not stop being wrong by being a turn old: in chat
    # 919748533727 the poisoned room_type_preference="ac" was extracted once and
    # then carried silently through later turns, which never mentioned a room
    # type at all — the branch that wrote it never ran again, so an
    # updates-only check would never have seen it. re-reading it from state is
    # what makes the next turn clean rather than the turn after next.
    merged: dict[str, Any] = {**state, **updates}
    merged, rejected_fields = validate_updates(state, merged)

    # A field we just refused must not be announced. The update-confirmation
    # prefix names each changed field and prints the value from state, so
    # leaving `adults` in this list while the contract has nulled it produced
    # "I've updated your details to number of adults to **None**" — the guest
    # told about a change that did not happen. Dropping it here lets the
    # missing-field path ask for it properly instead.
    if rejected_fields:
        declared = merged.get("updated_fields")
        if isinstance(declared, list):
            merged["updated_fields"] = [
                f for f in declared if f not in rejected_fields
            ]
        queued = merged.get("booking_queue")
        if isinstance(queued, list):
            # Rebuilt rather than edited in place: `merged` shares its nested
            # lists with `state`, and this function must not mutate its input.
            merged["booking_queue"] = [
                {
                    **entry,
                    "updated_fields": [
                        f for f in entry["updated_fields"] if f not in rejected_fields
                    ],
                }
                if isinstance(entry, dict)
                and isinstance(entry.get("updated_fields"), list)
                else entry
                for entry in queued
            ]

    # Authoritative lifecycle invariant (Sir's spec):
    # LLM understanding must NEVER downgrade or override an active lifecycle stage.
    #
    # `abandoned_draft` is the one exception, and it has to be here rather than in
    # the caller. It is set by `clear_in_progress_booking_fields`, which runs when
    # the guest explicitly asks to start over — and that override has just written
    # `lifecycle_stage: "lead"` into `merged`. Without this guard the invariant
    # read `state` (which context_loader had already set to the booking's own
    # stage) and put it straight back, so "start fresh" reset the draft and then
    # the very same turn routed to `stage_payment_check` as if nothing had
    # happened. A reset is not the LLM downgrading a stage; it is the guest
    # ending it, and the booking row itself is deliberately left alone.
    current_lifecycle = state.get("lifecycle_stage")
    if (
        current_lifecycle in ("payment_check", "confirmed", "in_house", "checked_out")
        and not merged.get("abandoned_draft")
    ):
        merged["lifecycle_stage"] = current_lifecycle
        merged["booking_stage"] = current_lifecycle

    return cast(ConversationState, merged)

