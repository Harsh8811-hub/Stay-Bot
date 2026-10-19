"""LangGraph conditional routing helpers."""

from app.services.repetition_guard import REPEAT_ESCALATION_THRESHOLD
from app.utils.intent_overrides import resolve_negotiation_flag
from app.workflows.state import ConversationState


def route_after_gate(state: ConversationState) -> str:
    """Route after feature gate: blocked → save, allowed → intent router."""
    if state.get("next_action") == "blocked":
        return "blocked"
    return "allowed"


def route_by_intent(state: ConversationState) -> str:
    """Route to the appropriate handler based on detected intent."""
    intent = state.get("intent", "UNKNOWN")
    booking_stage = state.get("booking_stage", "new")

    escalation_type = state.get("escalation_issue_type", "NORMAL")
    if escalation_type in ("MEDICAL", "SAFETY", "SECURITY", "LEGAL"):
        return "escalation"

    # ── Canonical Stage Enforcements (Sir's Spec) ─────────────────────────
    # Booking stage controls the bot's behavior, NOT the guest's latest message.
    lifecycle_stage = state.get("lifecycle_stage") or state.get("booking_stage")
    user_msg_clean = (state.get("user_message") or "").strip().lower()

    # Detect payment proof / payment claims in Stage 1 to transition to Stage 1b
    is_payment_claim = (
        state.get("payment_proof_submitted")
        or any(p in user_msg_clean for p in ("maine payment kar diya", "payment kar diya", "payment done", "already paid", "paid rs", "transfer kar diya", "upi kar diya"))
    )
    if is_payment_claim and lifecycle_stage in ("lead", "payment_check", "new", "collecting", "rooms_shown", "quote_shown", "awaiting_payment"):
        return "stage_payment_check"

    if lifecycle_stage == "payment_check":
        return "stage_payment_check"

    if lifecycle_stage == "confirmed":
        return "stage_confirmed"

    if lifecycle_stage == "in_house":
        return "stage_in_house"

    if lifecycle_stage == "checked_out":
        return "stage_checked_out"

    # If there's a pending lifecycle action awaiting confirmation, always
    # route back to lifecycle_handler regardless of intent.
    if state.get("lifecycle_action_pending"):
        return "lifecycle"

    if state.get("next_action") in ("lead_created", "need_custom_rooms", "opted_out"):
        return "greeting"

    if booking_stage == "cancellation_pending" and intent != "FAQ":
        return "cancel_booking"

    if intent == "HUMAN_REQUEST":
        return "escalation"

    if intent == "ESCALATE":
        return "escalation"

    if state.get("escalation_required"):
        return "escalation"

    if state.get("escalation_failure_count", 0) >= 3:
        return "escalation"

    if state.get("repeat_response_count", 0) >= REPEAT_ESCALATION_THRESHOLD:
        return "escalation"

    if intent == "GREETING":
        return "greeting"

    if intent == "QUEUE_MODIFICATION":
        return "greeting" # We just route it to generate_response which is what "greeting" routes to in graph.py

    if intent == "SHOW_CHANGE_MENU":
        return "greeting"

    if intent == "COMPLAINT":
        return "escalation"

    if intent == "HOTEL_TASK":
        return "hotel_task"

    if intent == "ROOM_SERVICE_ORDER":
        return "room_service"

    if intent == "AMENITY_QUERY":
        return "amenity_query"

    if intent == "THANK_YOU":
        return "thank_you"

    if intent == "FAQ":
        return "faq"

    if intent in ("SELECTED_ROOM_DETAILS", "BOOKING_SUMMARY", "ROOM_FEATURE_LOOKUP"):
        return "room_query"

    if intent in ("SHOW_PRICES", "SHOW_LOCATION", "SHOW_AMENITIES", "CHECK_EXISTING_BOOKING", "SOMETHING_ELSE", "SHOW_CHECKIN_POLICY"):
        return "menu"

    if intent == "PAYMENT_STATUS":
        return "payment"

    if intent in ("BOOK_ROOM", "CHECK_AVAILABILITY", "SELECT_ROOM"):
        return "booking_flow"
        
    if intent == "MODIFY_BOOKING":
        return "handle_modification"

    if intent == "CANCEL_BOOKING":
        return "cancel_booking"

    if intent == "CONFIRM_BOOKING":
        # Shared with `validator`, through one function so the two cannot drift.
        # That function prefers this turn's LLM verdict (CONFIRM/CHANGE) and
        # falls back to the word-bounded regex when the model abstained.
        is_negotiation_or_change = resolve_negotiation_flag(
            state, state.get("user_message") or ""
        )
        if is_negotiation_or_change:
            return "booking_flow"
        if (booking_stage == "quote_shown" or state.get("total_price")) and state.get("guest_name"):
            return "payment"
        return "booking_flow"

    if intent == "PRICE_MATCH":
        return "price_match"

    if intent == "LIFECYCLE_RESET":
        return "lifecycle"

    if intent == "SILENT_DOCUMENT_UPLOAD":
        return "silent_upload"

    if intent == "BANK_DETAILS":
        return "bank_details"

    return "unknown"


def route_by_validation(state: ConversationState) -> str:
    """Route based on validation results."""
    next_action = state.get("next_action", "")

    if next_action in ("need_dates", "need_guests", "invalid_dates", "invalid_room_selection"):
        return "need_info"

    if next_action == "search_rooms":
        return "search"

    if next_action == "show_quote":
        return "quote"

    if next_action == "ready_for_payment":
        return "payment"

    return "need_info"


def route_after_search(state: ConversationState) -> str:
    """Route after room search."""
    next_action = state.get("next_action", "")

    if next_action == "no_rooms":
        return "no_rooms"

    if next_action == "rooms_shown":
        return "done"

    if next_action == "show_quote":
        return "quote"

    return "done"


def route_after_payment(state: ConversationState) -> str:
    """Route after payment handling."""
    next_action = state.get("next_action", "")

    if next_action == "payment_verified":
        return "confirm"

    return "done"
