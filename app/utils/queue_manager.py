from datetime import datetime, timezone, timedelta
from app.workflows.state import ConversationState

import logging
logger = logging.getLogger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))

def _now_iso() -> str:
    """Current time in IST, ISO 8601 format."""
    return datetime.now(_IST).strftime("%Y-%m-%dT%H:%M:%S+05:30")


def advance_booking_queue(state: ConversationState, terminal_status: str) -> dict:
    """
    Called when the active booking reaches a terminal state
    (confirmed, cancelled, failed, abandoned, skipped, expired).

    Stamps timestamps on the current booking, advances the index, and
    loads the next booking's fields into state.

    Per-booking audit trail fields:
        created_at   — set once at queue creation
        last_updated — updated on every status transition
        completed_at — set when a terminal status is reached
    """
    queue = list(state.get("booking_queue", []) or [])
    active_idx = state.get("active_booking_index", 0)
    now = _now_iso()

    completed = list(state.get("completed_bookings", []) or [])
    failed    = list(state.get("failed_bookings",    []) or [])

    # ── Stamp the current booking ──────────────────────────────
    if len(queue) > active_idx:
        current = dict(queue[active_idx])           # shallow copy — don't mutate in-place
        current["queue_status"]   = terminal_status
        current["terminal_status"] = terminal_status
        current["last_updated"]   = now
        current["completed_at"]   = now             # terminal = done, regardless of outcome
        queue[active_idx] = current

        from app.observability.metrics import metrics
        metrics.incr(f"queue.{terminal_status}")

        queue_id = state.get("queue_id", "UNKNOWN_QUEUE")
        logger.info(
            f"QUEUE_{terminal_status.upper()}: "
            f"Booking {active_idx + 1} of {len(queue)} in {queue_id} | "
            f"guest={current.get('guest_name')} | completed_at={now}"
        )

        if terminal_status == "confirmed":
            completed.append(current)
        else:
            failed.append(current)

    updates = {
        "completed_bookings": completed,
        "failed_bookings":    failed,
        "booking_queue":      queue,
    }

    # ── Advance to next booking ────────────────────────────────
    if active_idx + 1 < len(queue):
        next_idx     = active_idx + 1
        next_booking = dict(queue[next_idx])
        next_booking["queue_status"] = "processing"
        next_booking["last_updated"] = now
        # completed_at stays None until it reaches a terminal state
        queue[next_idx] = next_booking

        updates["active_booking_index"] = next_idx
        updates["booking_queue"]        = queue

        from app.observability.metrics import metrics
        metrics.incr("queue.advanced")
        queue_id = state.get("queue_id", "UNKNOWN_QUEUE")
        logger.info(
            f"QUEUE_ADVANCED: index {active_idx} → {next_idx} in {queue_id} | "
            f"next_guest={next_booking.get('guest_name')}"
        )

        # Load next booking's fields into active session
        fields_to_load = [
            "adults", "children", "num_rooms",
            "check_in", "check_out",
            "room_type_preference",
            "guest_name", "guest_phone", "guest_email",
        ]
        for f in fields_to_load:
            updates[f] = next_booking.get(f)

        # Reset session-level transient state
        updates.update({
            "booking_stage":        "collecting",
            "next_action":          "search",
            "selected_rooms":       [],
            "available_rooms":      [],
            "allocation_options":   [],
            "price_breakdown":      None,
            "total_price":          None,
            "payment_link":         None,
            "payment_status":       None,
            "booking_id":           None,
            "intent":               "BOOK_ROOM",
            "room_type_not_available": False,
        })

        # Build transition message for the guest
        from app.prompts import templates
        name  = next_booking.get("guest_name", "the next guest")
        ci    = next_booking.get("check_in")
        co    = next_booking.get("check_out")
        dates = f"({ci} to {co})" if ci else ""

        if terminal_status == "confirmed":
            status_msg = "The previous booking is confirmed."
        elif terminal_status == "cancelled":
            status_msg = "The previous booking was cancelled."
        else:
            status_msg = "The previous booking could not be completed."

        msg = f"{status_msg}\n\nNow, let's continue with the booking for **{name}** {dates}."
        updates["bot_response_prepend"] = templates.text_card(msg)
        updates["queue_advanced"]       = True

    else:
        # ── All bookings exhausted ─────────────────────────────
        queue_id = state.get("queue_id", "UNKNOWN_QUEUE")
        logger.info(f"QUEUE_FINISHED: All {len(queue)} bookings done in {queue_id}")
        updates.update({
            "booking_queue":       [],
            "active_booking_index": 0,
            "queue_id":            None,
        })

    return updates
