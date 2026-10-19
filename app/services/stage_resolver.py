"""
Deterministic Booking Stage Resolver — StayChat Architecture.

CORE PRINCIPLE:
The bot's behaviour is decided by the guest's booking stage, not by the last message.
The stage resolver MUST run before the LLM.
The LLM MUST NEVER decide or override the booking stage.

Allowed Stages:
- lead (Stage 1): Inquiry / sales
- payment_check (Stage 1b): Payment under verification (reassured & pending)
- confirmed (Stage 2): Pre-arrival guidance (ZERO SELLING)
- in_house (Stage 3): In-stay concierge
- checked_out (Stage 4): Post-stay feedback & retention
- cancelled: Cancelled booking

Timezone: Property timezone (Asia/Kolkata for Hotel Sahu & Sanidhyam).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time
from enum import Enum
from typing import Any
from zoneinfo import ZoneInfo

from app.db.mongodb import bookings_collection
from app.repositories import booking_repository

logger = logging.getLogger(__name__)

KOLKATA_TZ = ZoneInfo("Asia/Kolkata")
DEFAULT_CHECK_IN_TIME = time(12, 0)   # 12:00 PM
DEFAULT_CHECK_OUT_TIME = time(11, 0)  # 11:00 AM


class BookingStage(str, Enum):
    LEAD = "lead"                    # Stage 1
    PAYMENT_CHECK = "payment_check"  # Stage 1b
    CONFIRMED = "confirmed"          # Stage 2
    IN_HOUSE = "in_house"            # Stage 3
    CHECKED_OUT = "checked_out"      # Stage 4
    CANCELLED = "cancelled"


STAGE_ALLOWED_TOOLS: dict[str, list[str]] = {
    BookingStage.LEAD.value: [
        "search_rooms",
        "show_room_options",
        "create_payment_link",
        "get_faq",
        "check_promotions",
    ],
    BookingStage.PAYMENT_CHECK.value: [
        "check_payment_status",
        "raise_owner_task",
        "get_faq",
    ],
    BookingStage.CONFIRMED.value: [
        "get_booking_details",
        "get_hotel_directions",
        "raise_owner_task",
        "get_faq",
    ],
    BookingStage.IN_HOUSE.value: [
        "get_booking_details",
        "raise_service_task",
        "quote_stay_extension",
        "get_local_guide",
        "get_hotel_amenities",
        "get_faq",
    ],
    BookingStage.CHECKED_OUT.value: [
        "submit_feedback",
        "get_review_link",
        "raise_owner_task",
        "get_faq",
        "start_new_inquiry",
    ],
    BookingStage.CANCELLED.value: [
        "get_faq",
        "start_new_inquiry",
    ],
}


@dataclass
class GuestStageContext:
    """Deterministic context computed before LLM execution."""
    stage: BookingStage
    stage_value: str
    active_booking: dict[str, Any] | None = None
    active_booking_id: str | None = None
    guest_name: str | None = None
    check_in_date: str | None = None
    check_out_date: str | None = None
    room_number: str | None = None
    amount_paid: float = 0.0
    total_amount: float = 0.0
    allowed_tools: list[str] = field(default_factory=list)
    is_separate_booking_inquiry: bool = False
    fail_safe_triggered: bool = False
    # True only when the stage came from the CRM's staff-confirmation signal rather
    # than from a booking row of our own. `understand` reads this: its ghost guard
    # demotes any "confirmed" stage that has no `booking_id`, which is every
    # CRM-confirmed guest, so without it the promotion below is undone one node later.
    crm_confirmed: bool = False
    # The CRM's own name for this guest's room. Only ever set alongside
    # `crm_confirmed` — a guest with a booking row of ours reads the room off that row.
    crm_room_type: str | None = None


def _parse_iso_date(val: Any) -> date | None:
    if not val:
        return None
    if isinstance(val, date):
        return val
    if isinstance(val, datetime):
        return val.date()
    val_str = str(val).strip()
    try:
        return datetime.strptime(val_str[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def compute_booking_stage_from_record(
    booking: dict[str, Any],
    now_kolkata: datetime,
) -> tuple[BookingStage, bool]:
    """
    Computes booking stage from a booking document and current clock in Asia/Kolkata.
    Returns (BookingStage, fail_safe_triggered).
    """
    raw_status = str(booking.get("booking_status") or "").lower().strip()
    raw_stage = str(booking.get("stage") or "").lower().strip()

    if raw_stage == BookingStage.PAYMENT_CHECK.value or booking.get("payment_check_pending"):
        return BookingStage.PAYMENT_CHECK, False

    if raw_status == "cancelled" or raw_stage == BookingStage.CANCELLED.value:
        return BookingStage.CANCELLED, False

    if raw_status in ("pending_payment", "payment_pending", "draft", "new"):
        if booking.get("payment_proof_submitted") or booking.get("claimed_paid"):
            return BookingStage.PAYMENT_CHECK, False
        return BookingStage.LEAD, False

    if booking.get("checked_out_at") or raw_status == "checked_out" or raw_stage == BookingStage.CHECKED_OUT.value:
        return BookingStage.CHECKED_OUT, False

    ci_date = _parse_iso_date(booking.get("check_in") or booking.get("check_in_date"))
    co_date = _parse_iso_date(booking.get("check_out") or booking.get("check_out_date"))

    if booking.get("checked_in_at") or raw_status == "checked_in" or raw_stage == BookingStage.IN_HOUSE.value:
        if co_date:
            check_out_dt = datetime.combine(co_date, DEFAULT_CHECK_OUT_TIME, tzinfo=KOLKATA_TZ)
            if now_kolkata >= check_out_dt:
                return BookingStage.CHECKED_OUT, False
        return BookingStage.IN_HOUSE, False

    if not ci_date or not co_date:
        if raw_status != "confirmed" and raw_stage != BookingStage.CONFIRMED.value:
            return BookingStage.LEAD, False
        logger.warning(
            "STAGE_RESOLVER_FAILSAFE | booking_id=%s missing valid dates; failing safe to confirmed",
            booking.get("booking_id"),
        )
        return BookingStage.CONFIRMED, True

    check_in_dt = datetime.combine(ci_date, DEFAULT_CHECK_IN_TIME, tzinfo=KOLKATA_TZ)
    check_out_dt = datetime.combine(co_date, DEFAULT_CHECK_OUT_TIME, tzinfo=KOLKATA_TZ)

    if now_kolkata < check_in_dt:
        return BookingStage.CONFIRMED, False
    elif check_in_dt <= now_kolkata < check_out_dt:
        return BookingStage.IN_HOUSE, False
    else:
        return BookingStage.CHECKED_OUT, False


def _same_stay_check_out(dismissed_check_out: str | None, found_check_out: str | None) -> bool:
    """Whether a dismissed confirmation names the same stay as the one we just found.

    Compared as dates when both parse, because the CRM and the bot disagree about
    the spelling ("2027-01-18" vs "2027-01-18T00:00:00") and a string compare would
    read those as two different stays and quietly un-dismiss. Falls back to a
    trimmed string compare so a shape neither parser knows still behaves sanely —
    equal spellings still match, which is the conservative direction: it keeps the
    dismissal the guest asked for rather than silently reviving a sale.
    """
    if not dismissed_check_out:
        return False
    dismissed = _parse_iso_date(dismissed_check_out)
    found = _parse_iso_date(found_check_out)
    if dismissed is not None and found is not None:
        return dismissed == found
    return str(dismissed_check_out).strip() == str(found_check_out).strip()


async def _crm_confirmed_stage_context(
    hotel_id: int | str,
    customer_phone: str,
    now_kolkata: datetime,
    dismissed_check_out: str | None = None,
) -> GuestStageContext | None:
    """The guest's stage as the CRM's staff confirmation implies it, if it should decide.

    A stay confirmed by staff in the Control Centre never lands in the bot's own
    `bookings` collection. `resolve_guest_stage` therefore reads "no bookings" and
    demotes the guest to a lead — the bot then sells, and asks for money, from
    someone who has already paid and been confirmed.

    Pranav's rule, 2026-10-13: a confirmed stay whose check-out date has NOT passed
    keeps the guest in the confirmed stage (the FAQ/concierge path, zero selling);
    once that date has passed the confirmation is spent and the guest is treated
    normally. No check-out date we can parse means we cannot say the stay has not
    ended, which is the same answer as "passed".

    Returns `None` — "no opinion" — for every one of those cases, so the caller
    falls back to the behaviour it already had. This can only ever add a stage,
    never remove one.

    `dismissed_check_out` is the one input that lets it *remove* a stage, and it
    is the guest's own doing: an explicit "start fresh" while a confirmation
    stands writes that stay's check-out onto the session, and this function then
    reports no opinion about that same stay for as long as the session lives.
    Scoped to the stay, not to the guest — a confirmation naming a different
    check-out is a different booking and still applies.
    """
    from app.config.settings import get_settings

    if not getattr(get_settings(), "CRM_CONFIRMED_STAGE_ENABLED", False):
        return None

    from app.repositories.crm_confirmation_repository import lookup_crm_confirmation

    found = await lookup_crm_confirmation(hotel_id, customer_phone)
    if found is None or not found.confirmed:
        return None

    if _same_stay_check_out(dismissed_check_out, found.check_out):
        logger.info(
            "STAGE_CRM_CONFIRMATION_DISMISSED | hotel=%s phone=%s check_out=%s — guest "
            "asked to start fresh, so this confirmation does not set the stage",
            str(hotel_id),
            customer_phone[-4:] if len(customer_phone) >= 4 else customer_phone,
            found.check_out,
        )
        return None

    check_out = _parse_iso_date(found.check_out)
    if check_out is None or now_kolkata.date() > check_out:
        return None

    logger.info(
        "STAGE_RESOLVED_FROM_CRM | hotel=%s phone=%s check_out=%s — confirmed stay, "
        "no booking row of ours",
        str(hotel_id),
        customer_phone[-4:] if len(customer_phone) >= 4 else customer_phone,
        check_out.isoformat(),
    )
    return GuestStageContext(
        stage=BookingStage.CONFIRMED,
        stage_value=BookingStage.CONFIRMED.value,
        check_in_date=found.check_in or None,
        check_out_date=found.check_out or None,
        allowed_tools=STAGE_ALLOWED_TOOLS.get(BookingStage.CONFIRMED.value, []),
        crm_confirmed=True,
        crm_room_type=found.room_type or None,
    )


async def resolve_guest_stage(
    hotel_id: int | str,
    customer_phone: str,
    *,
    now_dt: datetime | None = None,
    explicit_stage: str | None = None,
    session_stage: str | None = None,
    dismissed_crm_check_out: str | None = None,
    dismissed_booking_id: str | None = None,
) -> GuestStageContext:
    """
    Resolve the guest's booking stage before LLM execution.
    Multi-tenant safe: strictly scoped by hotel_id and normalized customer phone.
    """
    now_kolkata = now_dt.astimezone(KOLKATA_TZ) if now_dt else datetime.now(KOLKATA_TZ)
    str_hid = str(hotel_id)

    if explicit_stage:
        try:
            st_enum = BookingStage(explicit_stage)
            return GuestStageContext(
                stage=st_enum,
                stage_value=st_enum.value,
                allowed_tools=STAGE_ALLOWED_TOOLS.get(st_enum.value, []),
            )
        except Exception:
            pass

    if not customer_phone:
        return GuestStageContext(
            stage=BookingStage.LEAD,
            stage_value=BookingStage.LEAD.value,
            allowed_tools=STAGE_ALLOWED_TOOLS[BookingStage.LEAD.value],
        )

    # This runs on EVERY turn, for EVERY guest, from context_loader — the entry
    # node. An unhandled failure here does not degrade one feature, it returns
    # no reply at all. So a lookup that fails is treated exactly like a lookup
    # that found nothing: the guest is a lead, and the turn proceeds. The
    # alternative — letting it propagate — turns a Mongo blip into an outage of
    # the whole conversation, including for guests who have no booking at all.
    try:
        bookings = await booking_repository.get_recent_bookings(
            hotel_id=str_hid,
            customer_phone=customer_phone,
            limit=5,
        )
    except Exception:
        logger.exception(
            "STAGE_LOOKUP_FAILED | hotel=%s phone=%s — degrading to lead",
            str_hid,
            customer_phone[-4:] if len(customer_phone) >= 4 else customer_phone,
        )
        bookings = []

    # A guest who asked to start fresh has dismissed the stay they were on. Drop
    # that row here, at the single point where the rows are read, so everything
    # downstream — the confirmed/in-house branches, the payment_check preservation,
    # the CRM fallback, the candidates loop — sees the guest as they now are
    # instead of each having to remember the dismissal. If it was their only row
    # the list empties and the `if not bookings:` branch below answers.
    if dismissed_booking_id:
        kept = [
            b for b in bookings
            if str(b.get("booking_id") or "") != str(dismissed_booking_id)
        ]
        if len(kept) != len(bookings):
            logger.info(
                "STAGE_BOOKING_ROW_DISMISSED | hotel=%s phone=%s dismissed=%s rows=%d→%d — "
                "guest asked to start fresh, so this booking no longer sets the stage",
                str_hid,
                customer_phone[-4:] if len(customer_phone) >= 4 else customer_phone,
                dismissed_booking_id,
                len(bookings),
                len(kept),
            )
        bookings = kept

    if not bookings:
        # If there are no bookings in DB, a guest CANNOT have a confirmed, in-house,
        # or checked-out stay. Preserving a ghost confirmed stage causes the bot to
        # send 'BK-CONFIRMED' slips to unconfirmed guests. Only pending payment checks
        # are preserved without a finalized booking record.
        if session_stage == BookingStage.PAYMENT_CHECK.value:
            return GuestStageContext(
                stage=BookingStage.PAYMENT_CHECK,
                stage_value=BookingStage.PAYMENT_CHECK.value,
                allowed_tools=STAGE_ALLOWED_TOOLS.get(BookingStage.PAYMENT_CHECK.value, []),
            )

        # A stay confirmed by staff in the Control Centre also has no row here, so it
        # lands in this branch too. Ask the CRM before demoting the guest to a lead —
        # but only after the payment_check case above, which must keep its own answer.
        crm_ctx = await _crm_confirmed_stage_context(
            hotel_id, customer_phone, now_kolkata, dismissed_crm_check_out
        )
        if crm_ctx is not None:
            return crm_ctx

        return GuestStageContext(
            stage=BookingStage.LEAD,
            stage_value=BookingStage.LEAD.value,
            allowed_tools=STAGE_ALLOWED_TOOLS[BookingStage.LEAD.value],
        )

    candidates: list[tuple[int, BookingStage, dict[str, Any], bool]] = []

    for b in bookings:
        st, fail_safe = compute_booking_stage_from_record(b, now_kolkata)
        priority = 99
        if st == BookingStage.IN_HOUSE:
            priority = 1
        elif st == BookingStage.CONFIRMED:
            priority = 2
        elif st == BookingStage.PAYMENT_CHECK:
            priority = 3
        elif st == BookingStage.CHECKED_OUT:
            priority = 4
        elif st == BookingStage.LEAD:
            priority = 5
        elif st == BookingStage.CANCELLED:
            priority = 6

        candidates.append((priority, st, b, fail_safe))

    candidates.sort(key=lambda x: x[0])
    chosen_priority, chosen_stage, chosen_booking, fail_safe_used = candidates[0]

    if chosen_stage == BookingStage.CANCELLED and len(candidates) == 1:
        chosen_stage = BookingStage.LEAD

    b_id = chosen_booking.get("booking_id")
    g_name = chosen_booking.get("guest_name") or chosen_booking.get("customer_name")
    ci = chosen_booking.get("check_in") or chosen_booking.get("check_in_date")
    co = chosen_booking.get("check_out") or chosen_booking.get("check_out_date")
    room_num = chosen_booking.get("room_number")
    paid = float(chosen_booking.get("amount_paid") or 0.0)
    total = float(chosen_booking.get("total_price") or chosen_booking.get("amount") or 0.0)

    logger.info(
        "STAGE_RESOLVED | hotel=%s phone=%s stage=%s booking_id=%s fail_safe=%s",
        str_hid,
        customer_phone[-4:] if len(customer_phone) >= 4 else customer_phone,
        chosen_stage.value,
        b_id,
        fail_safe_used,
    )

    return GuestStageContext(
        stage=chosen_stage,
        stage_value=chosen_stage.value,
        active_booking=chosen_booking,
        active_booking_id=b_id,
        guest_name=g_name,
        check_in_date=str(ci) if ci else None,
        check_out_date=str(co) if co else None,
        room_number=room_num,
        amount_paid=paid,
        total_amount=total,
        allowed_tools=STAGE_ALLOWED_TOOLS.get(chosen_stage.value, []),
        fail_safe_triggered=fail_safe_used,
    )


async def set_booking_stage_for_test(
    booking_id: str,
    stage: str,
    *,
    hotel_id: int | str = "111111",
    customer_phone: str = "919999999999",
    guest_name: str = "Test Guest",
    check_in: str | None = None,
    check_out: str | None = None,
    room_name: str = "Deluxe Room King Bed",
    room_number: str | None = None,
    amount_paid: float = 3000.0,
    total_amount: float = 3000.0,
    payment_proof_submitted: bool = False,
    claimed_paid: bool = False,
) -> dict[str, Any]:
    """
    Test hook: Directly upserts a booking record into MongoDB with specified stage and dates.
    Allows automated acceptance tests to deterministically verify S51–S64.
    """
    coll = bookings_collection()
    str_hid = str(hotel_id)
    now = datetime.now(KOLKATA_TZ)

    status_map = {
        BookingStage.LEAD.value: "pending_payment",
        BookingStage.PAYMENT_CHECK.value: "pending_payment",
        BookingStage.CONFIRMED.value: "confirmed",
        BookingStage.IN_HOUSE.value: "confirmed",
        BookingStage.CHECKED_OUT.value: "confirmed",
        BookingStage.CANCELLED.value: "cancelled",
    }
    booking_status = status_map.get(stage, "confirmed")

    doc = {
        "booking_id": booking_id,
        "hotel_id": str_hid,
        "customer_phone": customer_phone,
        "customer_identifier": customer_phone,
        "guest_name": guest_name,
        "stage": stage,
        "booking_status": booking_status,
        "check_in": check_in,
        "check_out": check_out,
        "check_in_date": check_in,
        "check_out_date": check_out,
        "room_name": room_name,
        "room_type": room_name,
        "room_number": room_number,
        "amount_paid": amount_paid,
        "total_price": total_amount,
        "payment_status": "paid" if amount_paid >= total_amount else "partial",
        "payment_proof_submitted": payment_proof_submitted or (stage == BookingStage.PAYMENT_CHECK.value),
        "claimed_paid": claimed_paid or (stage == BookingStage.PAYMENT_CHECK.value),
        "created_at": now,
        "updated_at": now,
    }

    if stage == BookingStage.IN_HOUSE.value:
        doc["checked_in_at"] = now
    elif stage == BookingStage.CHECKED_OUT.value:
        doc["checked_in_at"] = now
        doc["checked_out_at"] = now

    await coll.replace_one({"booking_id": booking_id}, doc, upsert=True)
    logger.info("TEST_HOOK_BOOKING_UPSERTED | id=%s stage=%s hotel=%s", booking_id, stage, str_hid)
    return doc
