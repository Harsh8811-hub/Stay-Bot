import logging
from datetime import datetime, timezone
from typing import Any

from app.config.settings import get_settings
from app.db.mongodb import bookings_collection
from app.models.activity_schemas import ActivityLog
from app.repositories import guest_repository
from app.repositories.activity_repository import log_activity

logger = logging.getLogger(__name__)

# Active stay / hold statuses (excludes cancelled)
ACTIVE_BOOKING_STATUSES = [
    "confirmed",
    "pending_payment",
    "partial",
    "checked_in",
]

# Same ₹10 the payment paths use (`app/services/vision/validator.py:31`,
# `app/workflows/nodes/payment/status.py:20`), so a rounding-sized gap between
# what was recorded and the total does not flip a settled booking to "partial".
_AMOUNT_TOLERANCE = 10.0


async def _drop_legacy_bookings_ttl_index(coll) -> None:
    """
    Remove the TTL index that used to be created here, if it exists.

    It was:

        coll.create_index(
            "hold_expires_at",
            expireAfterSeconds=300,
            partialFilterExpression={"booking_status": "pending_payment"},
        )

    and it deleted live bookings. `payment/link.py` stamps every hold with
    `hold_expires_at = now + 30m` while leaving `booking_status` at
    `pending_payment`, so the row fell inside the partial filter from the moment
    it was written, and MongoDB erased it 300s after that timestamp — roughly 35
    minutes after the guest was shown its booking id. Only a row promoted to
    `confirmed` in the meantime escaped, because it left the partial filter.

    Measured on production: of the 34 bookings created in October 2026, 25 were
    deleted and the 9 survivors were exactly the ones that had reached
    `confirmed`. The downstream damage is the reported symptom. The guest keeps
    the booking id the bot printed, pays by screenshot some time later, the
    receipt validator looks for a booking to ground the price against, finds
    none, and falls through to its `expected_amount <= 0` arm — which tells the
    guest "our front desk team has been notified" instead of confirming. 28 of
    the 34 payment escalations in October carried `Total: ₹0, expected ₹0`.

    Dropped rather than merely not re-created: the index already exists in
    production, and `create_index` only ever adds. Best-effort, and last, so a
    failure here can never stop the other indexes from being ensured.
    """
    try:
        for name, spec in (await coll.index_information()).items():
            is_hold_ttl = (
                list(spec.get("key") or []) == [("hold_expires_at", 1)]
                and spec.get("expireAfterSeconds") is not None
            )
            if is_hold_ttl:
                await coll.drop_index(name)
                logger.warning(
                    "Dropped destructive bookings TTL index %r "
                    "(expireAfterSeconds=%s) — it was deleting live bookings",
                    name,
                    spec.get("expireAfterSeconds"),
                )
    except Exception as e:  # noqa: BLE001 — never block startup on this
        logger.warning("Could not drop legacy bookings TTL index: %s", e)


async def ensure_indexes() -> None:
    """Create indexes. Idempotent."""
    coll = bookings_collection()
    await coll.create_index("booking_id", unique=True)
    await coll.create_index("hotel_id")
    await coll.create_index("guest_id")
    await coll.create_index("customer_phone")

    # No TTL index is created here any more — see
    # `_drop_legacy_bookings_ttl_index` above for what it did and why it is gone.
    await _drop_legacy_bookings_ttl_index(coll)

    # Multi-tenant lookups: hotel + phone / guest_id (latest booking)
    await coll.create_index(
        [("hotel_id", 1), ("customer_phone", 1), ("created_at", -1)],
        name="hotel_phone_created",
    )
    await coll.create_index(
        [("hotel_id", 1), ("guest_id", 1), ("created_at", -1)],
        name="hotel_guest_created",
    )
    logger.info("bookings indexes ensured")


async def cleanup_expired_holds() -> list[dict]:
    """
    Release inventory soft holds that have expired (30-min TTL).
    Returns the list of expired booking documents for re-offer messaging.
    """
    coll = bookings_collection()
    now = datetime.now(timezone.utc)
    cursor = coll.find({
        "booking_status": "pending_payment",
        # `$type: "date"` is load-bearing, not decoration. A hold is cleared by setting
        # `hold_expires_at` to the empty *string* (`stage_payment_check.py:80`), and BSON
        # orders strings before dates, so a bare `{"$lt": now}` also matches `""` and would
        # expire exactly the bookings whose holds were deliberately released.
        "hold_expires_at": {"$lt": now, "$type": "date"},
    })
    expired_docs = await cursor.to_list(length=100)
    if expired_docs:
        expired_ids = [d["booking_id"] for d in expired_docs if d.get("booking_id")]
        await coll.update_many(
            {"booking_id": {"$in": expired_ids}},
            {"$set": {"booking_status": "hold_expired", "updated_at": now}},
        )
    return expired_docs


async def create_booking(booking_data: dict[str, Any]) -> str:
    """Insert a confirmed booking document. Returns the booking_id."""
    from app.observability.metrics import metrics

    coll = bookings_collection()
    now = datetime.now(timezone.utc)
    booking_data["created_at"] = now
    booking_data["updated_at"] = now

    hotel_id = str(booking_data.get("hotel_id") or "111111")
    booking_data["hotel_id"] = hotel_id

    # Ensure canonical guest_id linkage
    phone = booking_data.get("customer_phone") or booking_data.get("customer_identifier") or ""
    name = booking_data.get("guest_name") or booking_data.get("customer_name")
    email = booking_data.get("customer_email") or booking_data.get("guest_email")

    if phone and not booking_data.get("guest_id"):
        guest = await guest_repository.get_or_create_guest(
            hotel_id=hotel_id,
            phone=phone,
            name=name,
            email=email,
        )
        booking_data["guest_id"] = guest["guest_id"]

    with metrics.timer("mongo.booking_write_latency_ms", hotel_id=hotel_id):
        await coll.replace_one(
            {"booking_id": booking_data["booking_id"]},
            booking_data,
            upsert=True,
        )

    status = booking_data.get("booking_status", "unknown")
    metrics.incr(
        "booking.create_total",
        outcome=status,
        hotel_id=hotel_id,
    )

    await log_activity(ActivityLog(
        hotel_id=str(hotel_id),
        activity_type="BOOKING_CREATED",
        reference_id=booking_data["booking_id"],
        booking_id=booking_data["booking_id"],
        customer_phone=phone,
        customer_name=name,
        description=f"Booking {booking_data['booking_id']} created via chatbot."
    ))
    # Reserve daily inventory for confirmed bookings (holds are not reserved).
    if booking_data.get("booking_status") == "confirmed":
        await _reserve_booking_inventory(booking_data)
    return booking_data.get("booking_id", "")


async def get_booking(booking_id: str) -> dict[str, Any] | None:
    """Fetch a booking by ID."""
    coll = bookings_collection()
    doc = await coll.find_one({"booking_id": booking_id})
    if doc:
        doc.pop("_id", None)
    return doc


async def _reserve_booking_inventory(booking: dict[str, Any]) -> None:
    """
    Idempotently reserve `daily_inventory` for a confirmed booking's rooms/nights.

    A per-booking `inventory_reserved` marker makes this safe to call from every
    confirmation path (chat confirmer, payment webhook, reconciliation) without
    double-decrementing inventory.

    The marker is written only AFTER every night is in stock. Claiming it first,
    as this did until 2026-10-08, meant a sold-out night left the booking marked
    reserved forever with no inventory actually held — and no way to retry,
    because every later call saw the marker and returned early.
    """
    booking_id = booking.get("booking_id")
    hotel_id = booking.get("hotel_id")
    check_in = booking.get("check_in")
    check_out = booking.get("check_out")
    if not booking_id or not hotel_id or not check_in or not check_out:
        return

    if booking.get("inventory_reserved") is True:
        return  # already reserved

    from app.services.inventory_service import reserve_inventory

    fully_reserved = True
    for room in (booking.get("rooms") or []):
        reserved = await reserve_inventory(hotel_id, room.get("room_id"), check_in, check_out, qty=1)
        if not reserved:
            fully_reserved = False
            logger.warning(
                f"INVENTORY_RESERVE_SHORTFALL | booking={booking_id} room={room.get('room_id')} "
                f"— marker left unset so this booking can be retried"
            )

    if not fully_reserved:
        return

    await bookings_collection().update_one(
        {"booking_id": booking_id, "inventory_reserved": {"$ne": True}},
        {
            "$set": {
                "inventory_reserved": True,
                "inventory_reserved_at": datetime.now(timezone.utc),
            }
        },
    )


async def find_prior_confirmed_by_phone(hotel_id: int | str, phone: str) -> dict[str, Any] | None:
    """Find the most recent confirmed booking for this guest at this hotel (returning guest)."""
    if not phone:
        return None
    str_hid = str(hotel_id)
    variants = [str_hid]
    if str_hid.isdigit():
        variants.append(int(str_hid))

    coll = bookings_collection()
    doc = await coll.find_one(
        {
            "hotel_id": {"$in": variants},
            "$or": [
                {"customer_identifier": phone},
                {"customer_phone": phone},
            ],
            "booking_status": "confirmed",
        },
        sort=[("confirmed_at", -1), ("created_at", -1)],
    )
    if doc:
        doc.pop("_id", None)
    return doc


VALID_CONFIRMATION_SOURCES = {
    "razorpay_webhook_verified",
    "pms_auto_confirm",
    "payment_verified_system",
    "staff_confirmed",
    "front_desk_verified",
}


async def set_booking_confirmed(
    booking_id: str,
    confirmation_source: str,
    payment_id: str | None = None,
    amount_paid: float | None = None,
) -> bool:
    """
    DB-level constraint: Only signature-verified webhook, PMS auto-confirm,
    or authorized staff confirmation can set status to CONFIRMED.
    Any unverified call is rejected.
    """
    if confirmation_source not in VALID_CONFIRMATION_SOURCES:
        logger.error(
            f"SECURITY GUARD: Rejected unverified attempt to confirm booking {booking_id} "
            f"with source '{confirmation_source}'"
        )
        return False

    coll = bookings_collection()
    now = datetime.now(timezone.utc)

    # `payment_status` describes the money, not the confirmation. Confirming a
    # booking does not make an outstanding balance disappear, and hardcoding
    # `"paid"` here is what produced a row carrying `payment_status: "paid"` and
    # `amount_due: 11300` at the same time (BK-20261011-6C18E226, 2026-10-16) —
    # after which the guest was told they had overpaid. `confirm_pending_booking`
    # below, which the payment-screenshot path uses, has always derived the status
    # this way; this makes the two agree.
    snapshot = await coll.find_one(
        {"booking_id": booking_id},
        {"total_price": 1, "amount": 1, "amount_paid": 1},
    ) or {}
    total = float(snapshot.get("total_price") or snapshot.get("amount") or 0.0)
    settled = amount_paid if amount_paid is not None else float(snapshot.get("amount_paid") or 0.0)
    # A total of 0 means we know nothing to be short of, which reads as settled.
    amount_due = max(0.0, total - settled) if total > 0 else 0.0

    update_fields: dict[str, Any] = {
        "booking_status": "confirmed",
        "stage": "confirmed",
        "payment_check_pending": False,
        "confirmation_source": confirmation_source,
        "confirmed_at": now,
        "updated_at": now,
    }
    if get_settings().CONFIRM_WRITES_HONEST_PAYMENT_STATUS:
        update_fields["payment_status"] = "paid" if amount_due <= _AMOUNT_TOLERANCE else "partial"
        if total > 0:
            # Written together with the status so the two cannot disagree.
            update_fields["amount_due"] = amount_due
    else:
        update_fields["payment_status"] = "paid"

    if payment_id:
        update_fields["payment_id"] = payment_id
    if amount_paid is not None:
        update_fields["amount_paid"] = amount_paid

    res = await coll.update_one(
        {"booking_id": booking_id},
        {
            "$set": update_fields,
            "$unset": {"hold_expires_at": ""},
        },
    )
    if res.modified_count > 0:
        doc = await get_booking(booking_id)
        if doc:
            await _reserve_booking_inventory(doc)
    return res.modified_count > 0


async def update_booking_payment(booking_id: str, payment_status: str) -> None:
    """Update payment status on a booking."""
    coll = bookings_collection()
    await coll.update_one(
        {"booking_id": booking_id},
        {"$set": {
            "payment_status": payment_status,
            "updated_at": datetime.now(timezone.utc),
        }},
    )


async def cancel_booking(booking_id: str) -> bool:
    """Cancel a confirmed booking in the database. Returns True if updated, False otherwise."""
    coll = bookings_collection()
    now = datetime.now(timezone.utc)
    res = await coll.update_one(
        {"booking_id": booking_id, "booking_status": {"$ne": "cancelled"}},
        {"$set": {
            "booking_status": "cancelled",
            "updated_at": now,
        }},
    )

    if res.modified_count > 0:
        doc = await get_booking(booking_id)
        if doc:
            hid = doc.get("hotel_id", 111111)
            # Return reserved daily inventory for the cancelled stay.
            # Only when this booking actually reserved inventory (confirmed stays).
            ci, co = doc.get("check_in"), doc.get("check_out")
            if ci and co and doc.get("inventory_reserved"):
                from app.services.inventory_service import release_inventory
                for room in (doc.get("rooms") or []):
                    await release_inventory(hid, room.get("room_id"), ci, co, qty=1)
            await log_activity(ActivityLog(
                hotel_id=int(hid) if str(hid).isdigit() else 111111,
                activity_type="BOOKING_CANCELLED",
                reference_id=booking_id,
                booking_id=booking_id,
                customer_phone=doc.get("customer_phone"),
                customer_name=doc.get("guest_name"),
                description=f"Booking {booking_id} cancelled."
            ))

    return res.modified_count > 0


async def confirm_pending_booking(
    booking_id: str, payment_status: str, amount_paid: float, amount_due: float
) -> bool:
    """Convert a pending hold into a confirmed booking."""
    from app.observability.metrics import metrics

    coll = bookings_collection()
    now = datetime.now(timezone.utc)
    with metrics.timer("mongo.booking_write_latency_ms"):
        res = await coll.update_one(
            {"booking_id": booking_id},
            {
                "$set": {
                    "booking_status": "confirmed",
                    "payment_status": payment_status,
                    "amount_paid": amount_paid,
                    "amount_due": amount_due,
                    "updated_at": now,
                },
                "$unset": {"hold_expires_at": ""}
            },
        )
    ok = res.modified_count > 0
    if ok:
        doc = await get_booking(booking_id)
        if doc:
            await _reserve_booking_inventory(doc)
    metrics.incr("booking.confirm_total", outcome="ok" if ok else "noop")
    return ok


async def get_latest_booking(
    hotel_id: int | str,
    customer_phone: str,
    *,
    booking_status: str | list[str] | None = "confirmed",
) -> dict[str, Any] | None:
    """
    Fetch the latest booking for a customer at a specific hotel.
    Multi-tenant safe: always scoped by hotel_id + normalized phone variants.
    """
    from app.utils.phone import phone_lookup_variants
    from app.observability.metrics import metrics

    if not hotel_id or not customer_phone:
        return None

    str_hid = str(hotel_id)
    variants = phone_lookup_variants(customer_phone)
    if not variants:
        return None

    hid_variants = [str_hid]
    if str_hid.isdigit():
        hid_variants.append(int(str_hid))

    coll = bookings_collection()
    or_clauses: list[dict[str, Any]] = [
        {"customer_phone": {"$in": variants}},
        {"customer_identifier": {"$in": variants}},
    ]

    # Same gap as `get_recent_bookings`, and this is the reader that matters for
    # it: the payment-screenshot validator calls this with the guest's phone and
    # falls back to `session_state["total_price"]` when nothing is found. On a
    # row that carries no phone column that fallback is 0, and the validator's
    # `expected_amount <= 0` arm then tells the guest "our front desk team has
    # been notified" and files the screenshot for staff review — every time, for
    # a booking that was paid in full. Reachable-ness here decides whether a
    # payment is verified automatically or escalated.
    #
    # The soft suffix fallback below does not cover this: it scans candidate rows
    # for a *stored phone* and skips any row whose phone fields are empty, which
    # is exactly the population this branch is for. So this is not redundant with
    # it — it is the only path to those rows.
    #
    # `guest_id = GUEST-<hotel_id>-<phone>` is written by the app, so the hotel
    # and the phone are both pinned by the id itself; a hit cannot belong to a
    # different guest any more than a phone-column hit can. Only the digit-only
    # variants are usable, since a leading "+" never appears in the id.
    if getattr(get_settings(), "BOOKING_GUEST_ID_LOOKUP_ENABLED", False):
        guest_ids = sorted(
            {f"GUEST-{h}-{v}" for h in hid_variants for v in variants if v.isdigit()}
        )
        if guest_ids:
            or_clauses.append({"guest_id": {"$in": guest_ids}})

    query: dict[str, Any] = {
        "hotel_id": {"$in": hid_variants},
        "$or": or_clauses,
    }
    if isinstance(booking_status, str):
        query["booking_status"] = booking_status
    elif isinstance(booking_status, list):
        query["booking_status"] = {"$in": booking_status}

    with metrics.timer("mongo.get_latest_booking_latency_ms", hotel_id=str_hid):
        doc = await coll.find_one(query, sort=[("created_at", -1)])
        if doc:
            doc.pop("_id", None)
            return doc

        # Soft fallback: digit-suffix match within this hotel only
        digits = "".join(ch for ch in customer_phone if ch.isdigit())
        if len(digits) < 10:
            return None

        suffix = digits[-10:]
        soft_query: dict[str, Any] = {"hotel_id": {"$in": hid_variants}}
        if isinstance(booking_status, str):
            soft_query["booking_status"] = booking_status
        elif isinstance(booking_status, list):
            soft_query["booking_status"] = {"$in": booking_status}

        cursor = coll.find(soft_query).sort("created_at", -1).limit(25)
        async for candidate in cursor:
            for field in ("customer_phone", "customer_identifier"):
                stored = "".join(
                    ch for ch in str(candidate.get(field) or "") if ch.isdigit()
                )
                if stored and (
                    stored.endswith(suffix)
                    or suffix.endswith(stored[-10:] if len(stored) >= 10 else stored)
                ):
                    candidate.pop("_id", None)
                    return candidate
        return None


async def get_recent_bookings(
    hotel_id: int | str,
    customer_phone: str,
    *,
    limit: int = 3,
    booking_status: str | list[str] | None = None,
) -> list[dict[str, Any]]:
    """Recent bookings for this phone at this hotel (newest first)."""
    from app.utils.phone import phone_lookup_variants

    if not hotel_id or not customer_phone:
        return []

    str_hid = str(hotel_id)
    variants = phone_lookup_variants(customer_phone)
    if not variants:
        return []

    hid_variants = [str_hid]
    if str_hid.isdigit():
        hid_variants.append(int(str_hid))

    coll = bookings_collection()
    or_clauses: list[dict[str, Any]] = [
        {"customer_phone": {"$in": variants}},
        {"customer_identifier": {"$in": variants}},
    ]

    # The number is not always in those two columns. 373 of 413 rows predate the
    # writer that populates them and carry neither, but do carry
    # `guest_id = GUEST-<hotel_id>-<phone>`. Without this branch those guests
    # resolve to `lead` and get quoted rates for a stay they have already paid
    # for. `phone_lookup_variants` already yields both the 10-digit local and the
    # 12-digit 91-prefixed form the id is written with, so the same variants
    # build the same lookups — only the digit-only ones are usable, since a
    # leading "+" never appears in the id.
    #
    # A hit still requires the exact hotel and the exact phone, so this cannot
    # match a different guest any more than the phone columns can. The
    # `guest_id_1` and `hotel_guest_created` indexes cover it; sorted so the
    # built query is deterministic.
    if getattr(get_settings(), "BOOKING_GUEST_ID_LOOKUP_ENABLED", False):
        guest_ids = sorted(
            {f"GUEST-{h}-{v}" for h in hid_variants for v in variants if v.isdigit()}
        )
        if guest_ids:
            or_clauses.append({"guest_id": {"$in": guest_ids}})

    query: dict[str, Any] = {
        "hotel_id": {"$in": hid_variants},
        "$or": or_clauses,
    }
    if isinstance(booking_status, str):
        query["booking_status"] = booking_status
    elif isinstance(booking_status, list):
        query["booking_status"] = {"$in": booking_status}

    out: list[dict[str, Any]] = []
    cursor = coll.find(query).sort("created_at", -1).limit(max(1, min(limit, 10)))
    async for doc in cursor:
        doc.pop("_id", None)
        out.append(doc)
    return out


async def get_overlapping_bookings(hotel_id: int | str, check_in: str, check_out: str) -> list[dict[str, Any]]:
    """
    Fetch all confirmed bookings for a hotel that overlap with the requested dates.
    Dates are formatted as YYYY-MM-DD.
    Overlapping logic: booking_check_in < requested_check_out AND booking_check_out > requested_check_in.
    """
    if not check_in or not check_out:
        return []

    str_hid = str(hotel_id)
    hid_variants = [str_hid]
    if str_hid.isdigit():
        hid_variants.append(int(str_hid))

    coll = bookings_collection()
    now = datetime.now(timezone.utc)
    cursor = coll.find({
        "hotel_id": {"$in": hid_variants},
        "booking_status": {"$ne": "cancelled"},
        "check_in": {"$lt": check_out},
        "check_out": {"$gt": check_in},
        "$or": [
            {"hold_expires_at": {"$exists": False}},
            {"hold_expires_at": {"$gt": now}}
        ]
    })

    bookings = []
    async for doc in cursor:
        doc.pop("_id", None)
        bookings.append(doc)

    return bookings
