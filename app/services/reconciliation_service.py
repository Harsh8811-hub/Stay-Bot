"""
Payment Reconciliation Service (Ticket C2 / Spec §10.2)

Polls Razorpay / Backend every 5 minutes for pending_payment bookings created
in the last 30 minutes to auto-confirm if the webhook was lost/dropped.
Enforces §14 confirmation authority via 'payment_verified_system'.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from app.db.mongodb import bookings_collection
from app.repositories.booking_repository import cleanup_expired_holds, set_booking_confirmed
from app.services.confirmation_service import send_booking_confirmation_notification
from app.services.payment_service import verify_payment_status

logger = logging.getLogger(__name__)

RECONCILIATION_INTERVAL_SECONDS = 300  # 5 minutes


async def reconcile_pending_razorpay_orders() -> int:
    """
    Finds all bookings with booking_status='pending_payment' created within the last 35 minutes,
    checks their status with Razorpay/Backend, and auto-confirms any paid orders.
    Returns the count of confirmed bookings.
    """
    coll = bookings_collection()
    cutoff = datetime.utcnow() - timedelta(minutes=35)

    cursor = coll.find({
        "booking_status": "pending_payment",
        "created_at": {"$gte": cutoff},
    })
    pending_bookings = await cursor.to_list(length=100)

    confirmed_count = 0
    for booking in pending_bookings:
        booking_id = booking.get("booking_id")
        hotel_id = booking.get("hotel_id")
        payment_link_id = booking.get("payment_link_id")
        payment_link = booking.get("payment_link")

        if not booking_id or not hotel_id:
            continue

        try:
            res = await verify_payment_status(
                hotel_id=hotel_id,
                payment_link_id=payment_link_id,
                payment_link=payment_link,
            )

            if res.is_paid:
                logger.info(
                    f"RECONCILIATION SAFETY NET: Booking {booking_id} was paid on gateway but webhook dropped. "
                    f"Auto-confirming via payment_verified_system."
                )
                payment_id = res.raw.get("payment_id") if isinstance(res.raw, dict) else None
                success = await set_booking_confirmed(
                    booking_id=booking_id,
                    confirmation_source="payment_verified_system",
                    payment_id=payment_id,
                    amount_paid=res.amount_paid,
                )
                if success:
                    confirmed_count += 1
                    try:
                        await send_booking_confirmation_notification(
                            booking_id=booking_id,
                            payment_id=payment_id,
                            amount_paid=res.amount_paid,
                        )
                    except Exception as notify_err:
                        logger.error(
                            f"Reconciliation: failed to dispatch confirmation notification for {booking_id}: {notify_err}"
                        )
        except Exception as e:
            logger.error(f"Reconciliation error for booking {booking_id}: {e}")

    return confirmed_count


async def start_reconciliation_background_task():
    """Background loop running every 5 minutes."""
    logger.info("Starting Razorpay order reconciliation background loop (5 min interval)...")
    while True:
        try:
            await reconcile_pending_razorpay_orders()
        except Exception as e:
            logger.error(f"Error in reconciliation loop: {e}")
        await asyncio.sleep(RECONCILIATION_INTERVAL_SECONDS)


HOLD_CLEANUP_INTERVAL_SECONDS = 300  # 5 minutes


async def start_hold_cleanup_background_task():
    """
    Release expired soft holds every 5 minutes.

    This is the job the destructive bookings TTL index was standing in for, done the way it
    should have been: `cleanup_expired_holds` flips `booking_status` to `hold_expired` and
    never deletes the document. The TTL deleted live bookings instead — 25 October rows were
    lost that way — see `booking_repository.drop_destructive_ttl_index`.

    Caveat worth knowing: Cloud Run scales to zero, so a background loop only ticks while an
    instance happens to be warm. It runs constantly in practice because the bot serves live
    traffic, but it is best-effort, not a guarantee — an idle service sweeps nothing until the
    next request wakes it.
    """
    logger.info("Starting expired-hold cleanup background loop (5 min interval)...")
    while True:
        try:
            expired = await cleanup_expired_holds()
            if expired:
                logger.info("HOLD_CLEANUP released %d expired hold(s)", len(expired))
        except Exception as e:
            logger.error(f"Error in hold cleanup loop: {e}")
        await asyncio.sleep(HOLD_CLEANUP_INTERVAL_SECONDS)
