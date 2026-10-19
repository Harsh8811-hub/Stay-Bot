"""
Discount Engine — validates and applies coupons and other discounts deterministically.

Centralized discount handling ensures coupons do not stack improperly with 
other discounts (e.g. price match) and always follow deterministic business rules.
"""

import logging
import re
from datetime import date, datetime, timezone
from app.models.schemas import Coupon, PriceBreakdown
from app.db.mongodb import coupons_collection

logger = logging.getLogger(__name__)

class DiscountEngine:
    @staticmethod
    async def get_coupon(coupon_code: str, hotel_id: int | str) -> Coupon | None:
        """Fetch a coupon or promotion by code and hotel_id."""
        if not coupon_code or not str(coupon_code).strip():
            return None
        coll = coupons_collection()
        # `re.escape`: the code is whatever the guest typed, and this is a regex.
        # Unescaped, ".*" matched every promotion the hotel had — the widened
        # `$or` below reaches `coupon_name`/`name`, where promotions actually
        # keep their code, so a guest who was never offered a code was handed a
        # 10-15% discount. Escaping also means a code containing "." or "+" only
        # ever matches itself, which is what a coupon lookup should do.
        code_regex = {"$regex": f"^{re.escape(coupon_code.strip())}$", "$options": "i"}
        variants = [hotel_id, str(hotel_id)]
        if str(hotel_id).isdigit():
            variants.append(int(hotel_id))

        doc = await coll.find_one({
            "$or": [
                {"coupon_code": code_regex},
                {"coupon_name": code_regex},
                {"name": code_regex},
                {"title": code_regex},
            ],
            "hotel_id": {"$in": variants}
        })
        if not doc:
            return None

        return DiscountEngine._doc_to_coupon(doc, hotel_id, default_code=coupon_code)

    @staticmethod
    def _doc_to_coupon(doc: dict, hotel_id: int | str, default_code: str = "PROMO") -> Coupon:
        """Normalize document from promotions collection into Coupon schema."""
        c_code = (
            doc.get("coupon_code")
            or doc.get("coupon_name")
            or doc.get("name")
            or doc.get("title")
            or default_code
        )
        d_val = float(doc.get("discount_value") or doc.get("discount") or doc.get("value") or 0.0)
        raw_type = doc.get("discount_type") or doc.get("type")
        if raw_type:
            raw_str = str(raw_type).strip().lower()
            if raw_str in ("percentage", "percent", "%"):
                d_type = "percentage"
            elif raw_str in ("fixed", "flat", "amount", "inr"):
                d_type = "fixed"
            else:
                d_type = raw_str
        else:
            d_type = "fixed" if d_val > 100 else "percentage"

        is_act = doc.get("active")
        if is_act is None:
            is_act = doc.get("is_active") if doc.get("is_active") is not None else (doc.get("status") == "ACTIVE")

        v_from = doc.get("valid_from")
        if not v_from and doc.get("start_date"):
            try:
                v_from = datetime.fromisoformat(str(doc["start_date"]))
            except Exception:
                pass
        v_until = doc.get("valid_until")
        if not v_until and doc.get("end_date"):
            try:
                v_until = datetime.fromisoformat(str(doc["end_date"]))
            except Exception:
                pass

        return Coupon(
            coupon_code=str(c_code),
            hotel_id=hotel_id,
            active=bool(is_act),
            discount_type=str(d_type),
            discount_value=d_val,
            min_booking_value=doc.get("min_booking_value"),
            max_discount_amount=doc.get("max_discount_amount"),
            applicable_room_types=doc.get("applicable_room_types"),
            valid_from=v_from,
            valid_until=v_until,
            usage_limit=doc.get("usage_limit"),
            times_used=doc.get("times_used", 0),
        )

    @staticmethod
    async def get_active_default_promotion(
        hotel_id: int | str,
        check_in_date: str | None = None,
    ) -> Coupon | None:
        """
        Fetch the active hotel-wide promotion that applies by default (e.g. basic-promotion-10).
        These promotions do not require a guest to enter a promo code.
        """
        coll = coupons_collection()
        str_hid = str(hotel_id)
        variants = [str_hid, int(str_hid)] if str_hid.isdigit() else [str_hid]

        try:
            cursor = coll.find({
                "hotel_id": {"$in": variants},
                "$or": [
                    {"is_active": True},
                    {"active": True},
                    {"status": "ACTIVE"},
                ],
            })
            docs = await cursor.to_list(length=50)
        except Exception as e:
            logger.warning(f"Error fetching active default promotions for hotel {hotel_id}: {e}")
            return None

        if not docs:
            return None

        candidates: list[Coupon] = []
        for doc in docs:
            c_name = str(doc.get("coupon_name") or "").strip()
            promo_type = str(doc.get("promotion_type") or doc.get("type") or "").strip().lower()
            adv = doc.get("advanced_settings") or {}
            apply_all = adv.get("applyAll") is True

            # If coupon_name is non-empty, and applyAll is not True, and not basic-promotion,
            # this is a code-gated coupon requiring explicit guest entry
            if c_name and not apply_all and promo_type != "basic-promotion":
                continue

            coupon = DiscountEngine._doc_to_coupon(doc, hotel_id, default_code="PROMO")
            if not coupon.active or coupon.discount_value <= 0:
                continue

            # If last-minute promotion, only apply if booking is within bookablePeriod days of check-in
            if "last-minute" in promo_type or "last minute" in str(doc.get("name") or "").lower():
                if check_in_date:
                    try:
                        ci_clean = str(check_in_date).strip().split("T")[0].split(" ")[0]
                        ci_dt = datetime.strptime(ci_clean, "%Y-%m-%d").date()
                        today = date.today()
                        diff_days = (ci_dt - today).days
                        bookable_period = int(adv.get("bookablePeriod") or 1)
                        m_days = re.search(r'(\d+)\s*days?', str(doc.get("name") or ""), re.IGNORECASE)
                        if m_days:
                            bookable_period = max(bookable_period, int(m_days.group(1)))
                        if diff_days < 0 or diff_days > bookable_period:
                            continue
                    except Exception:
                        continue
                else:
                    continue

            # If early-bird promotion, only apply if booking is well in advance
            if "early-bird" in promo_type or "early bird" in str(doc.get("name") or "").lower():
                if check_in_date:
                    try:
                        ci_clean = str(check_in_date).strip().split("T")[0].split(" ")[0]
                        ci_dt = datetime.strptime(ci_clean, "%Y-%m-%d").date()
                        today = date.today()
                        diff_days = (ci_dt - today).days
                        min_advance = int(adv.get("advancePeriod") or 14)
                        if diff_days < min_advance:
                            continue
                    except Exception:
                        continue
                else:
                    continue

            candidates.append(coupon)

        if not candidates:
            return None

        # Prefer percentage discounts, then higher discount value
        candidates.sort(
            key=lambda c: (1 if c.discount_type == "percentage" else 0, c.discount_value),
            reverse=True,
        )
        return candidates[0]

    @staticmethod
    def validate_coupon(coupon: Coupon, breakdown: PriceBreakdown, check_in_date: str) -> tuple[bool, str]:
        """
        Validate if the coupon can be applied to the current booking.
        Returns (is_valid, reason_if_invalid).
        """
        if not coupon.active:
            return False, "This coupon is no longer active."

        # Check dates
        now = datetime.now(timezone.utc)
        if coupon.valid_from and now < coupon.valid_from.replace(tzinfo=timezone.utc):
            return False, "This coupon is not valid yet."
        if coupon.valid_until and now > coupon.valid_until.replace(tzinfo=timezone.utc):
            return False, "This coupon has expired."

        # Check usage limits
        if coupon.usage_limit is not None and coupon.times_used >= coupon.usage_limit:
            return False, "This coupon has reached its usage limit."

        # Check minimum booking value
        if coupon.min_booking_value and breakdown.subtotal < coupon.min_booking_value:
            return False, f"This coupon requires a minimum booking value of ₹{coupon.min_booking_value:,.2f}."

        # Check room type restrictions
        if coupon.applicable_room_types:
            booked_room_types = [room.room_name for room in breakdown.rooms]
            if not any(rt in coupon.applicable_room_types for rt in booked_room_types):
                return False, "This coupon is not applicable for the selected room type."

        return True, "Valid"

    @staticmethod
    def apply_discount(breakdown: PriceBreakdown, coupon: Coupon) -> PriceBreakdown:
        """
        Apply the coupon discount to the PriceBreakdown.
        Note: Modifies the breakdown in place and returns it.
        """
        # Calculate discount amount based on subtotal (before tax)
        if coupon.discount_type == "percentage":
            discount = breakdown.subtotal * (coupon.discount_value / 100.0)
        else:
            discount = coupon.discount_value

        if coupon.max_discount_amount:
            discount = min(discount, coupon.max_discount_amount)
            
        # Ensure discount doesn't exceed subtotal
        discount = min(discount, breakdown.subtotal)

        # Update breakdown
        breakdown.discount_amount = round(discount, 2)
        label = (
            f"{coupon.discount_value:.0f}% OFF"
            if coupon.discount_type == "percentage"
            else f"₹{coupon.discount_value:,.0f} OFF"
        )
        if "coupon" in coupon.coupon_code.lower():
            breakdown.discount_reason = f"Coupon: {coupon.coupon_code} ({label})"
        else:
            breakdown.discount_reason = f"{coupon.coupon_code} ({label})"
        breakdown.coupon_code = coupon.coupon_code

        # Recalculate tax and total
        discounted_subtotal = breakdown.subtotal - breakdown.discount_amount
        # Assuming tax is proportional to discounted subtotal
        breakdown.tax_amount = round(discounted_subtotal * (breakdown.tax_percent / 100.0), 2)
        breakdown.total = float(round(round(discounted_subtotal + breakdown.tax_amount, 2) / 10) * 10)

        return breakdown
