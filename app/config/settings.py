"""
Application settings loaded from environment variables.
Uses pydantic-settings for validation and type coercion.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration — all values come from .env or environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── MongoDB ──────────────────────────────────────────────────
    MONGODB_URI: str
    MONGODB_DATABASE: str = "staychat_web"

    # ── Hotel ────────────────────────────────────────────────────
    HOTEL_ID: str = "111111"
    HOTEL_NAME: str = "our hotel"

    # ── Gemini ───────────────────────────────────────────────────
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # ── ChromaDB ─────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = "./chroma_data"
    KNOWLEDGE_SIMILARITY_THRESHOLD: float = 1.2  # Max cosine distance for RAG confidence

    # ── App ──────────────────────────────────────────────────────
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # ── Rate Limiting ────────────────────────────────────────────
    MAX_MESSAGES_PER_MINUTE: int = 30
    MAX_MESSAGE_LENGTH: int = 500

    # ── Session ──────────────────────────────────────────────────
    SESSION_TTL_HOURS: int = 24

    # How many previous messages of the conversation are handed to the model —
    # both as `RECENT CONVERSATION` in the understand prompt and as
    # `messages_history` in state.
    #
    # The single place to change it. It used to be a literal `10` in two
    # callers, which is half of what `save_state` keeps (`session.messages[-20:]`),
    # so the bot was reading half of what it had saved. Raised to 20 on
    # 2026-09-23 to match the store rather than leave the two numbers to drift.
    HISTORY_WINDOW_MESSAGES: int = 20

    # ── Human Takeover ───────────────────────────────────────────
    # Inactivity window after the latest staff message before AI resumes.
    #
    # 300 (5 min), lowered from 600 on 2026-10-08 to match the CRM. The CRM's
    # `ManualOverride.expires_at` is `now + timedelta(minutes=5)` at both of its
    # creation sites (`api_gateway/views/messages.py`, `webhook_receiver/views.py`),
    # so it stops withholding guest messages at 5 minutes. While this was 600 the
    # two disagreed: a guest writing at minute 6 got past the CRM and was then
    # paused *again* by the bot, and heard nothing for another four minutes.
    # One number, on both sides.
    TAKEOVER_TIMEOUT_SECONDS: int = 300  # 5 minutes; keeps the CRM and the bot agreed

    # ── Context Refresh ──────────────────────────────────────────
    # Whether `POST /api/admin/context-refresh` extracts *typed* staff decisions
    # (pricing, payment, amenities, booking) and applies them as conversation
    # overrides, instead of writing a free-text summary note.
    #
    # It changes what a guest can be charged: with the flag on, a price staff
    # agreed in the takeover becomes the number the quote and payment paths read.
    # A note never did — `prompt_summary()` does not render `notes`, so a summary
    # saying "₹3,000" left the engine quoting its own figure, and the bot
    # contradicted itself between prose and the payment link.
    #
    # ON as of 2026-09-28. It is a *default*, not a constant: pydantic-settings
    # gives the environment precedence, so setting
    # CONTEXT_REFRESH_STRUCTURED_DECISIONS=false on the Cloud Run service turns it
    # back off without a deploy. That route needs `iam.serviceaccounts.actAs`,
    # which this project's operator account does not hold — so in practice the
    # working undo is a revert commit, which Cloud Build deploys under its own
    # service account. Flipping the default here is how it was turned on.
    CONTEXT_REFRESH_STRUCTURED_DECISIONS: bool = True

    # ── Direct booking discount ──────────────────────────────────
    # Whether the bot may answer "any discount?" by granting one itself — the
    # "Exclusive Direct Booking Privilege" ladder in `price_match_handler`:
    # 5% off now, another 5% on the second ask, capped by the season's
    # negotiation limit.
    #
    # OFF as of 2026-10-18 by the client's decision — there is no direct
    # booking discount. With it off the handler still validates a competitor
    # screenshot (a real price match), but it never reduces the price on its
    # own turn and never promises a percentage.
    #
    # It rewrote `total_price`, `amount_due` and `discount_amount` directly,
    # so it moved the number a guest is charged. That is why the code stays
    # behind a flag instead of being deleted: if the policy comes back, the
    # undo is one environment variable rather than a revert commit.
    DIRECT_BOOKING_DISCOUNT_ENABLED: bool = False

    # ── Start Fresh ──────────────────────────────────────────────
    # Whether an explicit "start fresh" wipes the draft even for a guest whose
    # *stay* is already confirmed.
    #
    # Why it needs a switch at all: the reset lives in `apply_intent_overrides`,
    # and `understand` answers a confirmed guest from a branch that returns
    # before that function is ever called. So the reset was unreachable for
    # exactly the guests most likely to want one — measured 2026-10-17, a
    # "/start fresh" logged `STAGE_RESOLVED_FROM_CRM | confirmed stay, no booking
    # row of ours` then `STAGE_2_CONFIRMED_DISPATCH`, no reset, no override line.
    #
    # The reservation is never touched either way — only the session state is
    # dropped, so the guest can start a new enquiry while their booking stands.
    # Turning this off restores the old behaviour, where such a guest keeps the
    # confirmed-stage replies.
    #
    # ON as of 2026-10-17. As with the flag above, the environment wins over this
    # default, but this project's operator account cannot mutate the Cloud Run
    # service, so the working undo is a revert commit.
    START_FRESH_RESETS_CONFIRMED_STAY: bool = True

    # ── Start fresh: a booking row of ours ───────────────────────
    # Whether "start fresh" also dismisses a booking the bot itself created.
    #
    # The flag above covers a stay staff confirmed in the CRM, which never lands in
    # our `bookings` collection. A guest who booked *through the bot* has a row, and
    # `resolve_guest_stage` reads that row on every turn — so without this their
    # reset dropped the draft and nothing else, and the next message came back
    # confirmed. Measured on prod 2026-10-17: `/start fresh` saved `Stage: new`, then
    # `book room , tomorrow 3 people` resolved `stage=confirmed` again off
    # BK-20261016-E7B4FA89.
    #
    # Scoped to the booking id the guest had at the moment they asked, so a later
    # booking is a different id and is honoured normally. The reservation itself is
    # never touched — this changes what the bot treats as the active stay, not the
    # booking. Turning it off restores the old behaviour.
    START_FRESH_DISMISSES_BOOKING_ROW: bool = True

    # ── Group bookings (5+ rooms / 15+ adults) ───────────────────
    # Whether the bot sells a large party itself instead of handing it to the
    # front desk.
    #
    # T6 (2026-09) capped self-serve at four rooms so that anything bigger became
    # a "our reservations team will assist you directly" card. Two things went
    # wrong with that in production:
    #
    #   1. The card repeated on *every* turn, because the guard that sends it is
    #      the same guard that blocks the search — nothing in the flow ever set
    #      `group_lead_submitted`, so there was no way out of the loop.
    #   2. It was a dead end, not a handoff. The guest was asked for contact
    #      details they had already given (their WhatsApp number is the session
    #      id) and, on 2026-10-17, a real 7-room / 17-adult enquiry on 9064956832
    #      received the same paragraph four times and no quote at all.
    #
    # ON: the party is searched and priced like any other, with T6's four-room
    # ceiling lifted in `allocation_engine`. OFF: the old handoff, but sent once
    # per conversation rather than every turn.
    GROUP_BOOKING_SELF_SERVE: bool = True

    # ── Takeover Resume ──────────────────────────────────────────
    # Whether the bot *speaks* when a human takeover ends, instead of only
    # remembering it.
    #
    # Before this, `POST /api/admin/context-refresh` summarised the staff
    # conversation and wrote it into the session, then returned — it never
    # composed or sent anything, so a guest sitting in a paused chat heard
    # silence until they happened to write again. The CRM does call the endpoint
    # when its override expires (`run_override_scheduler`); nothing came of it.
    #
    # ON as of 2026-10-08, and the default is the only working switch: flipping
    # the Cloud Run env var needs `iam.serviceaccounts.actAs`, which the operator
    # account does not hold. The real undo is a revert commit. This sends a real
    # message to a real guest, so treat it accordingly.
    TAKEOVER_RESUME_ENABLED: bool = True

    # ── GST ──────────────────────────────────────────────────────
    # A single rate applied to every property, overriding both the per-hotel
    # `taxes` slabs and the built-in Rs 7,500 / 5% / 18% fallback.
    #
    # This exists because per-property GST is not representable in the database
    # today: all four `hotel_settings` docs carry `tax = {"gst": "", "pan": ""}`
    # (an empty GSTIN) and `tax_config.slab = "include"`, which is inclusive-vs-
    # exclusive *wording*, not a rate. Nothing anywhere stores a percentage, so
    # every property was priced by the same hardcoded slab table.
    #
    # It sat at 5.0 from 2026-09-29 to 2026-09-30, which flattened *every* room
    # to the low band — including the ones at Rs 7,500 a night and above, which
    # have been taxed at 18% since the September 2025 reform. Pranav's ruling on
    # 2026-09-30: Hotel Sahu has no room at Rs 7,500 or more, so the low band
    # covers all of what it sells, and the high band has to be free to bite for
    # the properties that do sell above the threshold.
    #
    # Pricing is GST-**inclusive**, so the rate does not change what a guest
    # pays — it changes only how the invoice splits between taxable value and
    # tax. A too-low rate under-remits GST and overstates revenue; it never
    # overcharges the guest.
    #
    # `None` = per-hotel slabs and the built-in Rs 7,500 / 5% / 18% fallback,
    # which is the correct behaviour and the state this is in now. Set a number
    # only to pin every property to one rate again. The real fix is a
    # per-property rate in `hotel_settings` that `booking_engine`'s `taxes`
    # reader can pick up.
    #
    # Whatever this is set to, `price_match_service` reads it too — check its
    # fallback before changing it, because that one is only reached when this is
    # `None`.
    GST_RATE_OVERRIDE_PERCENT: float | None = None

    # ── Extra-person charge ──────────────────────────────────────
    # Whether an adult above a room's base occupancy is charged the extra person
    # rate on a CP / MAP / AP plan, or only ever when a mattress was asked for.
    #
    # Pranav, 2026-10-06: *"On EP Plan we can allow extra person to stay in room
    # without any extra bed ... EP can have NO EXTRA BED (no extra charges) but
    # that SHOULD NOT be allowed on MAP and CP."*
    #
    # The old rule required the guest to have asked for a bed: both charge sites
    # read `if takes_bed and wants_bed` (`allocation_engine.py:578`) and
    # `if wants_bed and bed_allowed` (`:817`). So a guest on a **CP** plan who
    # said "no extra bed" was charged nothing at all for the third adult who ate
    # breakfast in that room — the plan they bought covers that guest, and the
    # bot gave it away. On **EP** the same silence is correct: the room-only rate
    # buys the room, and a guest who takes no bed takes nothing extra.
    #
    # With this on, `app/utils/meal_plan.charges_an_extra_person` gates the
    # charge on the plan's own type — read from `rate_type_id` where the
    # catalogue states it, falling back to the plan name. The money is still the
    # room type's `extra_adult_rate`: the *permission* is a plan property, the
    # *price* a room-type one.
    #
    # A plan we cannot classify charges nothing above base, so an unknown plan
    # can never start billing guests. The charge also lands on its own field
    # (`extra_bed_count` stays 0) so the quote can say "Extra person" rather than
    # claiming a bed the guest was never offered.
    #
    # `False` restores the bed-only charge exactly, for a one-push rollback.
    EXTRA_PERSON_CHARGED_ON_CP_MAP: bool = True

    # ── Paid early check-in / late check-out ─────────────────────
    # Whether the bot quotes the hotel's configured fee when a guest states an
    # arrival or departure time outside the standard ones.
    #
    # The hotels already sell this. The Control Centre page exists, operators use
    # it, and hotel 111111 has a live `early_late_checkout` row (`27% / 42% /
    # Rs 1502`) in production — but nothing in this repo read it, so a guest who
    # said "I'll reach by 7am, can I check in early?" got no price and no policy.
    # `app/services/early_late_checkout_service.resolve_early_late_fees` now
    # reads it and `booking_engine.format_price_quote` prints the fee as its own
    # line, with the disclaimer attached to the number.
    #
    # This flag is **not** the per-hotel switch and must not be treated as one.
    # That switch is `is_active` inside each hotel's own document, which the
    # Control Centre writes; a hotel with no row is off. This boolean exists only
    # so a defect in the fee path can be shut off in one push without editing
    # Mongo. Note it therefore cannot be flipped from a developer machine — Cloud
    # Run env changes need `iam.serviceaccounts.actAs` — so its real rollback is
    # the revert commit that carries it.
    #
    # The fee is charged ONCE for the stay, as a percentage of one night's room
    # rate (a rate that already includes GST, so the fee is never taxed again),
    # and never on a quote the guest did not ask a time for.
    #
    # `False` restores the old behaviour exactly, for a one-push rollback: no
    # line, no number, no disclaimer.
    EARLY_LATE_CHECKOUT_FEE_ENABLED: bool = True

    # Removed 2026-10-10: `EARLY_LATE_ASK_ENABLED` and the post-selection
    # question it gated. The bot does not raise the subject — Pranav's rule,
    # same day. The guest raises it, and `intent_overrides` answers it: a stated
    # clock is priced, and a mention with no clock gets the hotel's own policy
    # line and a request for the time.

    # ── Rate threshold protection ────────────────────────────────
    # Whether the bot bounds a computed nightly rate by the hotel's own floor and
    # ceiling, read from `hotel_settings.default_rate_threshold` (the Control
    # Centre's "Rate Threshold Protection" page).
    #
    # The page has existed for a while and one hotel has it configured in
    # production (`min 0.73 / max 5.57`), but nothing in this repo read it — the
    # number a guest was charged was unbounded by it. The one modifier on a
    # nightly rate that has no ceiling of its own is the seasonal multiplier
    # (`season_repository` floors it at 0.50 and caps it at nothing), so this is
    # the leash on the season. `app/utils/rate_threshold.py` holds the rule and
    # the two pricing paths both apply it.
    #
    # This flag is **not** the per-hotel switch. That switch is the hotel having
    # a usable window at all; a hotel with no row is not clamped, so the blast
    # radius of this feature today is the one hotel that has one. This boolean
    # exists only so a defect can be shut off in one push without editing Mongo.
    # Like every flag here it **cannot be flipped from a developer machine** —
    # Cloud Run env changes need `iam.serviceaccounts.actAs` — so its real
    # rollback is the revert commit that carries it.
    #
    # `False` restores the old behaviour exactly: every rate as the modifiers
    # left it.
    RATE_THRESHOLD_PROTECTION_ENABLED: bool = True

    # ── Room card ────────────────────────────────────────────────
    # Whether a child the hotel's own child policy makes FREE is absent from the
    # berths the room card packs for, while still counting toward the heads a
    # room must hold.
    #
    # A room has two ceilings and they are different numbers: `max_adults`, how
    # many adults it sleeps, and `max_occupancy`, how many people the hotel will
    # put in it. A free child shares the parents' bed, so it needs no berth of
    # its own and is still a person in the room.
    #
    # On 2026-09-29 this was off, so the card packed a party of five adults and a
    # free 7-year-old as six berths and quoted two King rooms at Rs 3,600 —
    # when a single Deluxe Room Four Bed, which seats 5 and holds 6, cost
    # Rs 3,400. The search had already found that room and offered it as its
    # cheapest option; only the card could not see it.
    #
    # `False` restores the old packing exactly, for a one-push rollback.
    FREE_CHILD_NEEDS_NO_BERTH: bool = True

    # ── Selection round-trip ─────────────────────────────────────
    # Whether a chosen set of rooms records the catalogue's own ceilings on
    # every `SelectedRoom`.
    #
    # `validator` has always *read* `max_adults` and `max_occupancy` off a
    # selected room (`validator.py:282-283`), but `SelectedRoom` never carried
    # either field, so the read always fell through to
    # `assigned_adults + assigned_children`. For the combination allocator that
    # fallback is the distribution it just built and it is correct; for the
    # requirements allocator it is the room's *base* occupancy, which is a
    # different number — a Deluxe King Bed is recorded there as 2 seats when the
    # room seats 3.
    #
    # Measured 2026-09-29, hotel 111111, six adults: the guest asked for
    # "2 Delux room king bed", the search offered `2 x King` at Rs 3,600, and
    # replying "1" drew
    # `VALIDATOR_SELECTION_MISMATCH: capacity=4/6 ... Triggering fresh search`
    # — forever, because two Kings recorded 4 seats against a party of 6 and the
    # validator threw the selection away and redrew the identical card.
    #
    # `False` restores the fallback exactly, for a one-push rollback.
    SELECTION_RECORDS_ROOM_CEILINGS: bool = True

    # Whether a selection made from the room card leaves `room_type_preference`
    # alone.
    #
    # The SELECT_ROOM branches write the matched option's `display_label` —
    # "2x DELUXE ROOM KING BED" — into `room_type_preference`, which is the field
    # the search reads as a room *type*: `room_searcher.py:73` →
    # `room_service.py:143` → `filter_rooms_by_reference`. So choosing a row
    # silently pinned every later search to the rooms that one set happened to
    # contain, and overwrote whatever the guest had actually asked for.
    #
    # Measured 2026-09-29 on the same transcript: after "child is 15 years old"
    # moved the party to six adults, the candidate pool held `DELUXE ROOM FOUR
    # BED` and nothing else, so `2 x King` at Rs 3,600 could not be offered
    # against `2 x Four Bed` at Rs 6,800. The selection is already recorded in
    # `selected_rooms`; the preference belongs to the guest's own words.
    #
    # `False` restores the write exactly, for a one-push rollback.
    SELECTION_DOES_NOT_PIN_ROOM_TYPE: bool = True

    # Whether a change to the *party* drops a `room_type_preference` the guest
    # stated for the old one.
    #
    # A preference is pinned against a party size. "A deluxe room" means a
    # different room for two people than for six, and one the guest named for
    # five says nothing at all about six — yet it survived the change and went on
    # narrowing every search that followed, which is how a party that grew by one
    # child kept being quoted the one room type it had already outgrown.
    #
    # Measured 2026-09-29 (hotel 111111): "child is 15 years old" moved the party
    # from five adults and a child to six adults. The preference carried over as
    # `Deluxe Room Four Bed`, so the candidate pool for the next search held that
    # room and nothing else, and `2 x Deluxe Room King Bed` at Rs 3,600 could not
    # be offered against `2 x Deluxe Room Four Bed` at Rs 6,800.
    #
    # **Only a *stale* preference.** A guest who restates one in the same message
    # as the new party — "6 adults, we want the king" — has that preference in
    # `changed_fields` too, and it is theirs and current, so it is kept. This
    # clears only what was carried over from an earlier turn.
    #
    # `False` restores the carry-over exactly, for a one-push rollback.
    PARTY_CHANGE_DROPS_ROOM_PREFERENCE: bool = True

    # ── Confirmed bookings ───────────────────────────────────────
    # Whether a session whose booking is already confirmed stops trying to
    # sell after that.
    #
    # Pranav, 2026-09-30: *"When booking confirmed, stop the bot on that chat
    # (it keeps saying book room etc, make it faq based i guess but if thats a
    # big change in code dont do it, jsut stop the bot)"*.
    #
    # The confirmed router in `understand.py` handled a new-booking request and
    # a greeting, and let everything else fall through to the general flow —
    # which re-runs the room search and the upsell on a guest who has already
    # paid.
    #
    # This is the smaller of the two options he offered: a message naming a
    # side topic (amenities, directions, check-in, invoice, or simply ending in
    # "?") still falls through to the amenity / location / FAQ branches, and
    # anything else gets one closing reply instead of a room card.
    #
    # `False` restores the fall-through exactly, for a one-push rollback.
    CONFIRMED_BOOKING_STOPS_THE_BOT: bool = True

    # ── Confirmation slip on a screenshot payment ────────────────
    # Whether verifying a UPI screenshot in chat also sends the guest the full
    # booking confirmation slip.
    #
    # Three routes already confirm a booking and send the slip: the Razorpay
    # webhook, staff confirmation in the Control Centre, and the nightly
    # reconciliation sweep. A fourth — the guest photographing a UPI receipt,
    # which is how Hotel Sahu's bookings are actually paid — confirmed the
    # booking and sent nothing. Measured 2026-10-11 on `BK-20261010-94330B97`
    # (Hotel Sahu, 11-12 Oct, ₹460 of ₹1,550): the guest's entire confirmation
    # was "Payment of ₹460 verified successfully! 🎉", with no dates, no room, no
    # address; the last slip of any kind reached any guest on 2026-09-01, and
    # only 5 of 412 bookings have `confirmation_sent`.
    #
    # This does NOT replace the acknowledgement line the guest gets today — the
    # slip is sent on top of it, so a slip failure costs them nothing they had.
    #
    # `False` stops the slip being sent from the screenshot path and leaves the
    # acknowledgement line exactly as it is, for a one-push rollback. Honest
    # caveat, same as the other flags here: this is a Cloud Run env var, so it
    # cannot actually be flipped from a machine without
    # `iam.serviceaccounts.actAs` — the real rollback is a revert commit.
    CONFIRMATION_SLIP_ON_SCREENSHOT_PAYMENT: bool = True

    # ── UPI screenshot auto-verification ─────────────────────────
    # Whether a payment screenshot alone can confirm a booking.
    #
    # Pranav, 2026-10-16: *"turn off UPI payment auto verification."*
    #
    # With this on, `validate_payment` (`app/services/vision/validator.py`)
    # reads the amount out of the guest's screenshot, decides it matches the
    # balance, writes `payment_status: paid` and calls
    # `confirm_pending_booking` — a booking confirmed by a picture, with no
    # human anywhere in the loop.
    #
    # With it off, arms A and B of the financial evaluation are skipped
    # entirely: the screenshot is **queued as a front-desk task**, the guest is
    # told the receipt has been received and the team will confirm shortly, and
    # **nothing is recorded as paid** — no `amount_paid`, no `payment_status`,
    # no confirmation. Only staff moving the task to COMPLETED confirms.
    #
    # The duplicate-UTR, tampering, FAILED/DECLINED and unreadable-amount
    # rejections all still run *before* this gate, so a screenshot the old code
    # would have rejected is still rejected, with the same wording.
    #
    # `True` restores auto-confirmation from the screenshot path exactly as it
    # behaved before. Same Cloud Run caveat as the flags above: pushing is the
    # only thing that moves this, so the rollback is a revert commit.
    UPI_SCREENSHOT_AUTO_VERIFICATION: bool = False

    # ── Confirmation writes an honest ledger ─────────────────────
    # Whether staff completing a payment task is taken as proof that the whole
    # booking total arrived.
    #
    # The bug, measured on `BK-20261011-6C18E226` (guest 917017478117) on
    # 2026-10-16: a ₹4,000 advance against a ₹15,300 booking. The bot correctly
    # told the guest "Total Paid: ₹4,000, Balance Due at check-in: ₹11,300" at
    # 13:03:29. Staff completed the payment task at 13:03:59, and at 13:06:57
    # the same bot said the ₹15,300 was fully paid — then, after a further
    # ₹2,000 screenshot, told the guest they had *overpaid* by ₹2,000 when in
    # fact they still owed about ₹5,000.
    #
    # Cause: `app/api/routes/tasks.py` substituted the booking's `total_price`
    # for the amount whenever staff did not state one, and passed it as
    # `amount_paid`. Confirming a booking is not evidence that money arrived.
    #
    # `False` records an amount only when the staff request actually carried one
    # — `UpdateTaskStatusRequest` has no such field, so the generic status route
    # records none, and `confirm-payment` records whatever `amount_paid` it was
    # given. `True` restores the substitution exactly.
    STAFF_CONFIRM_ASSERTS_FULL_PAYMENT: bool = False

    # Whether `set_booking_confirmed` derives `payment_status` from the money on
    # the record, instead of hardcoding `"paid"`.
    #
    # Same booking: the row ended up with `payment_status: "paid"` and
    # `amount_due: 11300` at the same time — a self-contradicting record, and
    # three different answers for the same fact across `bookings`,
    # `workflow_state` and the confirmation slip.
    #
    # `True` writes `"paid"` only when the recorded money actually covers the
    # total, and `"partial"` otherwise — which is what the sibling
    # `confirm_pending_booking` (used by the payment-screenshot path) already
    # does. It also writes the matching `amount_due` when a total is known, so
    # the two fields cannot disagree.
    #
    # `False` restores the hardcoded `"paid"`. Note the reach: this function is
    # also the webhook and reconciliation confirm path, and
    # `app/workflows/nodes/payment/link.py:160` branches on
    # `payment_status != "paid"` when deciding whether to reuse an existing
    # payment link — so a partial payment that used to read `"paid"` now reads
    # `"partial"` and may be issued a fresh link. That is the intended reading,
    # but it is the part to watch if a guest reports a duplicate link.
    CONFIRM_WRITES_HONEST_PAYMENT_STATUS: bool = True

    # ── Payment-turn understanding ───────────────────────────────
    # Whether the model gets a say in whether a guest's message is a decision
    # to pay, or a request to change something first.
    #
    # The rule it replaces is `_NEGOTIATION_OR_CHANGE` in
    # `app/utils/intent_overrides.py` — an eleven-word blocklist. It exists
    # because the payment step once fired on "Thoda kam karo", "Actually show me
    # non AC rooms instead" and "STOP". It now catches those three, and it
    # catches them because `actually\s+show` was added to the regex by hand
    # after the fact. The next change request phrased with different words
    # ("make it two rooms instead") goes to payment again.
    #
    # With this on, `classify_payment_turn` (gemini_service) answers
    # CONFIRM / CHANGE / UNCLEAR for the turn, and both payment gates
    # (`routing.route_by_intent` and `validator`) prefer that answer.
    #
    # **The classifier can only agree or abstain.** UNCLEAR, a timeout, a
    # missing key, a malformed reply, or the flag being off all fall back to
    # the regex — so a Gemini outage cannot open the payment step to a message
    # the regex would have held back, and cannot route a real confirmation into
    # the booking flow. Every disagreement between the two is logged.
    #
    # `False` restores the regex-only behaviour exactly, for a one-push
    # rollback, and skips the extra model call entirely.
    PAYMENT_TURN_LLM_VERDICT: bool = True

    # ── Language detection ───────────────────────────────────────
    # Whether the lexical language cues are weighted, or simply counted.
    #
    # Pranav, 2026-09-30: *"detection of Hindi, Hinglish (Roman Hindi) is not
    # consistent, it still replies in english most of the time."*
    #
    # The old rule was a straight head-count — `h_count >= e_count` for
    # Hinglish, `e_count > h_count` for English — and the two sides are not
    # comparable. Hinglish borrows English *content* words ("room", "book",
    # "price", "family"); English borrows no Hindi. So one borrowed noun could
    # outvote real Hindi:
    #
    #   "2 adults aur 1 child"      -> English   ("aur" was not even a cue)
    #   "room ki photo bhejo"       -> no cues at all -> English default
    #   "thank you" (mid-Hinglish)  -> English, and it stuck for the session
    #
    # Three changes, all in `understand.py`:
    #
    #   1. The cue set gains the function words Roman Hindi is actually made
    #      of — `kya`, `mujhe`, `ka`, `ki`, `ke`, `aur`, `wala`, `bhejo`. Their
    #      absence is most of why messages scored zero Hindi cues.
    #   2. Hindi wins ties and near-ties: English must score more than *twice*
    #      the Hindi count to take the turn. An English word inside a Hindi
    #      sentence is not evidence of English.
    #   3. An acknowledgement — "ok", "thanks", "done", "yes" — changes
    #      nothing, from the cue counts or from the model's read of them. It
    #      answers a question; it does not choose a language.
    #
    # Measured on 34 real strings against a human read of the same 34: 3
    # wrong under the head-count, 0 under the weighted rule. Exactly three
    # strings change: "room ki photo bhejo" and "2 adults aur 1 child" become
    # Hinglish, and "thank you" stops choosing a language.
    #
    # `False` restores the head-count and the legacy cue set exactly, for a
    # one-push rollback.
    LANGUAGE_DETECTION_WEIGHTS_HINDI: bool = True

    # ── Guest name capture ───────────────────────────────────────
    # Whether the name prompt reads a *name* out of the guest's reply, or takes
    # the reply whole.
    #
    # Pranav, 2026-09-30: *"better name parsing when asking for guests name,
    # right now when name is asked it takes the whole thing no matter what the
    # customer sent"*.
    #
    # The old read took `state["user_message"]` verbatim, stripped one known
    # prefix, and stored whatever survived if it was at most four words, held no
    # digit, and matched a letters-only regex. So `"hi i am rahul"` passed every
    # check and the guest was saved as **"Hi I Am Rahul"**, and `"Rahul English"`
    # swallowed the language answer into the name. The checks looked for reasons
    # to *reject* a whole message; none of them looked for a name.
    #
    # `app/utils/person_name.py` scans instead: greetings and self-introduction
    # filler are dropped, the leading run of name-shaped tokens is kept, and the
    # scan stops at the first token that is not name-shaped. A reply with no
    # name in it now returns `None` — which means the bot asks again, the safe
    # direction, rather than calling the guest "Looking".
    #
    # `False` restores the verbatim read exactly, for a one-push rollback.
    NAME_CAPTURE_READS_ONLY_THE_NAME: bool = True

    # ── Payment Backend ──────────────────────────────────────────
    STAYCHAT_BACKEND_URL: str = "https://staychat-backend-master-316221817495.europe-west1.run.app"
    STAYCHAT_PAYMENT_ORDER_KEY: str = "jzEWLPDtJCDzpkM3LD6XfiSlcL4QPhed"

    # Declared so pydantic-settings will actually read them: this model sets
    # extra="ignore", so an env var with no matching field is discarded and
    # getattr(settings, ...) returns None. Without these two lines the Razorpay
    # webhook computed an empty secret and rejected every request with 400.
    #
    # Both default to empty, and payment_webhook fails closed on an empty
    # secret. Do NOT give these a hardcoded fallback: anyone who can read this
    # repository could then forge a "payment captured" webhook and confirm
    # bookings without paying. Set the real value in the deploy environment.
    RAZORPAY_WEBHOOK_SECRET: str = ""
    RAZORPAY_KEY_SECRET: str = ""

    # ── Media Processing ─────────────────────────────────────────
    GOOGLE_APPLICATION_CREDENTIALS: str | None = None
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_IMAGE_TYPES: list[str] = ["jpg", "jpeg", "png"]
    ALLOWED_PDF_TYPES: list[str] = ["pdf"]
    ALLOWED_AUDIO_TYPES: list[str] = ["mp3", "wav", "m4a", "ogg", "webm"]

    # ── CRM & Staff Authentication (GCP/VM Environment Ready) ────
    STAFF_PHONE_NUMBERS: str = ""  # Comma-separated staff phone numbers for VM/GCP env fallback
    CRM_BACKEND_URL: str = "https://combot-crm-v2dot1-316221817495.asia-south1.run.app/api/backend/messages/send/"
    CRM_CHAT_HISTORY_URL: str = "https://combot-crm-v2dot1-316221817495.asia-south1.run.app/api/backend/messages/"
    CRM_MONGO_URI: str = "mongodb+srv://staychataideployment_db_user:EWMMB7tjgguZfBNw@chatbot.9xabdau.mongodb.net/"
    CRM_MONGO_DATABASE: str = "bot_migration"
    BACKEND_API_SECRET_KEY: str = "django-insecure-staychat_super_secret_key"

    # ── CRM confirmation lookup ──────────────────────────────────
    # Whether the bot may ask the CRM whether a guest's stay is already confirmed by
    # staff, and suppress its own payment demand if so.
    #
    # The problem this addresses (2026-10-12): a booking confirmed by staff in the
    # Control Centre is written to the CRM's database and stops there. Nothing carries
    # it into the bot's `bookings`, so the stage resolver still reads `pending_payment`
    # and re-enters the payment flow — a guest who has already paid is asked for the
    # advance again. Measured on hotel 111111: paid 2026-10-11 13:26, staff confirmed
    # 13:34, and at 06:37 the next morning the bot minted a new booking and demanded
    # ₹8,040.
    #
    # This is a READ of another team's cluster, in the guest's turn. Switched ON
    # 2026-10-13 after a live replay on hotel 111111: same real confirmed guest, same
    # dates, flag off → the bot minted a booking and demanded a ₹2,300 advance via UPI;
    # flag on → "our team has already confirmed this booking", no booking, no demand.
    #
    # `False` restores the old behaviour exactly, but NOT from a machine: it is a Cloud
    # Run env var and this account lacks `iam.serviceaccounts.actAs`, so the real
    # rollback is a revert commit. Set it here, not in the deploy config, for that reason.
    CRM_CONFIRMATION_LOOKUP_ENABLED: bool = True

    # The confidence below which a CRM `CONFIRMED_BOOKING` is not believed. Measured
    # 2026-10-12 across the CRM's 167 confirmed contacts: 155 sit at ≥0.95, and the 3
    # below 0.9 all carry no dates and no amount, so this floor drops exactly the ones
    # that could not be matched to a stay anyway.
    CRM_CONFIRMATION_MIN_CONFIDENCE: float = 0.9

    # Which CRM database holds `contact`. Deliberately NOT `CRM_MONGO_DATABASE` above —
    # that one says `bot_migration`, which is not where the CRM keeps its data, and
    # nothing reads it today.
    #
    # The CRM's real collections live in a database named `test`: its own `MONGO_URL`
    # carries no database in the path, so MongoEngine falls back to its default name.
    # The name is an accident rather than a decision, which is exactly why it is a
    # setting: if the CRM ever names the database properly, that is a config change and
    # not a code change. Open question for the CRM team.
    CRM_CONFIRMATION_DATABASE: str = "test"

    # Whether a CRM-confirmed stay whose check-out date has NOT passed puts the
    # guest into the confirmed stage — the FAQ/concierge path, zero selling — even
    # though the bot owns no booking row for them. Pranav's rule, 2026-10-13: "if
    # the end date of the stay has not passed act like an faq bot, and if it has
    # passed act normally."
    #
    # No check-out date on the CRM contact → we cannot say the stay has not ended →
    # the guest is treated normally. Fails closed, like everything else here.
    #
    # Deliberately separate from `CRM_CONFIRMATION_LOOKUP_ENABLED`: that flag also
    # drives the payment-demand suppression, so turning the stage promotion off must
    # not silently switch the payment fix back on.
    CRM_CONFIRMED_STAGE_ENABLED: bool = True

    # Translate the buttons and lists the templates author into the `interactive` payload
    # Meta accepts, instead of flattening them into the message body.
    #
    # The templates have authored `button_group` cards since 2026-10-14, but nothing
    # converted them and `format_cards_for_whatsapp` has no branch for the type, so every
    # button was dropped and the guest received the card's body text alone — the main menu
    # arrived as "Here is what I can help you with at Hotel Sahu:" with no options under it.
    # The CRM passes an `{"type": "interactive", "data": {...}}` card straight through to
    # Meta (`combot_crm_v2/webhook_receiver/views.py:305`), so only the bot side changes.
    #
    # ON is the default **because production is already broken without it** — the gate
    # cannot be flipped from this account (`iam.serviceaccounts.actAs`), so a default of
    # False would ship the dead menu and need a second deploy to fix. Turning it off in the
    # Cloud Run console is the kill switch: every prompt falls back to the numbered text it
    # carried before, which is the older, known-good behaviour.
    WHATSAPP_BUTTONS_ENABLED: bool = True

    # Say which half of a room request could not be met, and what IS free.
    #
    # AC / Non-AC is a *hard* inventory filter: when no room of the requested
    # class is free, the search returns empty and the guest is told "we don't
    # have rooms available that match your requirements" — even though other
    # rooms are sitting open for the same dates. A live guest (918017904770,
    # 2026-10-17) asked for one non-AC room for 18–21 Oct, when both non-AC
    # room types were closed on the 20th but two AC room types were free on
    # every night; the bot said the same sentence three times and only a staff
    # member quoting a different property moved the booking.
    #
    # ON by default for the same reason as the flags above: the gate cannot be
    # flipped from this account, so a default of False would ship the dead-end
    # message and need a second deploy to fix. Turning it off in the Cloud Run
    # console restores the plain "no rooms available" card exactly as it was.
    ROOM_CLASS_GAP_HINT_ENABLED: bool = True

    # Answer what the guest asked while a payment is under verification.
    #
    # `handle_stage_payment_check` returns one fixed paragraph — "we have noted
    # your payment, the front desk is verifying it" — for every turn it serves.
    # That is the right reply to a payment claim and the wrong reply to
    # everything else. Live guest 919833515350 (2026-10-17) was in this stage
    # after saying "I have already paid 1500", then asked "Hi", "Can u send me
    # my booking confirmation" and "But aaj 3 baje mera checkin hai.. jaldi se
    # confirmation bhejiye" — and got the same paragraph verbatim three times,
    # with no acknowledgement that they had a check-in in a few hours. Staff
    # confirmed by hand minutes later.
    #
    # ON by default for the same reason as the flags above: the gate cannot be
    # flipped from this account. Turning it off restores the single paragraph.
    PAYMENT_CHECK_STATUS_REPLIES_ENABLED: bool = True

    # ── A confirmed guest asking for the room list ────────────────────────────
    #
    # "Room types & rates" and "Book a room" are reply buttons on the menu a
    # confirmed guest is still shown, and `routing.py` sends every confirmed turn
    # to `stage_confirmed` no matter which button was tapped. The tap used to
    # reach that handler's tail and come back with the directions/parking
    # paragraph — an answer to a different question. ZERO SELLING still forbids
    # showing the catalogue, so the reply is S57's: the current stay is safe, is
    # this a separate booking? Pranav's ruling, 2026-10-18.
    #
    # ON by default for the same reason as the flags above. Turning it off
    # restores the canned paragraph for a rooms tap.
    CONFIRMED_ROOMS_ASK_SEPARATE_BOOKING: bool = True

    # ── Booking lookup by `guest_id` ──────────────────────────────────────────
    #
    # `get_recent_bookings` and the stage resolver used to match a guest on
    # `customer_phone` and `customer_identifier` only. 373 of the 413 rows in the
    # bookings collection carry neither, because they predate the writer that
    # populates them — so those guests resolved to `lead` and were quoted rates
    # for a stay that was already confirmed and paid for.
    #
    # The number is not missing, only unread: those rows carry
    # `guest_id = GUEST-<hotel_id>-<phone>`, and the `guests` collection holds the
    # same number under `phone` (they agreed on 56 of 59 sampled rows). This flag
    # adds a third `$or` branch matching that id, which the `guest_id_1` and
    # `hotel_guest_created` indexes already cover.
    #
    # ON by default for the same reason as the flag above: the gate cannot be
    # flipped from this account, so a default of False would ship the broken
    # behaviour and need a second deploy to fix. Turn it off in the Cloud Run
    # console to fall back to the phone-only lookup, which is byte-identical to
    # the query this code ran before.
    BOOKING_GUEST_ID_LOOKUP_ENABLED: bool = True

    # ── Gemini token usage logging ────────────────────────────────────────────
    #
    # Emits one `GEMINI_USAGE` log line per Gemini call, carrying prompt /
    # cached / output / total token counts. Nothing in this codebase read
    # `response.usage_metadata` before, so LLM spend could not be attributed to
    # a function and there was no way to tell whether Google's implicit context
    # caching was hitting the large static prefixes this bot resends each turn.
    #
    # ON by default because the flag cannot be flipped from this account — a
    # default of False would ship a build with no visibility and need a second
    # deploy to fix. Turn it off in the Cloud Run console if the log volume ever
    # becomes a problem; the wrapper then does nothing and behaviour is
    # byte-identical to the call it made before.
    GEMINI_USAGE_LOGGING: bool = True


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton — parsed once on first call."""
    return Settings()
