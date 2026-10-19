"""Save State Node — persist conversation state to MongoDB (always last before END)."""

import logging
from datetime import datetime

from app.models.schemas import MessageRecord
from app.repositories import session_repository
from app.services.repetition_guard import next_repeat_count
from app.utils.guardrails import sanitize_output
from app.workflows.session_mapper import state_to_session
from app.workflows.state import ConversationState

logger = logging.getLogger(__name__)


def flatten_response_cards(cards: list) -> list[dict]:
    """Flatten nested card lists (text_card() already returns a list)."""
    flat: list[dict] = []
    for item in cards:
        if isinstance(item, list):
            flat.extend(flatten_response_cards(item))
        elif isinstance(item, dict):
            flat.append(item)
    return flat


async def save_state(state: ConversationState) -> ConversationState:
    """
    Persist the conversation state back to MongoDB.
    This is always the LAST node before END.
    """
    bot_response = flatten_response_cards(state.get("bot_response", []))

    # Count consecutive identical replies. This has to happen before state_to_session,
    # because that call is what writes to Mongo: anything computed after it is returned to END
    # but never persisted. The values go in as overrides for the same reason.
    last_hash, repeat_count = next_repeat_count(
        state.get("last_response_hash"),
        bot_response,
        state.get("repeat_response_count"),
    )
    if repeat_count > 1:
        logger.info(
            "Repeat reply detected (%s in a row) | session=%s",
            repeat_count,
            state.get("session_id"),
        )

    session = state_to_session(
        state,
        include_messages=True,
        last_response_hash=last_hash,
        repeat_response_count=repeat_count,
    )

    # Preserve conversation_overrides if this turn's state omitted them
    # (prevents wiping Human Takeover / future domain overrides).
    if not state.get("conversation_overrides"):
        existing = await session_repository.get_session(session.session_id)
        if existing and getattr(existing, "conversation_overrides", None):
            session.conversation_overrides = existing.conversation_overrides

    if state.get("user_message"):
        session.messages.append(MessageRecord(
            role="user",
            content=state["user_message"],
            timestamp=datetime.utcnow(),
        ))

    if bot_response:
        bot_text_parts = []
        for card in bot_response:
            if card.get("text"):
                bot_text_parts.append(card["text"])
            elif card.get("type") == "room_options":
                bot_text_parts.append("[Showing Room Options]")
            elif card.get("type") == "quote":
                bot_text_parts.append("[Showing Price Quote]")
            elif card.get("type") == "payment":
                bot_text_parts.append("[Showing Payment Link]")

        bot_text_combined = sanitize_output("\n".join(bot_text_parts))
        session.messages.append(MessageRecord(
            role="assistant",
            content=bot_text_combined,
            timestamp=datetime.utcnow(),
        ))

    session.messages = session.messages[-20:]

    await session_repository.upsert_session(session)
    logger.info(f"Session saved: {session.session_id} | Stage: {session.booking_stage.value}")

    return {
        **state,
        "bot_response": bot_response,
        "last_response_hash": last_hash,
        "repeat_response_count": repeat_count,
    }
