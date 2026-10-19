"""
Admin API Routes
"""

import logging
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from bson import ObjectId
from datetime import datetime, UTC

from typing import Any, Dict
from app.services.hotel_config_service import invalidate_cache
from app.db.mongodb import knowledge_requests_collection, knowledge_base_collection, booking_sessions_collection
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/admin", tags=["Admin"])


class InvalidateCacheRequest(BaseModel):
    hotel_id: int | None = None


@router.post("/config/invalidate-cache")
async def invalidate_hotel_config_cache(request: InvalidateCacheRequest):
    """
    Invalidates the hotel configuration cache.
    If hotel_id is provided, only that hotel's cache is invalidated.
    Otherwise, the entire cache is cleared.
    """
    try:
        invalidate_cache(request.hotel_id)
        from app.services.personalization_service import invalidate_personalization_cache
        invalidate_personalization_cache(request.hotel_id)
        msg = f"Cache invalidated for hotel_id={request.hotel_id}" if request.hotel_id else "Entire config cache invalidated"
        logger.info(f"ADMIN_ACTION | {msg}")
        return {"status": "success", "message": msg}
    except Exception as e:
        logger.error(f"Failed to invalidate cache: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


class AnswerKnowledgeRequest(BaseModel):
    ticket_id: str
    hotel_id: int
    staff_id: int
    title: str
    answer: str


@router.post("/knowledge/answer-request")
async def answer_knowledge_request(request: AnswerKnowledgeRequest):
    """
    Staff answers a pending knowledge request.
    This creates a new Knowledge-Base article and triggers ChromaDB reindexing.
    """
    try:
        req_col = knowledge_requests_collection()
        kb_col = knowledge_base_collection()
        
        # Verify ticket exists
        ticket = await req_col.find_one({"_id": ObjectId(request.ticket_id), "hotel_id": request.hotel_id})
        if not ticket:
            raise HTTPException(status_code=404, detail="Knowledge request not found")
        
        now = datetime.now(UTC)
        
        # 1. Create Knowledge Base Article
        kb_doc = {
            "hotel_id": request.hotel_id,
            "title": request.title,
            "content": request.answer,
            "created_by": request.staff_id,
            "created_at": now,
            "updated_by": None,
            "updated_at": None
        }
        kb_insert = await kb_col.insert_one(kb_doc)
        logger.info(f"KNOWLEDGE_BASE_UPDATED | hotel_id={request.hotel_id} | kb_id={kb_insert.inserted_id}")
        
        # 2. Re-index ChromaDB
        await sync_knowledge_base_from_mongo(request.hotel_id)
        logger.info(f"CHROMA_REINDEX_COMPLETED | hotel_id={request.hotel_id}")
        
        # 3. Mark request as Answered
        await req_col.update_one(
            {"_id": ObjectId(request.ticket_id)},
            {
                "$set": {
                    "status": "Answered",
                    "staff_answer": request.answer,
                    "knowledge_base_article_id": str(kb_insert.inserted_id),
                    "answered_by": request.staff_id,
                    "answered_at": now,
                    "updated_at": now
                }
            }
        )
        logger.info(f"KNOWLEDGE_REQUEST_ANSWERED | hotel_id={request.hotel_id} | ticket_id={request.ticket_id}")
        
        return {"status": "success", "message": "Knowledge base updated and re-indexed"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to answer knowledge request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

# ═══════════════════════════════════════════════════════════════════
# Human Intervention (Takeover) Endpoints
# ═══════════════════════════════════════════════════════════════════

from app.services.takeover_engine import start_or_update_intervention, resolve_intervention

class TakeoverMessageRequest(BaseModel):
    hotel_id: int
    session_id: str
    staff_id: int | str
    message: str
    reason: str = "NEGOTIATION"

@router.post("/interventions/message")
async def send_takeover_message(request: TakeoverMessageRequest):
    """
    Staff/CRM push path: a hotel staff message activates or extends Human Takeover
    (10-minute inactivity timer, session-persisted conversation_overrides).
    """
    try:
        intervention_id = await start_or_update_intervention(
            hotel_id=request.hotel_id,
            session_id=request.session_id,
            staff_id=request.staff_id,
            staff_message=request.message,
            reason=request.reason
        )
        return {"status": "success", "intervention_id": intervention_id}
    except Exception as e:
        logger.error(f"Takeover message failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process takeover message")


class ResumeInterventionRequest(BaseModel):
    intervention_id: str

@router.post("/interventions/resume")
async def resume_ai_intervention(request: ResumeInterventionRequest):
    """
    Staff manually ends Human Takeover. Clears session takeover flag.
    Pending decisions may be captured but are NOT applied to booking/pricing yet.
    """
    try:
        await resolve_intervention(request.intervention_id)
        return {"status": "success", "message": "Intervention resolved, AI resumed."}
    except Exception as e:
        logger.error(f"Failed to resume intervention: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to resume AI")


@router.get("/metrics")
async def get_production_metrics():
    """
    In-process production metrics snapshot for ops / dashboards.

    Includes Gemini/Chroma/Mongo latency, feature-gate decisions, booking/payment
    outcomes, takeovers, knowledge gaps, tasks, and chat turn rates.
    """
    from app.observability.metrics import metrics
    return metrics.snapshot()


# ═══════════════════════════════════════════════════════════════════
# Debug / diagnostics
# ═══════════════════════════════════════════════════════════════════

@router.get("/debug/failures")
async def get_recent_failures(
    limit: int = 50,
    component: str | None = None,
    hotel_id: int | None = None,
    session_id: str | None = None,
):
    """
    Recent in-process failures (ring buffer, last ~200).
    Grep Cloud Run logs by ``trace_id`` for the full story.
    """
    from app.observability.failure_store import recent_failures
    from app.observability.tracing import get_trace_id

    items = recent_failures(
        limit=limit,
        component=component,
        hotel_id=hotel_id,
        session_id=session_id,
    )
    return {
        "trace_id": get_trace_id(),
        "count": len(items),
        "failures": items,
    }


@router.get("/debug/session/{session_id}")
async def debug_session_dump(session_id: str, activity_limit: int = 30):
    """
    Full diagnostic dump for a booking session — stage, overrides, takeover,
    recent activity logs, recent interventions, and matching in-memory failures.
    """
    from app.db.mongodb import activity_logs_collection, human_interventions_collection
    from app.observability.failure_store import recent_failures
    from app.observability.tracing import get_trace_id, mask_phone
    from app.repositories import session_repository

    session = await session_repository.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    session_data = session.model_dump(mode="json")
    phone = session.customer_identifier
    hotel_id = session.hotel_id

    # Recent activity for this guest / hotel
    activities: list[dict] = []
    try:
        coll = activity_logs_collection()
        query: dict = {"hotel_id": hotel_id}
        if phone:
            query["customer_phone"] = phone
        cursor = coll.find(query, sort=[("timestamp", -1)]).limit(max(1, min(activity_limit, 100)))
        async for doc in cursor:
            doc["_id"] = str(doc.pop("_id", ""))
            activities.append(doc)
    except Exception as e:
        logger.warning("debug session activity query failed: %s", e)

    interventions: list[dict] = []
    try:
        icoll = human_interventions_collection()
        cursor = icoll.find(
            {"$or": [{"session_id": session_id}, {"hotel_id": hotel_id, "session_id": session_id}]},
            sort=[("started_at", -1)],
        ).limit(10)
        async for doc in cursor:
            doc["_id"] = str(doc.pop("_id", ""))
            interventions.append(doc)
    except Exception as e:
        logger.warning("debug session interventions query failed: %s", e)

    # Redact full phone in response
    if "customer_identifier" in session_data and session_data["customer_identifier"]:
        session_data["customer_identifier_masked"] = mask_phone(session_data["customer_identifier"])
        session_data["customer_identifier"] = mask_phone(session_data["customer_identifier"])

    logger.info(
        "ADMIN_DEBUG_SESSION | session=%s hotel=%s stage=%s",
        session_id,
        hotel_id,
        session.booking_stage,
    )

    return {
        "trace_id": get_trace_id(),
        "session": session_data,
        "activity_logs": activities,
        "interventions": interventions,
        "recent_failures": recent_failures(limit=20, session_id=session_id),
        "hints": {
            "grep_cloud_logs": f'trace_id OR sess={session_id}',
            "metrics": "/api/admin/metrics",
            "failures": "/api/admin/debug/failures",
        },
    }


@router.get("/debug/phone/{phone}")
async def debug_phone_lookup(phone: str, hotel_id: int | None = None):
    """
    Find active session + recent failures for a guest phone (digits).
    Useful when you only have the WhatsApp number from CRM logs.
    """
    from app.config.settings import get_settings
    from app.observability.failure_store import recent_failures
    from app.observability.tracing import get_trace_id, mask_phone
    from app.repositories import session_repository
    from app.utils.phone import digits_only

    digits = digits_only(phone) or phone.strip()
    hid = hotel_id or get_settings().HOTEL_ID
    session = await session_repository.find_active_session(hid, digits)
    return {
        "trace_id": get_trace_id(),
        "phone_masked": mask_phone(digits),
        "hotel_id": hid,
        "active_session_id": session.session_id if session else None,
        "booking_stage": session.booking_stage if session else None,
        "session_debug_url": (
            f"/api/admin/debug/session/{session.session_id}" if session else None
        ),
        "recent_failures": recent_failures(limit=20, hotel_id=hid),
    }

class ContextUpdateRequest(BaseModel):
    hotel_id: str
    session_id: str
    context: Dict[str, Any]

@router.post("/context/update")
async def update_context(request: ContextUpdateRequest):
    """
    Update the bot's active session state for a given guest with new context/summary data.
    """
    if not request.hotel_id or not request.session_id:
        raise HTTPException(status_code=400, detail="hotel_id and session_id are required")
        
    session_col = booking_sessions_collection()
    
    hid = int(request.hotel_id) if request.hotel_id.isdigit() else request.hotel_id
    session = await session_col.find_one({
        "hotel_id": hid,
        "session_id": request.session_id
    })
    
    if not session:
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "error": f"Session not found for hotel_id={request.hotel_id}, session_id={request.session_id}"
            }
        )

    current_overrides = session.get("conversation_overrides", {})
    
    def deep_merge(target, update):
        for key, value in update.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                deep_merge(target[key], value)
            else:
                target[key] = value
        return target
        
    new_overrides = deep_merge(current_overrides, request.context)
    
    await session_col.update_one(
        {"_id": session["_id"]},
        {"$set": {"conversation_overrides": new_overrides}}
    )
    
    updated_keys = list(request.context.keys())
    if "overrides" in request.context and isinstance(request.context["overrides"], dict):
        updated_keys.extend([f"overrides.{k}" for k in request.context["overrides"].keys()])
        
    logger.info(f"CONTEXT_UPDATE | hotel_id={request.hotel_id} | session_id={request.session_id} | keys_updated={updated_keys}")
    
    return {
        "success": True,
        "session_id": request.session_id,
        "message": "Context updated successfully."
    }

class ContextRefreshRequest(BaseModel):
    hotel_id: str | int
    phone_number: str | None = None
    phone: str | None = None
    bot_off_time: str | None = ""
    bot_on_time: str | None = ""
    start_time: str | None = ""
    end_time: str | None = ""


def _parse_hotel_id(raw: str | int) -> int | None:
    """Hotel ids are ints everywhere downstream; a non-numeric id names no hotel."""
    return int(raw) if str(raw).isdigit() else None


def _normalize_phone(raw: str) -> str:
    """Digits only, so WhatsApp and web resolve to the same session key."""
    from app.utils.phone import digits_only
    return digits_only(raw) or raw.strip()


async def save_context_to_session(
    hotel_id: int,
    phone: str,
    summary: str,
    decisions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Save context to active session, or pending_context if no session exists.

    `decisions` are the typed staff decisions from
    `extract_staff_decisions_from_messages`, applied through the same
    `apply_staff_decisions` the takeover path uses. A summary alone is advisory —
    `prompt_summary()` renders the override domains and never `notes`, so a
    summary saying "₹3,000" left the pricing engine free to quote its own number
    while the bot said ₹3,000 in prose.
    """
    from datetime import timedelta
    from app.repositories import session_repository
    from app.db.mongodb import pending_context_collection
    from app.models.schemas import MessageRecord
    from app.models.conversation_overrides import ConversationOverrides
    from app.services.takeover_engine import apply_staff_decisions

    digits = _normalize_phone(phone)
    session = await session_repository.find_active_session(hotel_id, digits)

    if session:
        # Session exists — apply decisions, then record the prose.
        if not getattr(session, "conversation_overrides", None):
            session.conversation_overrides = ConversationOverrides()
        if decisions:
            # The summary rides along: apply_decisions files it in the audit trail
            # and `add_note`, so it must not be added a second time below.
            await apply_staff_decisions(
                session,
                decisions,
                summary=summary,
                source="staff_update",
                reason="CONTEXT_REFRESH",
            )
        else:
            session.conversation_overrides.add_note(f"Staff Summary: {summary}")

        session.messages.append(MessageRecord(
            role="system",
            content=f"SYSTEM NOTE: Human agent interacted with the guest. Summary: {summary}"
        ))
        await session_repository.upsert_session(session)
        logger.info(
            f"CONTEXT_SAVED_TO_SESSION | hotel={hotel_id} | phone={digits} | "
            f"decisions={sorted(decisions or {})}"
        )
        return {"status": "success", "target": "session"}

    # No active session — store in pending_context collection
    pending_col = pending_context_collection()
    now = datetime.now(UTC)
    doc = {
        "hotel_id": hotel_id,
        "phone": digits,
        "staff_summary": summary,
        "created_at": now,
        "expires_at": now + timedelta(hours=24),
    }
    # Carried to hydration, which applies them the same way the session branch
    # does. Absent on the legacy path, so `context_loader` degrades to the note.
    if decisions:
        doc["staff_decisions"] = decisions
    await pending_col.update_one(
        {"hotel_id": hotel_id, "phone": digits},
        {"$set": doc},
        upsert=True,
    )
    logger.info(f"CONTEXT_SAVED_TO_PENDING | hotel={hotel_id} | phone={digits} | reason=no_active_session")
    return {"status": "success", "target": "pending", "note": "Saved to pending — will apply on next guest turn"}


async def _run_context_refresh_background(
    hotel_id: int | str,
    phone: str,
    start_time: str,
    end_time: str
) -> None:
    """Background task: fetch history, generate summary, save to session or pending."""
    import time
    from app.config.settings import get_settings
    from app.services.crm_service import crm_service
    from app.services.gemini_service import summarize_human_chat
    from app.utils.phone import digits_only

    start_t = time.perf_counter()
    digits = digits_only(phone) or phone.strip()
    hid = int(hotel_id) if str(hotel_id).isdigit() else hotel_id

    try:
        # 1. Fetch chat history from CRM
        chats = await crm_service.fetch_chat_history(
            hotel_id=str(hotel_id),
            phone_number=phone,
            start_time=start_time,
            end_time=end_time
        )

        if not chats:
            logger.info(f"CONTEXT_REFRESH_SKIP | hotel={hid} | phone={digits} | reason=no_history")
            return

        # 2. Summarize with Gemini
        summary = await summarize_human_chat(chats)
        logger.info(f"CONTEXT_REFRESH_SUMMARY | hotel={hid} | phone={digits} | summary={summary[:100]}")

        # 2b. Typed decisions, so an agreed price is enforced rather than merely
        # described. Gated: with the flag off this is byte-identical to the old
        # note-only behaviour, because a decision can change what a guest is charged.
        decisions: dict[str, Any] = {}
        if get_settings().CONTEXT_REFRESH_STRUCTURED_DECISIONS:
            from app.services.takeover_engine import extract_staff_decisions_from_messages
            decisions, extracted_summary = await extract_staff_decisions_from_messages(chats)
            # The extractor reads the same window, so its summary is the better
            # one when it produced anything; keep the prose summary otherwise.
            summary = extracted_summary or summary
            logger.info(
                f"CONTEXT_REFRESH_DECISIONS | hotel={hid} | phone={digits} | "
                f"domains={sorted(decisions)}"
            )

        # 3. Save to active session or pending_context
        saved = await save_context_to_session(
            hid if isinstance(hid, int) else 0, digits, summary, decisions
        )

        # 4. Speak.
        #
        # Saving the context was only ever half of "the bot comes back". Without
        # this the guest heard nothing until they happened to write again, which
        # is the reported "the bot doesn't wake up". Only when there is a live
        # session — with none, there is no conversation to continue, and the
        # summary is already parked in `pending_context` for the next guest turn.
        if saved.get("target") == "session":
            try:
                from app.services.takeover_resume_service import resume_after_takeover

                await resume_after_takeover(
                    hotel_id=hid,
                    phone=digits,
                    started_at=start_time,
                )
            except Exception as e:
                logger.error(
                    f"CONTEXT_REFRESH_RESUME_ERROR | hotel={hid} | phone={digits} | error={e}",
                    exc_info=True,
                )

        duration = round(time.perf_counter() - start_t, 2)
        logger.info(f"CONTEXT_REFRESH_DONE | hotel={hid} | phone={digits} | duration={duration:.2f}s")

    except Exception as e:
        logger.error(f"CONTEXT_REFRESH_BG_ERROR | hotel={hid} | phone={digits} | error={e}", exc_info=True)


@router.post("/context-refresh", status_code=202)
async def context_refresh(request: ContextRefreshRequest, background_tasks: BackgroundTasks):
    """
    Immediately returns 202 Accepted.
    Fetches chat history from CRM and updates the bot's memory in the background.
    """
    from app.utils.phone import digits_only

    phone_raw = request.phone_number or request.phone or ""
    digits = digits_only(phone_raw) or phone_raw.strip()
    if not request.hotel_id or not digits:
        raise HTTPException(status_code=400, detail="hotel_id and phone_number required")

    start_time = request.bot_off_time or request.start_time or ""
    end_time = request.bot_on_time or request.end_time or ""
    hid = int(request.hotel_id) if str(request.hotel_id).isdigit() else request.hotel_id

    logger.info(f"CONTEXT_REFRESH_QUEUED | hotel={hid} | phone={digits}")

    background_tasks.add_task(
        _run_context_refresh_background,
        hotel_id=hid,
        phone=digits,
        start_time=start_time,
        end_time=end_time,
    )

    return {
        "status": "accepted",
        "message": "Context refresh queued successfully."
    }

