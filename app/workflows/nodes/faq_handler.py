"""
FAQ Handler Node — uses ChromaDB RAG + Gemini for hotel questions.
"""

import logging
from app.rag.chroma_service import query_knowledge
from app.services.gemini_service import answer_faq
from app.workflows.state import ConversationState
from app.repositories.room_repository import get_all_rooms
from app.prompts.templates import text_card

logger = logging.getLogger(__name__)


async def _rewrite_if_persona(text: str, state: dict) -> str:
    """Rewrite text through persona rewriter if a non-default persona is active."""
    hotel_id = state.get("hotel_id", 0)
    if not hotel_id:
        return text
    from app.services.personalization_service import personalization_service
    from app.services.persona_rewriter import rewrite_for_persona
    active_mode = await personalization_service.get_active_mode(str(hotel_id), state)
    if not active_mode or active_mode.is_default or active_mode.mode_id == "default":
        return text
    return await rewrite_for_persona(text, active_mode)


async def handle_faq(state: ConversationState) -> ConversationState:
    """
    Handle FAQ queries using RAG retrieval + Gemini.

    1. Query ChromaDB for relevant knowledge chunks.
    2. Fetch real-time room types from room_types/rate_plans collections.
    3. Pass chunks + room info + question to Gemini for natural answer.
    4. If no context found, return a generic fallback.
    """
    # Room-type pricing questions must show structured Option cards, not a prose FAQ answer
    from app.utils.intent_overrides import get_room_pricing_query
    from app.workflows.nodes.room_query_handler import handle_room_query

    user_message = state.get("user_message", "")
    room_type, nights, num_rooms = get_room_pricing_query(user_message, state)
    if room_type and nights and not state.get("allocation_options"):
        redirect = {
            **state,
            "intent": "BOOKING_SUMMARY",
            "pricing_query_room_type": room_type,
            "pricing_query_nights": nights,
            "room_type_preference": room_type,
        }
        if num_rooms:
            redirect["pricing_query_num_rooms"] = num_rooms
            redirect["num_rooms"] = num_rooms
        return await handle_room_query(redirect)

    question = state.get("faq_question") or user_message
    hotel_id = state.get("hotel_id", 0)

    # Retrieve from ChromaDB with metadata to get similarity scores
    rag_result = query_knowledge(
        query=question,
        hotel_id=hotel_id,
        n_results=3,
        return_metadata=True
    )
    
    rag_context = rag_result["text"] if isinstance(rag_result, dict) else rag_result
    best_score = rag_result["best_score"] if isinstance(rag_result, dict) else float('inf')

    # ── Fetch real-time room info & meal plans from room_types/rate_plans ──
    rooms = await get_all_rooms(hotel_id)
    if rooms:
        from app.services.pricing_engine import PricingEngine
        room_info = "Available Room Types and Meal Plans at the hotel (Rates are FINAL and INCLUSIVE of taxes/GST):\n"
        for r in rooms:
            desc = r.description if r.description else "No description available."
            plans_desc = []
            for rp in r.rate_plans:
                inc_p = PricingEngine.estimate_inclusive_price(rp.base_price)
                plans_desc.append(f"{rp.name}: ₹{inc_p:,.0f}/night")
            plans_str = f" [Rate Plans: {', '.join(plans_desc)}]" if plans_desc else ""
            amens = [k for k, v in r.amenities.items() if v]
            amens_str = f" Amenities: {', '.join(amens)}." if amens else ""
            room_info += f"- {r.name} (Type: {r.room_type}): {desc}. Capacity: {r.occupancy.base_adults} adults, {r.occupancy.base_children} children.{plans_str}{amens_str}\n"
        
        if rag_context:
            rag_context += f"\n\n{room_info}"
        else:
            rag_context = room_info

    # ── Direct MongoDB Knowledge Base Fallback if RAG is sparse ──
    try:
        from app.db.mongodb import knowledge_base_collection
        kb_col = knowledge_base_collection()
        variants = [hotel_id, str(hotel_id)]
        if str(hotel_id).isdigit():
            variants.append(int(hotel_id))

        kb_docs = await kb_col.find({"hotel_id": {"$in": variants}}).to_list(50)
        if kb_docs:
            kb_texts = [f"**{doc.get('title', 'Article')}**\n{doc.get('content', '')}" for doc in kb_docs if doc.get('content')]
            if kb_texts:
                kb_block = "Official Hotel Knowledge Base Articles:\n" + "\n\n---\n\n".join(kb_texts)
                rag_context = (rag_context + f"\n\n{kb_block}") if rag_context else kb_block
    except Exception as e:
        logger.warning(f"Error loading direct KB articles in faq_handler: {e}")

    # ── Inject Hotel Amenities Reference ──
    from app.services.amenity_service import handle_amenity_query
    amenity_cues = ("pool", "swimming", "gym", "fitness", "lift", "elevator", "wifi", "internet", "parking", "spa", "massage", "laundry", "power backup", "generator", "doctor", "wheelchair", "amenit", "facilit")
    q_lower = question.lower()
    is_amenity_inquiry = any(w in q_lower for w in amenity_cues)
    if is_amenity_inquiry:
        try:
            target_kw = "all"
            for cue in ("pool", "gym", "lift", "elevator", "wifi", "parking", "spa", "laundry", "power backup", "doctor", "wheelchair"):
                if cue in q_lower:
                    target_kw = cue
                    break
            amenity_info = await handle_amenity_query(hotel_id, target_kw, ["general"])
            if amenity_info:
                amenity_block = f"Official Hotel Facilities & Amenities Truth:\n{amenity_info}"
                rag_context = (rag_context + f"\n\n{amenity_block}") if rag_context else amenity_block
        except Exception as e:
            logger.warning(f"Error enriching FAQ with amenity info: {e}")

    # ── Inject Official Hotel Policies (Check-in/out, Pets, Smoking, ID Rules) ──
    from app.repositories import hotel_repository
    try:
        formatted_policies = await hotel_repository.get_formatted_hotel_policies(hotel_id)
        if formatted_policies:
            rag_context = (rag_context + f"\n\n{formatted_policies}") if rag_context else formatted_policies
    except Exception as e:
        logger.warning(f"Error enriching FAQ with hotel policies: {e}")

    # ── Inject Property Overview & Transit / Commute Context ──
    try:
        details_doc = await hotel_repository.get_hotel_details(hotel_id)
        if details_doc:
            prop = details_doc.get("property") or {}
            desc = prop.get("property_description")
            star = prop.get("star_rating")
            transit_list = details_doc.get("transit_directions") or []

            prop_lines = []
            if star or desc:
                prop_lines.append("Hotel Profile & Overview:")
                if star:
                    prop_lines.append(f"• Star Rating: {star}")
                if desc:
                    prop_lines.append(f"• Description: {desc}")
            if transit_list and isinstance(transit_list, list):
                prop_lines.append("Commute & Transit Options:")
                for t in transit_list[:3]:
                    cat = t.get("transit_category") or "Transit"
                    orig = t.get("origin_name") or ""
                    tm = t.get("approximate_commute_minutes")
                    dist = t.get("approximate_distance_km")
                    trans = t.get("recommended_transport") or "Taxi"
                    dirs = t.get("directions") or ""
                    url = t.get("google_maps_route_url") or ""
                    prop_lines.append(f"• From {cat} ({orig}): ~{tm} mins, {dist} km via {trans}. {dirs} {url}".strip())

            if prop_lines:
                prop_block = "\n".join(prop_lines)
                rag_context = (rag_context + f"\n\n{prop_block}") if rag_context else prop_block
    except Exception as e:
        logger.warning(f"Error enriching FAQ with hotel details: {e}")

    # ── Inject Current Booking State ──
    from app.workflows.session_mapper import state_to_session
    session = state_to_session(state)
    booking_summary = session.get_booking_context_summary()
    booking_context_str = f"CURRENT USER'S BOOKING STATE (Use this to answer questions about their booking):\n{booking_summary}"

    allocation_options = state.get("allocation_options", [])
    if allocation_options and state.get("check_in") and state.get("check_out"):
        from app.services.booking_engine import calculate_num_nights
        nights = calculate_num_nights(state["check_in"], state["check_out"])
        options_lines = [
            f"\nROOM OPTIONS CURRENTLY SHOWN TO THE GUEST ({nights} night(s), {state['check_in']} to {state['check_out']}):"
        ]
        for i, opt in enumerate(allocation_options, 1):
            from app.services.pricing_engine import PricingEngine
            per_night = float(opt.get("total_price_per_night_inclusive") or PricingEngine.estimate_inclusive_price(opt.get("total_price_per_night", 0)))
            per_night = float(round(per_night / 10) * 10)
            total = float(round((per_night * nights) / 10) * 10)
            room_names = ", ".join(
                f"{r.get('room_name', 'Room')}" for r in opt.get("rooms", [])
            )
            options_lines.append(
                f"- Option {i}: {room_names} — ₹{per_night:,.0f}/night, approx ₹{total:,.0f} for {nights} night(s) (incl. GST)"
            )
        booking_context_str += "\n" + "\n".join(options_lines)
    
    if rag_context:
        rag_context += f"\n\n{booking_context_str}"
    else:
        rag_context = booking_context_str

    if not rag_context:
        fallback_text = "I don't have that information right now. Our front desk team would be happy to help — feel free to contact them! 📞"
        fallback_text = await _rewrite_if_persona(fallback_text, state)
        return {
            **state,
            "rag_context": None,
            "bot_response": text_card(fallback_text),
            "next_action": "faq_answered",
        }
        
    # ── Knowledge Gap Detection ──
    from app.config.settings import get_settings
    settings = get_settings()
    
    # If the distance is higher than the threshold, it means the query is not similar enough to our KB.
    # Higher cosine distance = less similar.
    is_room_or_recommendation_query = any(w in question.lower() for w in (
        "kaun sa", "kon sa", "which", "recommend", "suggest", "room", "kamra", "option", "rate", "price", "kitna", "best", "comfort"
    )) and bool(state.get("allocation_options"))

    if best_score > settings.KNOWLEDGE_SIMILARITY_THRESHOLD and not is_room_or_recommendation_query and not is_amenity_inquiry:
        from app.services.knowledge_gap_engine import verify_and_log_knowledge_gap
        session_id = state.get("session_id", "unknown")
        
        is_gap = await verify_and_log_knowledge_gap(
            hotel_id=hotel_id,
            session_id=session_id,
            question=question,
            rag_context=rag_context,
            best_score=best_score
        )
        
        if is_gap:
            gap_text = "I'm not entirely sure about that right now, but I have noted down your question so our team can provide you with the correct information soon. Please feel free to ask the front desk directly! 📞"
            gap_text = await _rewrite_if_persona(gap_text, state)
            return {
                **state,
                "rag_context": rag_context,
                "bot_response": text_card(gap_text),
                "next_action": "faq_answered",
            }

    hotel_name = state.get("hotel_name", "our hotel")

    # Generate natural answer using Gemini + RAG context
    answer = await answer_faq(
        question=question,
        rag_context=rag_context,
        hotel_name=hotel_name,
        hotel_id=hotel_id,
        session_id=state.get("session_id", ""),
    )

    booking_stage = state.get("booking_stage", "new")
    if booking_stage == "cancellation_pending":
        booking_id = state.get("booking_id", "your booking")
        answer += (
            f"\n\nDo you still wish to proceed with the cancellation of booking **{booking_id}**? "
            f"Please reply with **'Yes, cancel'** to proceed, or **'No, keep booking'** to abort."
        )
    else:
        from app.utils.booking_resume import append_booking_resume
        answer = append_booking_resume(answer, state)

    return {
        **state,
        "rag_context": rag_context,
        "bot_response": text_card(answer),
        "next_action": "faq_answered",
    }
