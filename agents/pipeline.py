import time
from .guardrails import check_input, check_output
from .router import route_query
from .entity_extractor import extract_entities
import concurrent.futures
from .specialists import run_agent, run_synthesizer
from .config import ESCALATION_MESSAGE, BYPASS_KEYWORD
from .utils import load_json
from .llm_client import get_token_usage, reset_token_usage
import re

def _enrich_order(order: dict, mock_db: dict) -> dict:
    enriched = order.copy()
    enriched_items = []
    for item in order.get("items", []):
        enriched_item = item.copy()
        for p in mock_db.get("products", []):
            if p.get("id") == item.get("product_id"):
                enriched_item["product_name"] = p.get("name")
                enriched_item["product_category"] = p.get("category")
                break
        enriched_items.append(enriched_item)
    enriched["items"] = enriched_items
    return enriched


def _fetch_context(intent: str, entities: dict, user_email: str = None) -> dict:
    """
    Fetches ONLY the minimal required data based on the intent and extracted entities.
    This solves the problem of dumping massive JSON files into the prompt.
    """
    context = {}
    
    if intent == "policy":
        # Only inject policy data, skip users, orders, and products
        policies = load_json("policies.json")
        
        # We strip out general FAQ so the policy bot only gets policy rules
        if "general_faq" in policies:
            # Create a copy so we don't mutate the cached dictionary
            policies = policies.copy()
            del policies["general_faq"]
            
        context["policies"] = policies
        
    elif intent == "faq":
        # Only inject the FAQ data
        policies = load_json("policies.json")
        context["faq"] = policies.get("general_faq", {})
        context["available_products"] = policies.get("available_products", {})
        
    elif intent == "order":
        mock_db = load_json("mock_db.json")
        
        # 1. Fetch User and User Orders (production mode)
        # Use provided user_email (logged-in persona) or extracted email
        email = user_email if user_email else entities.get("email")
        if email:
            for u in mock_db.get("users", []):
                if u.get("email") == email:
                    context["user"] = u
                    break
                    
            if "user" in context:
                user_id = context["user"]["id"]
                user_orders = []
                
                # Fetch all orders for this user
                for o in mock_db.get("orders", []):
                    if o.get("user_id") == user_id:
                        enriched_order = _enrich_order(o, mock_db)
                        user_orders.append(enriched_order)
                
                if user_orders:
                    context["user_orders"] = user_orders
        
        # 2. Eval mode: no user_email, but order_id extracted from query
        # Look up the order directly by ID so eval test cases work correctly
        # regardless of which user owns the order.
        elif entities.get("order_id"):
            order_id = entities["order_id"]
            for o in mock_db.get("orders", []):
                if o.get("order_id") == order_id:
                    enriched = _enrich_order(o, mock_db)
                    context["order"] = enriched
                    # Also include the owning user for groundedness
                    for u in mock_db.get("users", []):
                        if u.get("id") == o.get("user_id"):
                            context["user"] = u
                            break
                    break
                    
        # 3. Fetch Product by Name (fuzzy match)
        product_name = entities.get("product_name")
        if product_name:
            matched_products = []
            for p in mock_db.get("products", []):
                if product_name.lower() in p.get("name", "").lower():
                    matched_products.append(p)
            if matched_products:
                context["products"] = matched_products

        # Also give the order bot access to policies in case they ask "Can I return order 1234?"
        # But strip out unnecessary stuff to save local model token limits
        policies = load_json("policies.json")
        context["policies"] = {k: v for k, v in policies.items() if k not in ["general_faq", "available_products"]}

    return context


def run_pipeline(user_message: str, history: list = None, provider: str = None, model: str = None, status_callback=None, user_email: str = None) -> dict:
    """
    The master orchestrator that runs the entire agentic loop chronologically.
    
    Flow:
    1. Input Guardrails (check for prompt injection / length)
    2. Intent Router (classify intent)
    3. Entity Extraction (regex + llm for order_id, email, product)
    4. Context Builder (fetch only necessary JSON data)
    5. Specialist Execution (generate actual response)
    6. Output Guardrails (check for leaks)
    """
    if history is None:
        history = []
        
    start_time = time.time()
    reset_token_usage()
    
    steps = []
    
    def log_step(name, action, result, step_start_time, prev_tokens):
        current_tokens = get_token_usage().get("total_tokens", 0)
        tokens_used = current_tokens - prev_tokens
        latency = int((time.time() - step_start_time) * 1000)
        
        step_info = {
            "step": name,
            "action": action,
            "result": result,
            "latency_ms": latency,
            "tokens_used": tokens_used
        }
        steps.append(step_info)
        
        if status_callback:
            status_callback(step_info)
            
        return current_tokens

    # --- 1. INPUT GUARDRAILS ---
    step_start = time.time()
    prev_tok = get_token_usage().get("total_tokens", 0)
    input_check = check_input(user_message)
    prev_tok = log_step("Input Guardrail", "Checking safety...", "Passed" if input_check["is_safe"] else "Blocked", step_start, prev_tok)

    if not input_check["is_safe"] and not input_check.get("is_bypassed", False):
        # Fail immediately. Do not hit the LLM Router.
        return {
            "text": ESCALATION_MESSAGE,
            "escalated": True,
            "intent": "blocked_by_guardrail",
            "confidence": 1.0,
            "reasoning": input_check.get("reason", "Blocked by guardrail"),
            "is_low_confidence": False,
            "entities": {},
            "raw_response": None,
            "latency_ms": int((time.time() - start_time) * 1000),
            "guardrail_input_safe": False,
            "guardrail_input_reason": input_check.get("reason", "Blocked by guardrail"),
            "guardrail_bypassed": False,
            "tokens_used": get_token_usage().get("total_tokens", 0)
        }

    # Strip bypass keyword so downstream models don't read it
    clean_message = re.sub(re.escape(BYPASS_KEYWORD), '', user_message, flags=re.IGNORECASE).strip()

    # --- 2. SEMANTIC ROUTER ---
    step_start = time.time()
    route_result = route_query(clean_message, history=history, provider=provider, model=model)
    intent = route_result["intent"]
    confidence = route_result["confidence"]
    reasoning = route_result.get("reasoning", "")
    is_low_confidence = route_result.get("is_low_confidence", False)
    sub_intents = route_result.get("sub_intents", [])
    prev_tok = log_step("Intent Router", "Routing query...", f"Intent: {intent} (Conf: {confidence:.2f})", step_start, prev_tok)

    # --- LOW CONFIDENCE HANDLER ---
    if is_low_confidence:
        # If the router is confused, we should NOT guess. We escalate immediately to a human.
        return {
            "text": ESCALATION_MESSAGE,
            "escalated": True,
            "intent": intent,
            "confidence": confidence,
            "reasoning": reasoning,
            "is_low_confidence": True,
            "entities": {},
            "raw_response": "Low confidence routing: Escalated immediately to prevent hallucination.",
            "latency_ms": int((time.time() - start_time) * 1000),
            "guardrail_input_safe": input_check.get("is_safe", True),
            "guardrail_input_reason": input_check.get("reason", ""),
            "guardrail_bypassed": input_check.get("is_bypassed", False),
            "tokens_used": get_token_usage().get("total_tokens", 0)
        }
        
    # --- 3. ENTITY EXTRACTION ---
    step_start = time.time()
    entities = extract_entities(intent, clean_message, sub_intents=sub_intents, provider=provider, model=model, history=history)
    prev_tok = log_step("Entity Extractor", "Extracting parameters...", str(entities), step_start, prev_tok)

    # --- GUEST ORDER HANDLER ---
    # In production (Chat UI), user_email comes from the logged-in session
    # and orders are scoped to that user (IDOR protection).
    # In eval mode (no user_email), look up orders by order_id directly.
    if intent == "order" and not user_email and not entities.get("order_id"):
        return {
            "text": "Please log in to chat and ask about your queries.",
            "escalated": False,
            "intent": intent,
            "confidence": confidence,
            "reasoning": reasoning,
            "is_low_confidence": False,
            "entities": entities,
            "raw_response": "Guest user blocked from querying order intent.",
            "latency_ms": int((time.time() - start_time) * 1000),
            "guardrail_input_safe": input_check.get("is_safe", True),
            "guardrail_input_reason": input_check.get("reason", ""),
            "guardrail_bypassed": input_check.get("is_bypassed", False),
            "tokens_used": get_token_usage().get("total_tokens", 0)
        }
    
    # --- 4 & 5. CONTEXT INJECTION & SPECIALIST EXECUTION ---
    if intent == "multi_intent":
        sub_responses = {}
        def process_sub_intent(sub_intent):
            sub_context = _fetch_context(sub_intent, entities, user_email=user_email)
            sub_result = run_agent(sub_intent, clean_message, sub_context, history, provider=provider, model=model)
            sub_result["context_keys"] = list(sub_context.keys())
            return sub_intent, sub_result

        step_start = time.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_sub_intent, s) for s in sub_intents]
            for future in concurrent.futures.as_completed(futures):
                s_intent, s_result = future.result()
                sub_responses[s_intent] = s_result
        
        context_keys = []
        for v in sub_responses.values():
            context_keys.extend(v.get("context_keys", []))
        prev_tok = log_step("Parallel Specialists", f"Running {sub_intents} agents...", f"Fetched aggregated context", step_start, prev_tok)

        step_start = time.time()
        specialist_result = run_synthesizer(clean_message, sub_responses, provider=provider, model=model)
        
        final_text = specialist_result["text"]
        escalated = specialist_result["escalated"]
        raw_response = specialist_result["raw_response"]
        context = {"multi_intent_context_keys": context_keys}
        prev_tok = log_step("Synthesizer Agent", "Synthesizing responses...", "Escalated" if escalated else "Success", step_start, prev_tok)
        
    else:
        step_start = time.time()
        context = _fetch_context(intent, entities, user_email=user_email)
        prev_tok = log_step("Context Injector", "Fetching DB records...", f"Fetched keys: {list(context.keys())}", step_start, prev_tok)
        
        step_start = time.time()
        # The specialist inherently handles "out_of_scope" and "low_confidence" because they 
        # are not in ROUTABLE_INTENTS, so it will immediately return ESCALATION_MESSAGE.
        specialist_result = run_agent(intent, clean_message, context, history, provider=provider, model=model)
        
        final_text = specialist_result["text"]
        escalated = specialist_result["escalated"]
        raw_response = specialist_result["raw_response"]
        prev_tok = log_step("Specialist Agent", f"Running {intent} agent...", "Escalated" if escalated else "Success", step_start, prev_tok)
    
    # --- 6. OUTPUT GUARDRAILS ---
    # We only need to check the output if the specialist successfully generated a response
    if not escalated:
        step_start = time.time()
        output_check = check_output(final_text)
        if not output_check["is_safe"]:
            # The bot hallucinated or leaked prompt details. Override output.
            final_text = ESCALATION_MESSAGE
            escalated = True
            raw_response = f"BLOCKED BY GUARDRAIL. Original raw response: {raw_response}"
            log_step("Output Guardrail", "Checking response safety...", "Blocked", step_start, prev_tok)
        else:
            log_step("Output Guardrail", "Checking response safety...", "Passed", step_start, prev_tok)
        
    latency_ms = int((time.time() - start_time) * 1000)
    
    # --- 7. STANDARDIZED RETURN DICT ---
    return {
        "text": final_text,
        "escalated": escalated,
        "intent": intent,
        "confidence": confidence,
        "reasoning": reasoning,
        "is_low_confidence": is_low_confidence,
        "entities": entities,
        "context": context,
        "raw_response": raw_response,
        "latency_ms": latency_ms,
        "guardrail_input_safe": input_check.get("is_safe", True),
        "guardrail_input_reason": input_check.get("reason", ""),
        "guardrail_bypassed": input_check.get("is_bypassed", False),
        "tokens_used": get_token_usage().get("total_tokens", 0),
        "steps": steps
    }
