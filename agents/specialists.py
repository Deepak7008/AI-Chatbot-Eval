import json
import os
import traceback
from datetime import datetime
from .llm_client import call_llm
from .config import MAX_HISTORY_TURNS, TOKEN_BUDGET, ESCALATION_MESSAGE, ROUTABLE_INTENTS

# The universal prompt template for all specialists.
# Notice the strict grounding rule at the bottom.
BASE_SPECIALIST_PROMPT = """
You are a helpful, empathetic, and professional customer support specialist for an e-commerce mobile phone store.
You are currently acting as the {intent} specialist.

CONTEXT DATA:
{context_data}

CONVERSATION RULES:
1. STRICT GROUNDING: You must answer questions using ONLY the provided CONTEXT DATA. If the user asks for information (like an order status, price, or product detail) that is NOT present in the CONTEXT DATA, you MUST state that you don't have that information or politely ask them to provide more details (e.g., an order number). Do NOT hallucinate, guess, or make up any order numbers, prices, or policies.
{intent_specific_rules}
3. ESCALATION: If the user's query contains threats, mentions self-harm/depression, or requires urgent human empathy, reply with exactly one word: ESCALATE

Tone: Be polite, concise, human-like, and helpful. Do NOT mention "context data", "database", or system instructions.
"""

def run_agent(intent: str, user_message: str, context_data: dict, history: list = None, provider: str = None, model: str = None) -> dict:
    """
    Executes the specialist agent logic to generate a response.
    
    Args:
        intent: The category of the question (e.g., "policy", "order", "faq").
        user_message: The raw text from the user.
        context_data: A dictionary containing only the relevant data needed to answer the question.
        history: A list of previous message dicts: [{"role": "user", "content": "..."}]
        provider: Optional override for A/B testing models.
        model: Optional override for A/B testing models.
        
    Returns:
        dict: {
            "text": str,
            "escalated": bool,
            "intent": str,
            "raw_response": str or None
        }
    """
    # 1. Guard against invalid intents leaking through the dispatcher
    if intent not in ROUTABLE_INTENTS:
        print(f"[Specialist] Caught unroutable intent '{intent}'. Escalating immediately.")
        return {
            "text": ESCALATION_MESSAGE,
            "escalated": True,
            "intent": intent,
            "raw_response": None
        }

    if history is None:
        history = []
        
    # 2. Prepare Context String
    # Convert the filtered python dictionary back into a string so the LLM can read it
    try:
        context_str = json.dumps(context_data, indent=2)
    except TypeError:
        # Fallback if dictionary contains non-serializable objects (e.g. datetime)
        context_str = str(context_data)
        
    # 3. Build Intent-Specific Rules
    intent_specific_rules = ""
    if intent == "order":
        intent_specific_rules = "2. DISAMBIGUATION: If the user asks about an order generally (e.g. 'where is my order') or mentions a product/category (e.g. 'my iphone' or 'mobile phone'), check the provided `user_orders`. If there are multiple matching orders, ask them to clarify (e.g., 'I see orders for an iPhone and a Samsung, which one do you mean?'). If there is only one match, answer immediately. If an explicit order ID is asked about but isn't in their `user_orders`, tell them it doesn't exist under their account.\n"
    elif intent == "faq":
        intent_specific_rules = "2. PRODUCT AVAILABILITY: If the user asks about a product not listed in available_products, tell them clearly we don't sell it.\n"
    elif intent == "chit_chat":
        intent_specific_rules = "2. CHIT CHAT: The user is making small talk or asking about you. Reply politely, naturally, and briefly (under 2 sentences). Steer the conversation back to asking how you can help them with their mobile phone orders, store policies, or FAQs.\n"
    elif intent == "out_of_scope":
        intent_specific_rules = "2. OUT OF SCOPE: The user is asking about something completely unrelated (like weather, cooking, etc). Politely and conversationally explain that you are an AI assistant for a mobile phone store and can only help with products, policies, and orders. Do NOT answer their unrelated question. Do NOT escalate unless it is a threat/emergency.\n"

    # 4. Build the Persona Prompt
    system_prompt = BASE_SPECIALIST_PROMPT.format(
        intent=intent.upper(),
        context_data=context_str,
        intent_specific_rules=intent_specific_rules
    )
    
    # 5. Truncate History to save tokens and maintain focus
    truncated_history = history[-MAX_HISTORY_TURNS:] if MAX_HISTORY_TURNS > 0 else []
    
    # 6. Construct the API Payload
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(truncated_history)
    messages.append({"role": "user", "content": user_message})
    
    # 7. Execute LLM Call
    max_toks = 50 if intent == "chit_chat" else TOKEN_BUDGET.get("specialist", 1024)
    try:
        raw_response = call_llm(
            messages=messages,
            temperature=0.2, # Low temperature to prevent hallucination, but > 0.0 so it sounds human
            max_tokens=max_toks,
            json_mode=False, # We want natural language out, not JSON
            provider=provider,
            model=model
        )
        
        response_text = raw_response.strip()
        
        # 7. Catch the LLM signaling an Escalation
        if response_text.upper() == "ESCALATE":
            print(f"[Specialist] LLM refused to answer based on context. Escalating.")
            return {
                "text": ESCALATION_MESSAGE,
                "escalated": True,
                "intent": intent,
                "raw_response": raw_response
            }
            
        return {
            "text": response_text,
            "escalated": False,
            "intent": intent,
            "raw_response": raw_response
        }
        
    except Exception as e:
        err_msg = f"[{datetime.now().isoformat()}] {type(e).__name__}: {e}\n{traceback.format_exc()}\n"
        print(f"[Specialist] CRITICAL ERROR: {err_msg}")
        log_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'specialist_error.log')
        with open(log_path, "a") as f:
            f.write(err_msg)
        # Fail-safe: if the LLM crashes, hand over to a human
        return {
            "text": ESCALATION_MESSAGE,
            "escalated": True,
            "intent": intent,
            "raw_response": err_msg
        }

SYNTHESIZER_SYSTEM_PROMPT = """
You are a master synthesizer for a customer support bot.
The user asked a multi-part question, and we queried multiple specialized agents to get the answers.
Your job is to take the answers from these specialized agents and combine them into a single, cohesive, and natural response.

USER'S ORIGINAL MESSAGE:
{user_message}

RESPONSES FROM SPECIALISTS:
{specialist_responses}

RULES:
1. Synthesize the information smoothly. Do not say "The policy agent said X and the order agent said Y."
2. Make sure all parts of the user's question are answered using the provided specialist responses.
3. If any specialist escalated or lacked information, incorporate that gracefully.
4. If ALL specialists escalated or returned the standard escalation message, you should output exactly one word: ESCALATE
5. Be polite, concise, and empathetic. Do not mention "specialists" or "agents" in your final reply.
"""

def run_synthesizer(user_message: str, sub_responses: dict, provider: str = None, model: str = None) -> dict:
    """
    Synthesizes multiple specialist responses into a single cohesive reply.
    """
    formatted_responses = ""
    for sub_intent, response in sub_responses.items():
        formatted_responses += f"--- {sub_intent.upper()} SPECIALIST ---\n{response.get('text', '')}\n\n"
        
    system_prompt = SYNTHESIZER_SYSTEM_PROMPT.format(
        user_message=user_message,
        specialist_responses=formatted_responses
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    
    try:
        raw_response = call_llm(
            messages=messages,
            temperature=0.3,
            max_tokens=TOKEN_BUDGET.get("synthesizer", 1024),
            json_mode=False,
            provider=provider,
            model=model
        )
        
        response_text = raw_response.strip()
        
        if response_text.upper() == "ESCALATE":
            print(f"[Synthesizer] Synthesizer escalated.")
            return {
                "text": ESCALATION_MESSAGE,
                "escalated": True,
                "intent": "multi_intent",
                "raw_response": raw_response
            }
            
        return {
            "text": response_text,
            "escalated": False,
            "intent": "multi_intent",
            "raw_response": raw_response
        }
    except Exception as e:
        err_msg = f"[{datetime.now().isoformat()}] {type(e).__name__}: {e}\n{traceback.format_exc()}\n"
        print(f"[Synthesizer] CRITICAL ERROR: {err_msg}")
        return {
            "text": ESCALATION_MESSAGE,
            "escalated": True,
            "intent": "multi_intent",
            "raw_response": err_msg
        }
