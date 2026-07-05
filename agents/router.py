import json
import re
from .llm_client import call_llm
from .config import VALID_INTENTS, ROUTABLE_INTENTS, LOW_CONFIDENCE_THRESHOLD
from .utils import extract_json

ROUTER_SYSTEM_PROMPT = """
You are a semantic router for a mobile phone customer support bot.
Your job is to route the user's message to the correct specialist agent.

Categories:
1. "policy": Questions about returns, refunds, shipping times/costs, and warranties.
2. "order": Questions about specific order statuses(including returns, refunds for a specific order), tracking numbers, or cancelling orders.
3. "faq": General store questions like hours, payment methods, trade-ins, or unlocked phones.
4. "chit_chat": Harmless small talk, greetings (e.g. "hi", "how are you"), or questions about the bot itself (e.g. "who are you?", "are you a robot?").
5. "out_of_scope": The question is completely unrelated to the mobile phone store
   (e.g., weather, cooking, general knowledge). Do NOT use this for edge cases — 
   only use it when the question clearly has nothing to do with mobile phones or orders.

If a user asks a multi-intent question (e.g., "Cancel my order and what's the return policy?"), 
output "multi_intent" as the intent, and provide a list of sub_intents (e.g., ["order", "policy"]).

You will receive recent conversation history. Use it to understand short/vague replies (e.g., if the user just says "1043" after the bot asked for an Order ID, the intent is still "order"). Always classify the intent of the LATEST user message.

You must output a JSON object with exactly these fields:
{
  "intent": "policy" | "order" | "faq" | "chit_chat" | "out_of_scope" | "multi_intent",
  "sub_intents": ["list", "of", "intents"] (only required if intent is "multi_intent"),
  "confidence": 0.0 to 1.0,
  "reasoning": "Brief explanation of why you chose this intent"
}

Do NOT wrap the JSON in markdown code fences. Output raw JSON only.
"""


def _clamp(value, min_val=0.0, max_val=1.0):
    """Clamps a number between min and max. Prevents LLM returning 1.5 or -0.2."""
    return max(min_val, min(max_val, value))


def route_query(user_message, history=None, provider=None, model=None):
    """
    Determines the intent of the user message.

    Args:
        user_message: The raw user input string.
        history: A list of previous message dicts for context.
        provider: Optional LLM provider override (e.g., "groq", "gemini").
        model: Optional model override (e.g., "llama-3.3-70b-versatile").
               Useful for A/B testing routing accuracy across models during evals.

    Returns:
        dict with keys:
            intent (str): "policy", "order", "faq", or "out_of_scope"
            confidence (float): 0.0 to 1.0, clamped
            reasoning (str): Why this intent was chosen
            is_low_confidence (bool): True if confidence < threshold
            sub_intents (list): List of sub-intents if intent is "multi_intent"
            raw_response (str | None): The raw LLM output (for debugging/logging)
    """
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT}
    ]
    
    # Inject minimal history to give context to short replies like "1044"
    if history:
        # Only grab the last 2-3 turns to prevent confusing the router with old intents
        messages.extend(history[-3:])
        
    messages.append({"role": "user", "content": user_message})

    try:
        raw_response = call_llm(
            messages,
            temperature=0.0,
            max_tokens=150,
            json_mode=True,
            provider=provider,
            model=model
        )

        result = extract_json(raw_response)

        if result is None:
            print(f"[Router] ERROR: Could not parse JSON from LLM response:\n{raw_response}")
            return {
                "intent": "out_of_scope",
                "confidence": 0.0,
                "reasoning": "Failed to parse router response as JSON. Escalating to human agent.",
                "is_low_confidence": True,
                "raw_response": raw_response
            }

        intent = result.get("intent", "").lower().strip()
        sub_intents = [s.lower().strip() for s in (result.get("sub_intents") or [])]
        confidence = result.get("confidence", 0.5)
        reasoning = result.get("reasoning", "No reasoning provided.")

        # Validate intent — if invalid, reset confidence to 0.0 (not the LLM's hallucinated score)
        if intent not in VALID_INTENTS:
            print(f"[Router] WARNING: LLM returned invalid intent '{intent}'. "
                  f"Falling back to 'out_of_scope' with confidence=0.0")
            intent = "out_of_scope"
            confidence = 0.0
            reasoning = f"Invalid intent from LLM ('{result.get('intent')}'), escalating to human agent."
            
        # Validate sub_intents if multi_intent
        if intent == "multi_intent":
            valid_sub_intents = [s for s in sub_intents if s in VALID_INTENTS and s not in ["out_of_scope", "chit_chat", "multi_intent"]]
            if not valid_sub_intents:
                print(f"[Router] WARNING: LLM returned multi_intent but no valid sub_intents: {sub_intents}. "
                      f"Falling back to 'out_of_scope' with confidence=0.0")
                intent = "out_of_scope"
                sub_intents = []
                confidence = 0.0
                reasoning = "Invalid sub_intents from LLM, escalating to human agent."
            else:
                sub_intents = valid_sub_intents

        # Clamp confidence to [0.0, 1.0]
        try:
            confidence = _clamp(float(confidence))
        except (ValueError, TypeError):
            print(f"[Router] WARNING: Could not parse confidence '{confidence}'. Defaulting to 0.0")
            confidence = 0.0

        is_low_confidence = confidence < LOW_CONFIDENCE_THRESHOLD

        if is_low_confidence:
            print(f"[Router] LOW CONFIDENCE ({confidence:.2f}): intent='{intent}', "
                  f"reasoning='{reasoning}'")

        return {
            "intent": intent,
            "confidence": confidence,
            "reasoning": reasoning,
            "is_low_confidence": is_low_confidence,
            "sub_intents": sub_intents,
            "raw_response": raw_response
        }

    except Exception as e:
        print(f"[Router] CRITICAL ERROR: {type(e).__name__}: {e}")
        return {
            "intent": "out_of_scope",
            "confidence": 0.0,
            "reasoning": f"Router crashed: {type(e).__name__}: {e}. Escalating to human agent.",
            "is_low_confidence": True,
            "raw_response": None
        }
