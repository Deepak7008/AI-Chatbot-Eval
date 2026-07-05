import re
from .llm_client import call_llm
from .config import TOKEN_BUDGET, ROUTABLE_INTENTS
from .utils import extract_json

EXTRACTOR_SYSTEM_PROMPT = """
You are an entity extraction module for an e-commerce store.
Your ONLY job is to extract specific product names or general product types from the user's message.

Output your findings as a strict JSON object:
{"product_name": "extracted_name_or_null"}

Examples:
- "Where is my iPhone 15?" -> {"product_name": "iPhone 15"}
- "My phone arrived broken." -> {"product_name": "phone"}
- "Where is my order ORD-1234?" -> {"product_name": null}
- "Can I return a charger?" -> {"product_name": "charger"}
- "Do you sell the Samsung Galaxy S25?" -> {"product_name": "Samsung Galaxy S25"}
- "Do you have the iPhone 16 Pro in stock?" -> {"product_name": "iPhone 16 Pro"}
- "Is the Google Pixel 9 available?" -> {"product_name": "Google Pixel 9"}

Output raw JSON only. Do not add any formatting or markdown.
"""

def extract_entities(intent: str, user_message: str, sub_intents: list = None, provider: str = None, model: str = None, history: list = None) -> dict:
    """
    Extracts structured entities (order_id, email, product_name) from the user's message.
    Uses fast regex first, and calls the LLM to extract complex entities like product names.
    If an order_id is missing from the current message, it will search the provided history.
    
    Returns:
        dict: {"order_id": str|None, "email": str|None, "product_name": str|None}
    """
    entities = {
        "order_id": None,
        "email": None,
        "product_name": None
    }
    
    if intent not in ROUTABLE_INTENTS:
        return entities  # No extraction needed for out_of_scope or unknown intents
    
    # 1. FAST REGEX EXTRACTION (Free & Instant)
    # Regex extractions are cheap, so we run them for ALL intents.
    # This catches emails and order IDs even in policy/faq questions.

    # Email: basic pattern to catch standard emails
    if intent == "order" or (intent == "multi_intent" and sub_intents and "order" in sub_intents):
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', user_message)
        if email_match:
            entities["email"] = email_match.group(0)
        
    # Order ID: Case-insensitive, handles spaces (e.g. "ord 1234", "ORD-5555", "#1042")
    order_match = re.search(r'(?:ord[\s-]?|#)(\d{4,})', user_message, re.IGNORECASE)
    
    # Fallback: bare number with context words (e.g. "order number is 10423")
    if not order_match:
        order_match = re.search(
            r'\border\s+(?:number\s+|id\s+|#\s*)?(\d{4,})\b',
            user_message,
            re.IGNORECASE
        )
        
    # Final Fallback: If the user just typed a 4+ digit number anywhere in the message (e.g. "Its 1044")
    # Since the intent router already confidently determined this is an "order" intent, 
    # any 4+ digit number is highly likely to be the order ID.
    if not order_match:
        order_match = re.search(r'\b(\d{4,})\b', user_message)

    if order_match:
        # Standardize format to match our DB keys (e.g. ORD-1042)
        entities["order_id"] = f"ORD-{order_match.group(1)}"
    elif history:
        # Check history if we didn't find an order_id in the current message
        for msg in reversed(history):
            if msg.get("role") == "user":
                h_match = re.search(r'(?:ord[\s-]?|#)(\d{4,})', msg.get("content", ""), re.IGNORECASE)
                if not h_match:
                    h_match = re.search(r'\border\s+(?:number\s+|id\s+|#\s*)?(\d{4,})\b', msg.get("content", ""), re.IGNORECASE)
                if not h_match:
                    h_match = re.search(r'\b(\d{4,})\b', msg.get("content", ""))
                
                if h_match:
                    entities["order_id"] = f"ORD-{h_match.group(1)}"
                    break

    # 2. LLM EXTRACTION (Smart fallback for fuzzy entities)
    # Product names are too varied for regex, so we use the LLM.
    # Only needed for 'order' and 'policy' intents — FAQ/out_of_scope don't need product context.
    if intent == "order" or (intent == "multi_intent" and sub_intents and "order" in sub_intents):
        messages = [
            {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
        
        try:
            raw_response = call_llm(
                messages=messages,
                temperature=0.0,  # 0.0 for deterministic extraction
                max_tokens=TOKEN_BUDGET.get("extractor", 128),
                json_mode=True,
                provider=provider,
                model=model
            )
            
            # Use our shared utility to safely parse JSON
            extracted_data = extract_json(raw_response) or {}
                    
            # Merge the LLM's findings into our main dictionary
            entities["product_name"] = extracted_data.get("product_name", None)
            
        except Exception as e:
            print(f"[Entity Extractor] API Error: {e}")
            # If the LLM fails, we FAIL GRACEFULLY. 
            # We don't crash, because we might already have the order_id/email from regex!
            pass
        
    return entities
