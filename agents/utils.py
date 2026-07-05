import json
import re
import os
import copy
import functools

def extract_json(text: str) -> dict:
    """
    Attempts to safely extract and parse a JSON object from a string.
    Handles raw JSON, markdown blocks (```json ... ```), and trailing text.
    Returns the parsed dict, or None if parsing fails.
    """
    if not text:
        return None
        
    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Extract from markdown code fences
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Find any JSON-like object string
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None

@functools.lru_cache(maxsize=32)
def _load_json_cached(filename: str) -> dict:
    """Internal cached loader. Returns the raw dict — callers should NOT use this directly."""
    path = os.path.join(os.path.dirname(__file__), "..", "data", filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Utils] ERROR: Could not load {filename}: {e}")
        return {}

def load_json(filename: str) -> dict:
    """
    Loads a JSON file from the data/ directory.
    Returns a deep copy so callers can safely mutate the result
    without corrupting the lru_cache for other callers.
    """
    return copy.deepcopy(_load_json_cached(filename))

