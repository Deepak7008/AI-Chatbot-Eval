"""
llm_client.py -- Unified LLM calling interface with provider abstraction and token tracking.

This module is the single point of contact for ALL LLM calls in the system.
It handles:
  1. Provider abstraction (Groq, Gemini, OpenRouter, Ollama, Mock)
  2. Token tracking (session-level usage accumulation)
  3. Retry with exponential backoff (resilience to rate limits)
  4. Environment variable fallbacks (LLM_PROVIDER, LLM_MODEL)
  5. JSON mode forcing (when LLM needs to return structured JSON)
  6. Mock judge support for testing and fallback scenarios
"""

import os
import time
import functools
import json

from dotenv import load_dotenv
load_dotenv()

try:
    import google.generativeai as genai
except ImportError:
    genai = None
    
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Global clients (lazy-loaded)
_groq_client = None
_gemini_configured = False
_openrouter_client = None
_ollama_client = None


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        if OpenAI is None:
            raise ImportError("openai package is not installed")
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing")
        _groq_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
    return _groq_client


def _configure_gemini():
    global _gemini_configured
    if not _gemini_configured:
        if genai is None:
            raise ImportError("google-generativeai package is not installed")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing")
        genai.configure(api_key=api_key)
        _gemini_configured = True

_openrouter_client = None

def _get_openrouter_client():
    global _openrouter_client
    if _openrouter_client is None:
        if OpenAI is None:
            raise ImportError("openai package is not installed")
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is missing")
        _openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    return _openrouter_client

def _get_ollama_client():
    global _ollama_client
    if _ollama_client is None:
        if OpenAI is None:
            raise ImportError("openai package is not installed")
        api_base = os.getenv("LOCAL_API_BASE", "http://localhost:11434/v1")
        _ollama_client = OpenAI(base_url=api_base, api_key="ollama")
    return _ollama_client

def retry_with_backoff(retries=3, backoff_in_seconds=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == retries - 1:
                        raise e
                    time.sleep(backoff_in_seconds * (2 ** attempt))
        return wrapper
    return decorator

# ── MODEL REGISTRY ────────────────────────────────────────────────────────────
# This is the single source of truth for all available models.
# The Streamlit UI reads this to populate dropdowns.
# Each model has: display name (for UI), provider, model_id (for API), and a description.

def _load_model_registry():
    registry_path = os.path.join(os.path.dirname(__file__), "..", "models.json")
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load models.json: {e}")
        return []

MODEL_REGISTRY = _load_model_registry()

def get_model_registry():
    """Returns the full model registry for UI dropdowns."""
    return MODEL_REGISTRY

def get_display_names():
    """Returns a list of display names for Streamlit selectbox."""
    return [m["display_name"] for m in MODEL_REGISTRY]

def resolve_model_from_display(display_name):
    """
    Given a display name from the UI dropdown, returns (provider, model_id).
    Example: "Llama 3.3 70B (Groq)" → ("groq", "llama-3.3-70b-versatile")
    """
    for m in MODEL_REGISTRY:
        if m["display_name"] == display_name:
            return m["provider"], m["model_id"]
    raise ValueError(f"Unknown model: {display_name}")


# ── TOKEN TRACKING ────────────────────────────────────────────────────────────
# Accumulates token usage across all calls in a session.
# In production, this would be written to a database per-request.

_TOKEN_USAGE = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
}

def get_token_usage():
    return _TOKEN_USAGE.copy()

def reset_token_usage():
    global _TOKEN_USAGE
    _TOKEN_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


# ── PUBLIC API ────────────────────────────────────────────────────────────────
# These are the ONLY two functions that other files should call.
# Both accept explicit provider + model from the UI.
# If not provided, they fall back to the first model in the registry (Llama 3.3 70B).

def call_llm(messages, temperature=0.2, max_tokens=1024, json_mode=False, provider=None, model=None):
    """
    Unified LLM calling interface for the CHATBOT.
    
    temperature=0.2  → Accurate (follows policy) but sounds natural, not robotic.
                       Use 0.0 for Router/Guardrails (called internally with override).
    max_tokens=1024  → ~750 words. Enough for a detailed support reply.
    """
    max_tokens = int(os.getenv("CHAT_MAX_TOKENS", max_tokens))
    provider = provider or os.getenv("LLM_PROVIDER", "groq").lower()
    model = model or os.getenv("LLM_MODEL", MODEL_REGISTRY[0]["model_id"] if MODEL_REGISTRY else "llama-3.3-70b-versatile")
    
    if provider == "groq":
        return _call_groq(messages, temperature, max_tokens, json_mode, model)
    elif provider == "gemini":
        return _call_gemini(messages, temperature, max_tokens, json_mode, model)
    elif provider == "ollama":
        return _call_ollama(messages, temperature, max_tokens, json_mode, model)
    elif provider == "openrouter":
        return _call_openrouter(messages, temperature, max_tokens, json_mode, model)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

def call_judge_llm(messages, temperature=0.0, max_tokens=2048, json_mode=False, provider=None, model=None):
    """
    Dedicated LLM calling interface for the JUDGE / EVALUATOR.
    
    Thin wrapper around call_llm with judge-specific defaults:
    - temperature=0.0  → Scores MUST be reproducible.
    - max_tokens=2048  → Judge needs more space for detailed reasoning.
    - Falls back to EVAL_JUDGE_PROVIDER / EVAL_JUDGE_MODEL env vars.
    """
    provider = provider or os.getenv("EVAL_JUDGE_PROVIDER", os.getenv("LLM_PROVIDER", "groq")).lower()
    model = model or os.getenv("EVAL_JUDGE_MODEL", os.getenv("LLM_MODEL", MODEL_REGISTRY[0]["model_id"] if MODEL_REGISTRY else "llama-3.3-70b-versatile"))
    
    return call_llm(messages, temperature=temperature, max_tokens=max_tokens,
                    json_mode=json_mode, provider=provider, model=model)


# ── PRIVATE: PROVIDER IMPLEMENTATIONS ────────────────────────────────────────

@retry_with_backoff(retries=3)
def _call_groq(messages, temperature, max_tokens, json_mode, model):
    client = _get_groq_client()
    
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
        
    response = client.chat.completions.create(**kwargs)
    
    # Track tokens (safe access with getattr to handle None attributes)
    if response.usage:
        _TOKEN_USAGE["prompt_tokens"] += getattr(response.usage, "prompt_tokens", 0) or 0
        _TOKEN_USAGE["completion_tokens"] += getattr(response.usage, "completion_tokens", 0) or 0
        _TOKEN_USAGE["total_tokens"] += getattr(response.usage, "total_tokens", 0) or 0
        
    return response.choices[0].message.content

@retry_with_backoff(retries=3)
def _call_gemini(messages, temperature, max_tokens, json_mode, model):
    _configure_gemini()
    
    # Convert OpenAI message format to Gemini format
    gemini_messages = []
    system_instruction = None
    
    for msg in messages:
        if msg["role"] == "system":
            system_instruction = msg["content"]
        elif msg["role"] == "user":
            gemini_messages.append({"role": "user", "parts": [msg["content"]]})
        elif msg["role"] == "assistant":
            gemini_messages.append({"role": "model", "parts": [msg["content"]]})
            
    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    if json_mode:
        generation_config.response_mime_type = "application/json"
        
    model_instance = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_instruction
    )
    
    response = model_instance.generate_content(
        gemini_messages,
        generation_config=generation_config
    )
    
    # Track tokens (safe access with getattr to handle None attributes)
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        _TOKEN_USAGE["prompt_tokens"] += getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
        _TOKEN_USAGE["completion_tokens"] += getattr(response.usage_metadata, 'candidates_token_count', 0) or 0
        _TOKEN_USAGE["total_tokens"] += getattr(response.usage_metadata, 'total_token_count', 0) or 0
        
    return response.text

@retry_with_backoff(retries=3)
def _call_ollama(messages, temperature, max_tokens, json_mode, model):
    client = _get_ollama_client()
    
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
        
    response = client.chat.completions.create(**kwargs)
    
    if response.usage:
        _TOKEN_USAGE["prompt_tokens"] += getattr(response.usage, "prompt_tokens", 0)
        _TOKEN_USAGE["completion_tokens"] += getattr(response.usage, "completion_tokens", 0)
        _TOKEN_USAGE["total_tokens"] += getattr(response.usage, "total_tokens", 0)
        
    return response.choices[0].message.content

@retry_with_backoff(retries=3)
def _call_openrouter(messages, temperature, max_tokens, json_mode, model):
    client = _get_openrouter_client()
    
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
        
    # OpenRouter tracking headers
    extra_headers = {
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "ChatBot+Eval",
    }
        
    response = client.chat.completions.create(extra_headers=extra_headers, **kwargs)
    
    if response.usage:
        _TOKEN_USAGE["prompt_tokens"] += getattr(response.usage, "prompt_tokens", 0)
        _TOKEN_USAGE["completion_tokens"] += getattr(response.usage, "completion_tokens", 0)
        _TOKEN_USAGE["total_tokens"] += getattr(response.usage, "total_tokens", 0)
        
    return response.choices[0].message.content