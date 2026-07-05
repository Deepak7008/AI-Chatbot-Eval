"""
config.py — Shared constants for the entire agent pipeline.

Why this file exists:
  Constants like VALID_INTENTS and LOW_CONFIDENCE_THRESHOLD are referenced by
  the router, pipeline dispatcher, eval harness, and dashboard. Keeping them here
  means one change propagates everywhere — no duplication, no drift.
"""

# ── ROUTING ───────────────────────────────────────────────────────────────────

# All intents the Router LLM is allowed to return.
# "out_of_scope" is used when the user's question is outside the domain
# (e.g., "What's the weather?") OR when the router fails to classify at all.
VALID_INTENTS = ["policy", "order", "faq", "chit_chat", "out_of_scope", "multi_intent"]

# Intents that map to a real specialist agent.
ROUTABLE_INTENTS = ["policy", "order", "faq", "chit_chat", "out_of_scope", "multi_intent"]

# If the router's confidence is below this value, the pipeline will flag the
# result as uncertain. The caller can decide to ask the user for clarification.
LOW_CONFIDENCE_THRESHOLD = 0.6

# ── ESCALATION ────────────────────────────────────────────────────────────────

# Human-readable message shown to the user when we escalate.
ESCALATION_MESSAGE = (
    "I'm sorry, I wasn't able to handle your request. "
    "Let me connect you to a customer support agent who can help you further."
)

# ── EVALUATION ────────────────────────────────────────────────────────────────

# Score thresholds used by the eval harness when grading judge outputs.
EVAL_PASS_THRESHOLD = 7        # Score >= 7 out of 10 is a PASS
EVAL_WARN_THRESHOLD = 5        # Score 5–6 is a WARNING (borderline)
                               # Score < 5 is a FAIL

# ── CONVERSATION HISTORY ──────────────────────────────────────────────────────

# Maximum number of previous conversation turns to pass to the specialist LLM.
# E.g., 5 means the last 5 messages (user + assistant combined).
# This prevents token bloat and context degradation in long chats.
MAX_HISTORY_TURNS = 5

# ── TOKEN BUDGETS ─────────────────────────────────────────────────────────────

# These mirror the defaults in llm_client.py but are kept here so the
# eval harness can reference them without importing llm_client.
TOKEN_BUDGET = {
    "router":     150,    # Only outputs ~30 tokens of JSON
    "extractor":  128,    # Only output the extracts based on user intent
    "guardrails": 128,    # Only outputs "safe" / "unsafe" + short reason
    "specialist": 1024,   # Full customer reply (~750 words)
    "synthesizer": 1024,  # Synthesis of multiple sub-intents
    "judge":      2048,   # Score + detailed reasoning (~1500 words)
}

# ── GUARDRAILS ─────────────────────────────────────────────────────────────
BYPASS_KEYWORD = "ByPass"      # Secret word that bypasses guardrails for testing
MAX_INPUT_LENGTH = 1000        # Max characters allowed per user message
