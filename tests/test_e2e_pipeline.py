"""
E2E Pipeline Test Suite — Data-Driven from tests/test_cases.json

Categories:
  1. guardrail_injection    — Prompt injections that MUST be blocked
  2. crisis_detection       — Distress signals → out_of_scope + escalate
  3. guardrail_bypass       — ByPass keyword skips guardrails
  4. token_stuffing         — Repetitive/spam inputs blocked by heuristic
  5. specialist_intents     — Correct routing + entity extraction
  6. edge_cases             — Ambiguous, vague, adversarial inputs
  7. output_guardrail       — Response must NOT leak system data
  8. low_confidence         — Router uncertainty → immediate escalation
  9. multi_turn             — Multi-turn context, injection, and tone shifts

To add new tests: edit tests/test_cases.json — no Python changes required.
"""
import pytest
import json
import os
import re
from agents.pipeline import run_pipeline
from agents.config import BYPASS_KEYWORD

# ── Load test data ──────────────────────────────────────────────────────
TEST_CASES_PATH = os.path.join(os.path.dirname(__file__), "test_cases.json")
with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
    test_data = json.load(f)


# ── Helper: generate readable test IDs from the JSON ────────────────────
def _ids(cases, key="id"):
    """Return a list of test IDs for pytest parametrize."""
    return [c.get(key, f"case{i}") for i, c in enumerate(cases)]


# =====================================================================
# 1. GUARDRAIL — PROMPT INJECTION
# =====================================================================
_gi = test_data.get("guardrail_injection", [])

@pytest.mark.parametrize("case", _gi, ids=_ids(_gi))
def test_guardrail_prompt_injection(case):
    """Prompt injections, role attacks, PII harvesting must be blocked."""
    result = run_pipeline(case["input"])

    if "expected_escalated" in case:
        assert result["escalated"] == case["expected_escalated"], \
            f"[{case['id']}] escalated mismatch on: {case['input']}"
    if "expected_intent" in case:
        assert result["intent"] == case["expected_intent"], \
            f"[{case['id']}] intent mismatch on: {case['input']}"
    if "expected_entity_key" in case:
        assert result["entities"].get(case["expected_entity_key"]) == case["expected_entity_value"], \
            f"[{case['id']}] entity mismatch on: {case['input']}"


# =====================================================================
# 2. CRISIS DETECTION
# =====================================================================
_cd = test_data.get("crisis_detection", [])

@pytest.mark.parametrize("case", _cd, ids=_ids(_cd))
def test_crisis_detection(case):
    """Distress signals must route to out_of_scope and escalate to a human."""
    result = run_pipeline(case["input"])

    assert result["escalated"] == case["expected_escalated"], \
        f"[{case['id']}] Must escalate crisis input: {case['input']}"
    assert result["intent"] == case["expected_intent"], \
        f"[{case['id']}] Crisis must map to out_of_scope, got: {result['intent']}"


# =====================================================================
# 3. GUARDRAIL BYPASS
# =====================================================================
_gb = test_data.get("guardrail_bypass", [])

@pytest.mark.parametrize("case", _gb, ids=_ids(_gb))
def test_guardrail_bypass_keyword(case):
    """Exact 'ByPass' keyword in input skips guardrail checks."""
    result = run_pipeline(case["input"])

    if "expected_intent_not" in case:
        assert result["intent"] != case["expected_intent_not"], \
            f"[{case['id']}] Bypass failed — still blocked on: {case['input']}"
    if "expected_intent" in case:
        assert result["intent"] == case["expected_intent"], \
            f"[{case['id']}] intent mismatch on: {case['input']}"
    if "expected_escalated" in case:
        assert result["escalated"] == case["expected_escalated"], \
            f"[{case['id']}] escalated mismatch on: {case['input']}"


# =====================================================================
# 4. TOKEN STUFFING
# =====================================================================
_ts = test_data.get("token_stuffing", [])

@pytest.mark.parametrize("case", _ts, ids=_ids(_ts))
def test_token_stuffing(case):
    """Repetitive spam inputs must be blocked by heuristic; legitimate long messages must pass."""
    result = run_pipeline(case["input"])

    assert result["escalated"] == case["expected_escalated"], \
        f"[{case['id']}] escalated mismatch on: {case['input']}"
    if "expected_intent" in case:
        assert result["intent"] == case["expected_intent"], \
            f"[{case['id']}] intent mismatch on: {case['input']}"


# =====================================================================
# 5. SPECIALIST INTENTS
# =====================================================================
_si = test_data.get("specialist_intents", [])

@pytest.mark.parametrize("case", _si, ids=_ids(_si))
def test_specialist_intents(case):
    """Valid queries must route to the correct agent and extract entities."""
    result = run_pipeline(case["input"])

    assert result["intent"] == case["expected_intent"], \
        f"[{case['id']}] intent mismatch on: {case['input']}"
    assert result["escalated"] == case["expected_escalated"], \
        f"[{case['id']}] escalated mismatch on: {case['input']}"
    if "expected_entity_key" in case:
        assert result["entities"].get(case["expected_entity_key"]) == case["expected_entity_value"], \
            f"[{case['id']}] entity '{case['expected_entity_key']}' mismatch on: {case['input']}"


# =====================================================================
# 6. EDGE CASES
# =====================================================================
_ec = test_data.get("edge_cases", [])

@pytest.mark.parametrize("case", _ec, ids=_ids(_ec))
def test_edge_cases(case):
    """Ambiguous, vague, adversarial, and out-of-scope inputs."""
    result = run_pipeline(case["input"])

    if "expected_intent" in case:
        assert result["intent"] == case["expected_intent"], \
            f"[{case['id']}] intent mismatch on: {case['input']}"
    if "expected_escalated" in case:
        assert result["escalated"] == case["expected_escalated"], \
            f"[{case['id']}] escalated mismatch on: {case['input']}"
    if "expected_is_low_confidence" in case:
        assert result["is_low_confidence"] == case["expected_is_low_confidence"], \
            f"[{case['id']}] low_confidence mismatch on: {case['input']}"
    if "expected_entity_key" in case:
        assert result["entities"].get(case["expected_entity_key"]) == case["expected_entity_value"], \
            f"[{case['id']}] entity mismatch on: {case['input']}"


# =====================================================================
# 7. OUTPUT GUARDRAIL
# =====================================================================
_og = test_data.get("output_guardrail", [])

@pytest.mark.parametrize("case", _og, ids=_ids(_og))
def test_output_guardrail(case):
    """Bot response must NOT leak system prompts, raw JSON, or PII."""
    result = run_pipeline(case["trigger_input"])
    response_text = result["text"]

    # Check for forbidden substrings
    for phrase in case.get("expected_output_must_not_contain", []):
        assert phrase.lower() not in response_text.lower(), \
            f"[{case['id']}] Response leaked: '{phrase}'"

    # Check for forbidden regex patterns (e.g. email addresses)
    forbidden_regex = case.get("expected_output_must_not_contain_regex")
    if forbidden_regex:
        assert not re.search(forbidden_regex, response_text), \
            f"[{case['id']}] Response matched forbidden regex: {forbidden_regex}"


# =====================================================================
# 8. LOW CONFIDENCE
# =====================================================================
_lc = test_data.get("low_confidence", [])

@pytest.mark.parametrize("case", _lc, ids=_ids(_lc))
def test_low_confidence(case):
    """Vague inputs must trigger low-confidence escalation; clear inputs must route normally."""
    result = run_pipeline(case["input"])

    assert result["is_low_confidence"] == case["expected_is_low_confidence"], \
        f"[{case['id']}] low_confidence mismatch on: {case['input']}"
    assert result["escalated"] == case["expected_escalated"], \
        f"[{case['id']}] escalated mismatch on: {case['input']}"
    if "expected_intent" in case:
        assert result["intent"] == case["expected_intent"], \
            f"[{case['id']}] intent mismatch on: {case['input']}"


# =====================================================================
# 9. MULTI-TURN CONVERSATIONS
# =====================================================================
_mt = test_data.get("multi_turn", [])

@pytest.mark.parametrize("case", _mt, ids=_ids(_mt))
def test_multi_turn(case):
    """Multi-turn context carryover, intent switches, mid-conversation injections."""
    history = []

    for turn_data in case["turns"]:
        result = run_pipeline(turn_data["input"], history=history)

        if "expected_intent" in turn_data:
            assert result["intent"] == turn_data["expected_intent"], \
                f"[{case['id']}][turn {turn_data['turn']}] intent mismatch on: {turn_data['input']}"
        if "expected_escalated" in turn_data:
            assert result["escalated"] == turn_data["expected_escalated"], \
                f"[{case['id']}][turn {turn_data['turn']}] escalated mismatch on: {turn_data['input']}"
        if "expected_entity_key" in turn_data:
            assert result["entities"].get(turn_data["expected_entity_key"]) == turn_data["expected_entity_value"], \
                f"[{case['id']}][turn {turn_data['turn']}] entity mismatch on: {turn_data['input']}"

        # Build conversation history for the next turn
        history.append({"role": "user", "content": turn_data["input"]})
        history.append({"role": "assistant", "content": result["text"]})
