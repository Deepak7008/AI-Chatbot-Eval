"""
bias_check.py — Position bias detection for LLM Judge reliability.

Why this file exists:
  LLM judges are susceptible to position bias — they may score an answer
  differently depending on whether it appears BEFORE or AFTER the reference
  answer in the prompt. This module detects and quantifies that bias.

How it works:
  For each test case, run the judge TWICE:
    1. Normal order:  Reference first, then Actual (the default in judge.py)
    2. Swapped order: Actual first, then Reference

  Then compare scores. If the judge is unbiased, scores should be identical
  (or very close). Large deltas indicate position bias.

Key outputs:
  - Per-case score delta (normal vs swapped)
  - Average delta across all cases (overall bias magnitude)
  - Bias direction: primacy (favors position 1) or recency (favors position 2)
  - Per-dimension breakdown (which dimensions are most biased?)

Research context:
  LMSYS Chatbot Arena and papers like "Judging LLM-as-a-Judge" (Zheng et al.)
  found position bias can shift scores by 0.5-1.0 on a 5-point scale.
  Our threshold: avg_delta < 0.3 = acceptable, > 0.5 = problematic.
"""

import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.llm_client import call_judge_llm
from evals.judge import (
    RUBRIC,
    DIMENSIONS,
    _parse_judge_response as parse_judge_response,
    compute_weighted_score,
)
from evals.db import save_bias_run, save_bias_result, update_bias_run


# ── SWAPPED PROMPT BUILDER ───────────────────────────────────────────────────

def build_swapped_prompt(query: str, reference: str, actual: str, context: str = None) -> list:
    """
    Build a judge prompt with SWAPPED position: actual answer appears first,
    reference answer appears second.

    This is identical to judge.py's build_judge_prompt() except the order
    of "Reference Answer" and "Chatbot's Actual Answer" is reversed in the
    user message. The system prompt (rubric, rules, format) stays the same.

    Why only swap the user message?
      The system prompt defines the rubric — it shouldn't change.
      The bias comes from the ORDER of content in the user message.
    """
    # Build rubric text (same as judge.py)
    rubric_text = ""
    for dim_name, dim_config in RUBRIC.items():
        hard_gate_note = ""
        if dim_config["hard_gate"]:
            hard_gate_note = f" [HARD GATE: Score < {dim_config['hard_gate_threshold']} = automatic FAIL]"
        rubric_text += (
            f"- **{dim_name}** (weight: {dim_config['weight']:.0%}){hard_gate_note}\n"
            f"  {dim_config['description']}\n"
            f"  Scale: {dim_config['scale']}\n\n"
        )

    context_section = ""
    if context:
        context_section = f"""
## Grounding Context
The chatbot had access to the following context when generating its answer.
Use this to evaluate Groundedness -- did the bot stick to this context or hallucinate?

```
{context}
```
"""

    # System prompt is IDENTICAL to the normal judge prompt
    system_prompt = f"""You are an impartial AI evaluation judge. Your job is to score a customer support chatbot's answer against a reference answer across 6 quality dimensions.

## Scoring Rubric
{rubric_text}

## Rules
1. Score each dimension independently on a 1-5 integer scale.
2. For EVERY score, provide a brief reasoning (1-2 sentences) explaining WHY you gave that score.
3. Be strict but fair. A score of 5 means near-perfect; reserve it for truly excellent answers.
4. If the actual answer is factually wrong compared to the reference, Accuracy MUST be <= 2.
5. If the actual answer contains information NOT in the context, Groundedness should be penalized.
6. Your output MUST be valid JSON matching the exact schema below.

## Required Output Format
```json
{{
  "accuracy": {{"score": <1-5>, "reasoning": "<why>"}},
  "groundedness": {{"score": <1-5>, "reasoning": "<why>"}},
  "safety": {{"score": <1-5>, "reasoning": "<why>"}},
  "helpfulness": {{"score": <1-5>, "reasoning": "<why>"}},
  "relevance": {{"score": <1-5>, "reasoning": "<why>"}},
  "tone": {{"score": <1-5>, "reasoning": "<why>"}}
}}
```

Do NOT include any text outside the JSON object. Do NOT wrap in markdown code blocks."""

    # SWAPPED user prompt: Actual answer FIRST, Reference SECOND
    user_prompt = f"""## User Query
{query}

## Chatbot's Actual Answer
{actual}

## Reference Answer (Gold Standard)
{reference}
{context_section}
## Task
Score the chatbot's actual answer against the reference answer using the 6-dimension rubric.
Return your evaluation as a JSON object."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# ── SINGLE CASE BIAS CHECK ───────────────────────────────────────────────────

def check_case_bias(
    query: str,
    reference: str,
    actual: str,
    context: str = None,
    provider: str = None,
    model: str = None,
) -> dict:
    """
    Run the judge twice (normal + swapped order) on a single test case
    and measure the position bias.

    Args:
        query:     The user's original question
        reference: The gold-standard reference answer
        actual:    The chatbot's generated answer
        context:   Optional grounding context
        provider:  Judge LLM provider
        model:     Judge LLM model

    Returns:
        Dict with:
          - normal_scores:  {dim: score} from normal order
          - swapped_scores: {dim: score} from swapped order
          - normal_weighted: Weighted score (normal)
          - swapped_weighted: Weighted score (swapped)
          - deltas:         {dim: normal - swapped} per dimension
          - abs_delta:      Absolute difference in weighted scores
          - direction:      "primacy" if normal > swapped, "recency" if swapped > normal, "none" if equal
    """
    # ── Run 1: Normal order (reference first, actual second) ──
    from evals.judge import build_judge_prompt
    normal_prompt = build_judge_prompt(query, reference, actual, context)

    try:
        normal_raw = call_judge_llm(
            normal_prompt,
            temperature=0.0,
            json_mode=True,
            provider=provider,
            model=model,
        )
        normal_parsed = parse_judge_response(normal_raw)
    except Exception as e:
        # Fail-safe: if the judge call fails, return error result
        normal_parsed = {dim: {"score": 1, "reasoning": f"ERROR: {e}"} for dim in DIMENSIONS}

    # ── Run 2: Swapped order (actual first, reference second) ──
    swapped_prompt = build_swapped_prompt(query, reference, actual, context)

    try:
        swapped_raw = call_judge_llm(
            swapped_prompt,
            temperature=0.0,
            json_mode=True,
            provider=provider,
            model=model,
        )
        swapped_parsed = parse_judge_response(swapped_raw)
    except Exception as e:
        swapped_parsed = {dim: {"score": 1, "reasoning": f"ERROR: {e}"} for dim in DIMENSIONS}

    # ── Compare scores ──
    normal_scores = {dim: normal_parsed[dim]["score"] for dim in DIMENSIONS}
    swapped_scores = {dim: swapped_parsed[dim]["score"] for dim in DIMENSIONS}

    normal_weighted = compute_weighted_score(normal_scores)
    swapped_weighted = compute_weighted_score(swapped_scores)

    deltas = {dim: normal_scores[dim] - swapped_scores[dim] for dim in DIMENSIONS}
    abs_delta = abs(normal_weighted - swapped_weighted)

    # Determine bias direction
    if normal_weighted > swapped_weighted:
        direction = "primacy"   # Judge favors whichever answer appears first
    elif swapped_weighted > normal_weighted:
        direction = "recency"   # Judge favors whichever answer appears second
    else:
        direction = "none"

    return {
        "normal_scores": normal_scores,
        "swapped_scores": swapped_scores,
        "normal_weighted": normal_weighted,
        "swapped_weighted": swapped_weighted,
        "deltas": deltas,
        "abs_delta": round(abs_delta, 4),
        "direction": direction,
    }


# ── BATCH BIAS CHECK ─────────────────────────────────────────────────────────

def run_bias_check(
    test_cases: list,
    provider: str = None,
    model: str = None,
    progress_callback=None,
) -> dict:
    """
    Run position bias detection across multiple test cases.

    This is the top-level function called by the Streamlit Eval page.
    It runs check_case_bias() on each case and aggregates the results.

    Args:
        test_cases:        List of dicts with query, reference_answer, context
        provider:          Judge LLM provider
        model:             Judge LLM model
        progress_callback: fn(current, total, case_id) for UI updates

    Returns:
        Dict with:
          - cases:             List of per-case results
          - avg_abs_delta:     Average absolute delta across cases
          - max_abs_delta:     Maximum delta (worst-case bias)
          - bias_direction:    Overall direction ("primacy", "recency", "mixed", "none")
          - dimension_deltas:  {dim: avg_delta} — which dimensions are most biased
          - verdict:           "LOW_BIAS", "MODERATE_BIAS", or "HIGH_BIAS"
          - interpretation:    Human-readable summary
    """
    if not test_cases:
        return {
            "cases": [],
            "avg_abs_delta": 0.0,
            "max_abs_delta": 0.0,
            "bias_direction": "none",
            "dimension_deltas": {},
            "verdict": "NO_DATA",
            "interpretation": "No test cases provided.",
        }

    import os
    chat_provider = os.getenv("LLM_PROVIDER", "groq").lower()
    chat_model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    resolved_provider = provider or os.getenv("EVAL_JUDGE_PROVIDER", chat_provider).lower()
    resolved_model = model or os.getenv("EVAL_JUDGE_MODEL", chat_model)

    if chat_provider == resolved_provider and chat_model == resolved_model:
        msg = f"⚠️ WARNING: Self-evaluation bias risk! Bias Judge ({resolved_provider}/{resolved_model}) is identical to Chatbot."
        try:
            import streamlit as st
            st.warning(msg)
        except ImportError:
            print(msg)

    # Initialize bias run in DB
    run_id = save_bias_run({
        "sample_size": len(test_cases)
    })

    case_results = []
    total = len(test_cases)

    for i, case in enumerate(test_cases):
        case_id = case.get("id", f"case_{i+1}")

        if progress_callback:
            progress_callback(i + 1, total, case_id)

        result = check_case_bias(
            query=case["query"],
            reference=case.get("reference_answer", ""),
            actual=case.get("actual_answer", case.get("reference_answer", "")),
            context=json.dumps(case.get("context", {})) if case.get("context") else None,
            provider=provider,
            model=model,
        )
        result["case_id"] = case_id
        
        if progress_callback:
            progress_callback(i + 1, total, case_id, result)
        
        # Save per-case result to DB
        if run_id:
            save_bias_result({
                "run_id": run_id,
                "case_id": case_id,
                "category": case.get("category", "unknown"),
                "base_score": result["normal_weighted"],
                "swapped_score": result["swapped_weighted"],
                "delta": result["normal_weighted"] - result["swapped_weighted"]
            })
            
        case_results.append(result)

    # ── Aggregate metrics ──
    abs_deltas = [r["abs_delta"] for r in case_results]
    avg_abs_delta = sum(abs_deltas) / len(abs_deltas) if abs_deltas else 0.0
    max_abs_delta = max(abs_deltas) if abs_deltas else 0.0

    # Per-dimension average delta (signed, to show direction)
    dimension_deltas = {}
    for dim in DIMENSIONS:
        dim_deltas = [r["deltas"][dim] for r in case_results]
        dimension_deltas[dim] = round(sum(dim_deltas) / len(dim_deltas), 4) if dim_deltas else 0.0

    # Overall bias direction
    directions = [r["direction"] for r in case_results if r["direction"] != "none"]
    if not directions:
        bias_direction = "none"
    else:
        primacy_count = sum(1 for d in directions if d == "primacy")
        recency_count = sum(1 for d in directions if d == "recency")
        if primacy_count > recency_count * 1.5:
            bias_direction = "primacy"
        elif recency_count > primacy_count * 1.5:
            bias_direction = "recency"
        else:
            bias_direction = "mixed"

    # Verdict
    if avg_abs_delta < 0.3:
        verdict = "LOW_BIAS"
        interpretation = (
            f"Low position bias detected (avg delta={avg_abs_delta:.3f}). "
            f"The judge is fairly consistent regardless of answer ordering. "
            f"Scores are reliable."
        )
    elif avg_abs_delta < 0.6:
        verdict = "MODERATE_BIAS"
        interpretation = (
            f"Moderate position bias detected (avg delta={avg_abs_delta:.3f}). "
            f"The judge shows some sensitivity to answer ordering. "
            f"Consider using a more capable judge model or averaging normal+swapped scores."
        )
    else:
        verdict = "HIGH_BIAS"
        interpretation = (
            f"High position bias detected (avg delta={avg_abs_delta:.3f}). "
            f"The judge scores are significantly affected by answer ordering. "
            f"Recommendation: switch to a more capable judge model, or use "
            f"the debiased average (normal+swapped)/2."
        )

    if run_id:
        update_bias_run(run_id, {
            "avg_abs_delta": round(avg_abs_delta, 4),
            "max_abs_delta": round(max_abs_delta, 4),
            "bias_direction": bias_direction,
            "verdict": verdict,
            "interpretation": interpretation
        })

    return {
        "cases": case_results,
        "avg_abs_delta": round(avg_abs_delta, 4),
        "max_abs_delta": round(max_abs_delta, 4),
        "bias_direction": bias_direction,
        "dimension_deltas": dimension_deltas,
        "verdict": verdict,
        "interpretation": interpretation,
    }


# ── STANDALONE TEST ──────────────────────────────────────────────────────────
# Run: python -m evals.bias_check

if __name__ == "__main__":
    print("=" * 60)
    print("Position Bias Detection -- Verification Test (3 cases)")
    print("=" * 60)

    TEST_PROVIDER = "groq"
    TEST_MODEL = "llama-3.3-70b-versatile"
    print(f"  Judge: {TEST_PROVIDER}/{TEST_MODEL}")

    # Use 3 diverse test cases from the dataset
    test_cases = [
        {
            "id": "BIAS-1",
            "query": "What is your return policy?",
            "reference_answer": "You can return most items within 30 days of delivery for a full refund. Items must be in original condition with tags attached.",
            "actual_answer": "Our return policy allows returns within 30 days. Items need to be in original condition with tags.",
        },
        {
            "id": "BIAS-2",
            "query": "Do you offer free shipping?",
            "reference_answer": "Yes, we offer free shipping on orders over $50. Standard shipping takes 5-7 business days.",
            "actual_answer": "Free shipping is available for orders above $50. Delivery usually takes 5 to 7 business days via standard shipping.",
        },
        {
            "id": "BIAS-3",
            "query": "How do I cancel my subscription?",
            "reference_answer": "You can cancel your subscription from your Account Settings page. Go to Settings > Subscriptions > Cancel. You will receive a confirmation email.",
            "actual_answer": "To cancel, visit Account Settings, then Subscriptions, and click Cancel. A confirmation email will be sent to you.",
        },
    ]

    print(f"\n  Running bias check on {len(test_cases)} cases...")
    print("  (Each case runs the judge TWICE: normal + swapped order)")
    print()

    result = run_bias_check(
        test_cases=test_cases,
        provider=TEST_PROVIDER,
        model=TEST_MODEL,
    )

    # Per-case results
    for case in result["cases"]:
        print(f"--- {case['case_id']} ---")
        print(f"  Normal weighted:  {case['normal_weighted']:.2f}")
        print(f"  Swapped weighted: {case['swapped_weighted']:.2f}")
        print(f"  Abs delta:        {case['abs_delta']:.4f}")
        print(f"  Direction:        {case['direction']}")
        print(f"  Per-dimension deltas:")
        for dim, delta in case["deltas"].items():
            marker = " <<" if abs(delta) >= 1 else ""
            print(f"    {dim:15s}: {delta:+d}{marker}")
        print()

    # Aggregate results
    print("=" * 60)
    print(f"  Avg absolute delta: {result['avg_abs_delta']:.4f}")
    print(f"  Max absolute delta: {result['max_abs_delta']:.4f}")
    print(f"  Bias direction:     {result['bias_direction']}")
    print(f"  Verdict:            {result['verdict']}")
    print()
    print(f"  Per-dimension avg deltas:")
    for dim, delta in result["dimension_deltas"].items():
        print(f"    {dim:15s}: {delta:+.4f}")
    print()
    print(f"  {result['interpretation']}")
    print()

    if result["verdict"] in ("LOW_BIAS", "MODERATE_BIAS"):
        print("[PASS] Bias check completed. Judge bias is within acceptable range.")
    else:
        print("[WARN] High bias detected. Consider using a different judge model.")
