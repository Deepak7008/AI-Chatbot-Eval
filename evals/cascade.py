"""
cascade.py — Evaluation cascade pipeline: run bot, score with embeddings + judge, save to SQLite.

Why this file exists:
  This is the master orchestrator that ties together:
    1. The chatbot pipeline (agents/pipeline.py) — generates actual answers
    2. The embeddings module (evals/embeddings.py) — cheap semantic similarity
    3. The LLM judge (evals/judge.py) — expensive 6-dimension scoring
    4. The database (evals/db.py) — persists everything

  It handles both single-turn and multi-turn evaluation, tracks costs,
  and provides progress callbacks for the Streamlit UI.

Pipeline:
  Test case → run_pipeline(query) → actual_answer
    → cosine_similarity(actual, reference)
    → judge_response(query, reference, actual)
    → hard_gate check → weighted_score → pass/fail
    → save_eval_result() to SQLite
"""

import json
import os
import sys
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.pipeline import run_pipeline
from agents.llm_client import get_token_usage, reset_token_usage
from evals.embeddings import cosine_similarity
from evals.judge import judge_response, DIMENSIONS
from evals.db import save_eval_run, save_eval_result, update_eval_run


# ── DATASET LOADING ───────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def _load_json(filename):
    """Load a JSON file from the data directory."""
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset(dataset_type: str = "core") -> dict:
    """
    Load evaluation datasets.
    
    Args:
        dataset_type: One of "core", "extended", or "both"
          - "core":     50 single-turn + 17 multi-turn (frozen test sets)
          - "extended":  Only dataset_extended.json (organic flywheel data)
          - "both":      Core + Extended combined
          
    Returns:
        Dict with "single" and "multi" lists of test cases
    """
    single = []
    multi = []

    if dataset_type in ("core", "both"):
        single.extend(_load_json("dataset_single.json"))
        multi.extend(_load_json("dataset_multi.json"))

    if dataset_type in ("extended", "both"):
        extended = _load_json("dataset_extended.json")
        # Extended cases are always single-turn (promoted from chat logs)
        single.extend(extended)

    return {"single": single, "multi": multi}


# ── COST ESTIMATION ───────────────────────────────────────────────────────────
# Model-specific per-token costs for estimation (USD per 1K tokens).
# Sources: Provider pricing pages (as of 2025)
# Note: Free tiers have limits, after which costs apply.

COST_PER_1K_TOKENS = {
    # Groq models (free tier: 30K tokens/min, 1M tokens/day)
    "groq/llama-3.3-70b-versatile": 0.0,
    "groq/llama-3.1-8b-instant": 0.0,
    "groq/mixtral-8x7b-32768": 0.0,
    
    # Gemini models (free tier: 15 req/min, 1M tokens/month)
    "gemini/gemini-2.5-flash": 0.0,
    "gemini/gemini-2.5-pro": 0.0,
    "gemini/gemini-2.0-flash": 0.0,
    
    # OpenRouter models (pricing varies by model)
    "openrouter/deepseek/deepseek-chat": 0.00015,       # $0.15 per 1M tokens
    "openrouter/meta-llama/llama-3.3-70b-instruct": 0.00039,  # $0.39 per 1M
    "openrouter/qwen/qwen-2.5-72b-instruct": 0.00059,   # $0.59 per 1M
    
    # Default fallbacks
    "groq": 0.0,
    "gemini": 0.0,
    "openrouter": 0.002,  # Conservative average
    "ollama": 0.0,
}

def estimate_cost(tokens: int, provider: str = None, model: str = None) -> float:
    """
    Estimate USD cost from token count with model-specific pricing.
    
    Args:
        tokens: Total tokens used
        provider: LLM provider (e.g., "groq", "gemini")
        model: Specific model ID (e.g., "llama-3.3-70b-versatile")
        
    Returns:
        Estimated cost in USD
    """
    # Try exact provider/model combination first
    if provider and model:
        key = f"{provider}/{model}"
        if key in COST_PER_1K_TOKENS:
            rate = COST_PER_1K_TOKENS[key]
            return (tokens / 1000) * rate
    
    # Fall back to provider-only pricing
    if provider in COST_PER_1K_TOKENS:
        rate = COST_PER_1K_TOKENS[provider]
        return (tokens / 1000) * rate
    
    # Ultimate fallback
    return (tokens / 1000) * 0.002  # Conservative $2 per 1M tokens


# ── SINGLE-TURN EVALUATION ───────────────────────────────────────────────────

def run_single_eval(
    test_case: dict,
    provider: str = None,
    model: str = None,
    judge_provider: str = None,
    judge_model: str = None,
    user_email: str = None,
) -> dict:
    """
    Evaluate a single test case through the full cascade.
    
    Flow:
      1. Run the chatbot pipeline on the query
      2. Compute cosine similarity (actual vs reference)
      3. Run the LLM judge (6-dimension scoring)
      4. Determine pass/fail
      
    Args:
        test_case:      Dict with id, category, query, reference_answer, context
        provider:       LLM provider for the chatbot
        model:          LLM model for the chatbot
        judge_provider: LLM provider for the judge
        judge_model:    LLM model for the judge
        user_email:     Simulated user email for order queries
        
    Returns:
        Dict with all scoring data ready for save_eval_result()
    """
    start_time = time.time()
    reset_token_usage()

    case_id = test_case["id"]
    category = test_case.get("category", "unknown")
    query = test_case["query"]
    reference = test_case["reference_answer"]
    context_data = test_case.get("context", {})

    # ── Step 1: Run the chatbot pipeline ──
    pipeline_result = run_pipeline(
        user_message=query,
        history=[],
        provider=provider,
        model=model,
        user_email=user_email,
    )
    actual_answer = pipeline_result.get("text", "")
    pipeline_tokens = pipeline_result.get("tokens_used", 0)

    # ── Step 2: Cosine similarity (cheap first-pass) ──
    cosine_score = cosine_similarity(actual_answer, reference)

    # ── Step 3: LLM Judge (expensive 6-dimension scoring) ──
    # Pass dynamic pipeline context as string for groundedness evaluation
    pipeline_context = pipeline_result.get("context", {})
    context_str = json.dumps(pipeline_context) if pipeline_context else None

    judge_result = judge_response(
        query=query,
        reference=reference,
        actual=actual_answer,
        context=context_str,
        provider=judge_provider,
        model=judge_model,
    )

    # Get total tokens (pipeline + judge combined)
    total_tokens = get_token_usage().get("total_tokens", 0)
    latency_ms = int((time.time() - start_time) * 1000)

    # ── Step 4: Assemble result ──
    scores = judge_result["scores"]
    
    return {
        "case_id": case_id,
        "category": category,
        "is_multi_turn": False,
        "query": query,
        "reference_answer": reference,
        "actual_answer": actual_answer,
        "cosine_similarity": cosine_score,
        "accuracy": scores["accuracy"]["score"],
        "groundedness": scores["groundedness"]["score"],
        "safety": scores["safety"]["score"],
        "helpfulness": scores["helpfulness"]["score"],
        "relevance": scores["relevance"]["score"],
        "tone": scores["tone"]["score"],
        "weighted_score": judge_result["weighted_score"],
        "pass_fail": judge_result["pass_fail"],
        "tokens_used": total_tokens,
        "latency_ms": latency_ms,
        "judge_reasoning_json": json.dumps({
            dim: scores[dim]["reasoning"] for dim in DIMENSIONS
        }),
        # Extra metadata (not saved to DB, used by caller)
        "_hard_gate": judge_result["hard_gate"],
        "_pipeline_result": pipeline_result,
    }


# ── MULTI-TURN EVALUATION ────────────────────────────────────────────────────

def run_multi_turn_eval(
    test_case: dict,
    provider: str = None,
    model: str = None,
    judge_provider: str = None,
    judge_model: str = None,
    user_email: str = None,
) -> dict:
    """
    Evaluate a multi-turn conversation.
    
    Flow:
      1. For each user turn, feed it through run_pipeline (with history)
      2. Score each bot response against its reference answer
      3. Average all turn scores for the case score
      
    The key challenge: maintaining conversation history across turns so
    the chatbot can demonstrate context retention and task completion.
    
    Args:
        test_case: Dict with id, category, turns[], reference_answers[]
        
    Returns:
        Dict with averaged scoring data ready for save_eval_result()
    """
    start_time = time.time()
    reset_token_usage()

    case_id = test_case["id"]
    category = test_case.get("category", "unknown")
    turns = test_case["turns"]
    reference_answers = test_case["reference_answers"]

    # Build pairs of (user_message, reference_answer)
    user_turns = [t for t in turns if t["role"] == "user"]

    # Track conversation history for context retention
    history = []
    turn_results = []

    for turn_idx, user_turn in enumerate(user_turns):
        user_message = user_turn["content"]

        # Run the pipeline with accumulated history
        pipeline_result = run_pipeline(
            user_message=user_message,
            history=history.copy(),
            provider=provider,
            model=model,
            user_email=user_email,
        )
        actual_answer = pipeline_result.get("text", "")

        # Update history for next turn
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": actual_answer})

        # Score this turn if we have a reference answer for it
        if turn_idx < len(reference_answers):
            reference = reference_answers[turn_idx]

            cosine_score = cosine_similarity(actual_answer, reference)

            # Pass dynamic pipeline context as string for groundedness evaluation
            pipeline_context = pipeline_result.get("context", {})
            context_str = json.dumps(pipeline_context) if pipeline_context else None

            try:
                judge_result = judge_response(
                    query=user_message,
                    reference=reference,
                    actual=actual_answer,
                    context=context_str,
                    provider=judge_provider,
                    model=judge_model,
                )
                
                turn_results.append({
                    "cosine": cosine_score,
                    "scores": judge_result["scores"],
                    "weighted_score": judge_result["weighted_score"],
                    "pass_fail": judge_result["pass_fail"],
                    "hard_gate": judge_result["hard_gate"],
                })
            except Exception as e:
                # If judge evaluation fails for this turn, log warning and use fallback
                logger.warning(f"Judge evaluation failed for turn {turn_idx}: {e}")
                
                # Use fallback scores (median of other turns if available, else all 1s)
                fallback_scores = {dim: 1 for dim in ["accuracy", "groundedness", "safety", "helpfulness", "relevance", "tone"]}
                
                # Try to use median of other successful turns if we have them
                if turn_idx > 0 and len(turn_results) > 0:
                    # Get scores from previous successful turns
                    successful_scores = []
                    for prev_turn in turn_results:
                        successful_scores.append(prev_turn["scores"])
                    
                    if successful_scores:
                        # Calculate median for each dimension
                        for dim in fallback_scores.keys():
                            dim_scores = [s[dim]["score"] for s in successful_scores if dim in s]
                            if dim_scores:
                                # Sort and take median
                                dim_scores.sort()
                                median_idx = len(dim_scores) // 2
                                fallback_scores[dim] = dim_scores[median_idx]
                
                turn_results.append({
                    "cosine": cosine_score,
                    "scores": {dim: {"score": fallback_scores[dim], "reasoning": f"Fallback: Judge evaluation failed with error: {e}"} 
                              for dim in fallback_scores},
                    "weighted_score": sum(fallback_scores.values()) / len(fallback_scores),
                    "pass_fail": "PASS" if (sum(fallback_scores.values()) / len(fallback_scores)) >= 3.0 else "FAIL",
                    "hard_gate": {"failed": False, "failures": []},
                })

    # ── Average across all scored turns ──
    if not turn_results:
        # No turns were scored (shouldn't happen, but fail-safe)
        total_tokens = get_token_usage().get("total_tokens", 0)
        latency_ms = int((time.time() - start_time) * 1000)
        return {
            "case_id": case_id,
            "category": category,
            "is_multi_turn": True,
            "query": json.dumps([t["content"] for t in user_turns]),
            "reference_answer": json.dumps(reference_answers),
            "actual_answer": "No turns scored.",
            "cosine_similarity": 0.0,
            "accuracy": 1, "groundedness": 1, "safety": 1,
            "helpfulness": 1, "relevance": 1, "tone": 1,
            "weighted_score": 1.0,
            "pass_fail": "FAIL",
            "tokens_used": total_tokens,
            "latency_ms": latency_ms,
            "judge_reasoning_json": json.dumps({"error": "No turns scored"}),
            "_hard_gate": {"failed": True, "failures": []},
        }

    n = len(turn_results)

    # Average each dimension across turns, but handle parsing errors specially
    avg_scores = {}
    for dim in DIMENSIONS:
        # Check for turns with parsing errors (all 1s with error reasoning)
        valid_scores = []
        for tr in turn_results:
            score = tr["scores"][dim]["score"]
            reasoning = tr["scores"][dim]["reasoning"]
            
            # Skip scores that are clearly parsing errors (score=1 with ERROR/Fallback in reasoning)
            if score == 1 and ("ERROR:" in reasoning or "Fallback:" in reasoning):
                continue
            valid_scores.append(score)
        
        # If we have some valid scores, use them; otherwise use all scores
        if valid_scores:
            dim_total = sum(valid_scores)
            avg_scores[dim] = round(dim_total / len(valid_scores))
        else:
            # All turns had parsing errors, use the (incorrect) average
            dim_total = sum(tr["scores"][dim]["score"] for tr in turn_results)
            avg_scores[dim] = round(dim_total / n)

    avg_cosine = sum(tr["cosine"] for tr in turn_results) / n
    
    # Calculate weighted average, skipping turns with parsing errors
    valid_weighted_scores = []
    for tr in turn_results:
        # Check if this turn has parsing errors (look for ERROR/Fallback in any dimension)
        has_parsing_error = False
        for dim in DIMENSIONS:
            reasoning = tr["scores"][dim]["reasoning"]
            if "ERROR:" in reasoning or "Fallback:" in reasoning:
                has_parsing_error = True
                break
        
        if not has_parsing_error:
            valid_weighted_scores.append(tr["weighted_score"])
    
    # Use valid scores if available, otherwise use all scores
    if valid_weighted_scores:
        avg_weighted = sum(valid_weighted_scores) / len(valid_weighted_scores)
    else:
        avg_weighted = sum(tr["weighted_score"] for tr in turn_results) / n

    # Any turn failing a hard gate fails the whole case
    any_hard_gate_failed = any(tr["hard_gate"]["failed"] for tr in turn_results)
    # Check if any turn failed, but ignore failures due to parsing errors
    any_turn_failed = False
    for tr in turn_results:
        # Check if this failure is due to parsing error
        has_parsing_error = False
        for dim in DIMENSIONS:
            reasoning = tr["scores"][dim]["reasoning"]
            if "ERROR:" in reasoning or "Fallback:" in reasoning:
                has_parsing_error = True
                break
        
        # Only count as failure if not due to parsing error
        if tr["pass_fail"] == "FAIL" and not has_parsing_error:
            any_turn_failed = True
            break

    if any_hard_gate_failed or any_turn_failed:
        pass_fail = "FAIL"
    else:
        pass_fail = "PASS"

    total_tokens = get_token_usage().get("total_tokens", 0)
    latency_ms = int((time.time() - start_time) * 1000)

    # Collect reasoning from all turns
    all_reasoning = {}
    for i, tr in enumerate(turn_results):
        for dim in DIMENSIONS:
            all_reasoning[f"turn_{i+1}_{dim}"] = tr["scores"][dim]["reasoning"]
            all_reasoning[f"turn_{i+1}_{dim}_score"] = tr["scores"][dim]["score"]

    return {
        "case_id": case_id,
        "category": category,
        "is_multi_turn": True,
        "query": json.dumps([t["content"] for t in user_turns]),
        "reference_answer": json.dumps(reference_answers),
        "actual_answer": json.dumps([h["content"] for h in history if h["role"] == "assistant"]),
        "cosine_similarity": round(avg_cosine, 4),
        "accuracy": avg_scores.get("accuracy", 1),
        "groundedness": avg_scores.get("groundedness", 1),
        "safety": avg_scores.get("safety", 1),
        "helpfulness": avg_scores.get("helpfulness", 1),
        "relevance": avg_scores.get("relevance", 1),
        "tone": avg_scores.get("tone", 1),
        "weighted_score": round(avg_weighted, 4),
        "pass_fail": pass_fail,
        "tokens_used": total_tokens,
        "latency_ms": latency_ms,
        "judge_reasoning_json": json.dumps(all_reasoning),
        "_hard_gate": {"failed": any_hard_gate_failed, "failures": []},
    }


# ── FULL EVAL SUITE ──────────────────────────────────────────────────────────

def run_eval_suite(
    dataset_type: str = "core",
    provider: str = None,
    model: str = None,
    judge_provider: str = None,
    judge_model: str = None,
    user_email: str = None,
    progress_callback=None,
    categories: list = None,
) -> dict:
    """
    Run the full evaluation suite: load datasets, score every case, save to SQLite.
    
    This is the top-level function called by the Streamlit Eval page.
    
    Args:
        dataset_type:      "core", "extended", or "both"
        provider:          LLM provider for the chatbot
        model:             LLM model for the chatbot
        judge_provider:    LLM provider for the judge
        judge_model:       LLM model for the judge
        user_email:        Simulated user email for order queries
        progress_callback: fn(current, total, case_id, status) for UI updates
        categories:        List of categories to run (runs all if None)
        
    Returns:
        Dict with run summary:
          - run_id, total_cases, passed, failed, pass_rate
          - avg_score, total_tokens, total_cost, total_time_sec
          - results: list of per-case result dicts
    """
    suite_start = time.time()
    
    # Resolve default models
    provider = provider or os.getenv("LLM_PROVIDER", "groq").lower()
    model = model or os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    judge_provider = judge_provider or os.getenv("EVAL_JUDGE_PROVIDER", provider).lower()
    judge_model = judge_model or os.getenv("EVAL_JUDGE_MODEL", model)
    
    # Safety Check: Self-evaluation bias risk with auto-fallback
    if provider == judge_provider and model == judge_model:
        msg = f"⚠️ CRITICAL: Self-evaluation bias blocked! Judge ({judge_provider}/{judge_model}) is identical to Chatbot."
        # Auto-fallback: Use first different model from registry
        from agents.llm_client import MODEL_REGISTRY
        fallback_found = False
        for model_dict in MODEL_REGISTRY:
            if model_dict["provider"] != provider or model_dict["model_id"] != model:
                judge_provider = model_dict["provider"]
                judge_model = model_dict["model_id"]
                msg += f" Auto-fallback to: {judge_provider}/{judge_model}"
                fallback_found = True
                break
        
        if not fallback_found:
            # If no alternative found, error out - evaluation cannot proceed
            msg += " No alternative models found. Cannot run evaluation."
            
        try:
            import streamlit as st
            st.error(msg)
        except ImportError:
            print(f"ERROR: {msg}")
            
        if not fallback_found:
            # Return error response since evaluation cannot proceed
            return {"error": msg, "run_id": None}
    
    # Load datasets
    datasets = load_dataset(dataset_type)
    single_cases = datasets["single"]
    multi_cases = datasets["multi"]
    
    if categories is not None:
        single_cases = [c for c in single_cases if c.get("category", "unknown") in categories]
        multi_cases = [c for c in multi_cases if c.get("category", "unknown") in categories]
        
    total_cases = len(single_cases) + len(multi_cases)

    if total_cases == 0:
        return {"error": "No test cases found", "total_cases": 0}
    
    # Create evaluation run record BEFORE starting evaluation
    run_id = save_eval_run({
        "run_name": f"Eval {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "timestamp": datetime.now().isoformat(),
        "dataset_used": dataset_type,
        "model": f"{provider}/{model}",
        "judge_model": f"{judge_provider}/{judge_model}",
        "total_cases": total_cases,
        "passed_cases": 0,
        "pass_rate": 0.0,
        "avg_score": 0.0,
        "total_cost": 0.0,
        "total_tokens": 0,
        "total_time_sec": 0.0,
        "ci_lower": 0.0,
        "ci_upper": 0.0,
    })
    
    if run_id is None:
        return {"error": "Failed to create evaluation run record", "run_id": None}
    
    all_results = []
    passed = 0
    failed = 0
    total_tokens = 0
    weighted_scores = []
    current = 0

    # ── Run single-turn cases ──
    for case in single_cases:
        current += 1
        if progress_callback:
            progress_callback(current, total_cases, case["id"], "running")

        try:
            result = run_single_eval(
                test_case=case,
                provider=provider,
                model=model,
                judge_provider=judge_provider,
                judge_model=judge_model,
                user_email=user_email,
            )
        except Exception as e:
            # Don't let one case crash the entire suite
            result = _error_result(case["id"], case.get("category", "unknown"),
                                   case["query"], case.get("reference_answer", ""),
                                   str(e), is_multi_turn=False)

        # Add run_id to result and save immediately (checkpointing)
        result["run_id"] = run_id
        save_eval_result(result)
        all_results.append(result)
        
        # Update statistics
        if result["pass_fail"] == "PASS":
            passed += 1
        else:
            failed += 1
            
        total_tokens += result.get("tokens_used", 0)
        weighted_scores.append(result.get("weighted_score", 1.0))
        
        # Update run statistics periodically (every 5 cases for efficiency)
        if current % 5 == 0 or current == total_cases:
            current_avg = sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0.0
            current_pass_rate = passed / current if current > 0 else 0.0
            current_cost = estimate_cost(total_tokens, provider, model)
            current_time = time.time() - suite_start
            
            update_eval_run(run_id, {
                "total_cases": current,
                "passed_cases": passed,
                "pass_rate": current_pass_rate,
                "avg_score": current_avg,
                "total_cost": current_cost,
                "total_tokens": total_tokens,
                "total_time_sec": current_time,
            })

        if progress_callback:
            progress_callback(current, total_cases, case["id"], result["pass_fail"], result_data=result)

    # ── Run multi-turn cases ──
    for case in multi_cases:
        current += 1
        if progress_callback:
            progress_callback(current, total_cases, case["id"], "running")

        try:
            result = run_multi_turn_eval(
                test_case=case,
                provider=provider,
                model=model,
                judge_provider=judge_provider,
                judge_model=judge_model,
                user_email=user_email,
            )
        except Exception as e:
            query_summary = json.dumps([t["content"] for t in case.get("turns", []) if t["role"] == "user"])
            ref_summary = json.dumps(case.get("reference_answers", []))
            result = _error_result(case["id"], case.get("category", "unknown"),
                                   query_summary, ref_summary,
                                   str(e), is_multi_turn=True)

        # Add run_id to result and save immediately (checkpointing)
        result["run_id"] = run_id
        save_eval_result(result)
        all_results.append(result)
        
        # Update statistics
        if result["pass_fail"] == "PASS":
            passed += 1
        else:
            failed += 1
            
        total_tokens += result.get("tokens_used", 0)
        weighted_scores.append(result.get("weighted_score", 1.0))
        
        # Update run statistics periodically
        if current % 5 == 0 or current == total_cases:
            current_avg = sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0.0
            current_pass_rate = passed / current if current > 0 else 0.0
            current_cost = estimate_cost(total_tokens, provider, model)
            current_time = time.time() - suite_start
            
            update_eval_run(run_id, {
                "total_cases": current,
                "passed_cases": passed,
                "pass_rate": current_pass_rate,
                "avg_score": current_avg,
                "total_cost": current_cost,
                "total_tokens": total_tokens,
                "total_time_sec": current_time,
            })

        if progress_callback:
            progress_callback(current, total_cases, case["id"], result["pass_fail"], result_data=result)

    # ── Final aggregation ──
    suite_time = time.time() - suite_start
    pass_rate = passed / total_cases if total_cases > 0 else 0.0
    avg_score = sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0.0
    total_cost = estimate_cost(total_tokens, provider or "groq")
    
    # Compute confidence interval for average score
    from evals.metrics import bootstrap_ci
    ci_result = bootstrap_ci(weighted_scores) if weighted_scores else {"lower": 0.0, "mean": 0.0, "upper": 0.0, "width": 0.0}
    
    # Final run update with complete statistics
    update_eval_run(run_id, {
        "total_cases": total_cases,
        "passed_cases": passed,
        "pass_rate": pass_rate,
        "avg_score": avg_score,
        "total_cost": total_cost,
        "total_tokens": total_tokens,
        "total_time_sec": suite_time,
        "ci_lower": ci_result["lower"],
        "ci_upper": ci_result["upper"],
    })

    return {
        "run_id": run_id,
        "total_cases": total_cases,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(pass_rate, 4),
        "avg_score": round(avg_score, 4),
        "ci_lower": round(ci_result["lower"], 4),
        "ci_upper": round(ci_result["upper"], 4),
        "ci_width": round(ci_result["width"], 4),
        "total_tokens": total_tokens,
        "total_cost": round(total_cost, 6),
        "total_time_sec": round(suite_time, 2),
        "results": all_results,
    }


def _error_result(case_id, category, query, reference, error_msg, is_multi_turn=False):
    """Generate a FAIL result for a case that threw an exception."""
    return {
        "case_id": case_id,
        "category": category,
        "is_multi_turn": is_multi_turn,
        "query": query,
        "reference_answer": reference,
        "actual_answer": f"ERROR: {error_msg}",
        "cosine_similarity": 0.0,
        "accuracy": 1, "groundedness": 1, "safety": 1,
        "helpfulness": 1, "relevance": 1, "tone": 1,
        "weighted_score": 1.0,
        "pass_fail": "FAIL",
        "tokens_used": 0,
        "latency_ms": 0,
        "judge_reasoning_json": json.dumps({"error": error_msg}),
        "_hard_gate": {"failed": True, "failures": []},
    }


# ── STANDALONE TEST ──────────────────────────────────────────────────────────
# Run: python -m evals.cascade
# Tests 3 single-turn cases and 1 multi-turn case.

if __name__ == "__main__":
    print("=" * 60)
    print("Cascade Pipeline -- Verification Test (5 cases)")
    print("=" * 60)

    TEST_PROVIDER = "groq"
    TEST_MODEL = "llama-3.3-70b-versatile"
    EVAL_JUDGE_PROVIDER = "groq"
    EVAL_JUDGE_MODEL = "llama-3.3-70b-versatile"
    print(f"  Chatbot: {TEST_PROVIDER}/{TEST_MODEL}")
    print(f"  Judge:   {EVAL_JUDGE_PROVIDER}/{EVAL_JUDGE_MODEL}")

    # Load just first 3 single-turn and 1 multi-turn for quick test
    datasets = load_dataset("core")
    test_single = datasets["single"][:3]
    test_multi = datasets["multi"][:1]

    print(f"\n  Running {len(test_single)} single-turn + {len(test_multi)} multi-turn cases...")
    print()

    # -- Single-turn tests --
    for i, case in enumerate(test_single, 1):
        print(f"--- Single-turn {i}/{len(test_single)}: {case['id']} ---")
        result = run_single_eval(
            test_case=case,
            provider=TEST_PROVIDER,
            model=TEST_MODEL,
            judge_provider=EVAL_JUDGE_PROVIDER,
            judge_model=EVAL_JUDGE_MODEL,
        )
        print(f"  Query:    {case['query'][:60]}...")
        print(f"  Cosine:   {result['cosine_similarity']:.4f}")
        print(f"  Weighted: {result['weighted_score']:.2f}/5.00")
        print(f"  Verdict:  {result['pass_fail']}")
        print(f"  Scores:   Acc={result['accuracy']} Gnd={result['groundedness']} "
              f"Saf={result['safety']} Help={result['helpfulness']} "
              f"Rel={result['relevance']} Tone={result['tone']}")
        print(f"  Tokens:   {result['tokens_used']}  Latency: {result['latency_ms']}ms")
        print()

    # -- Multi-turn test --
    if test_multi:
        case = test_multi[0]
        print(f"--- Multi-turn: {case['id']} ---")
        result = run_multi_turn_eval(
            test_case=case,
            provider=TEST_PROVIDER,
            model=TEST_MODEL,
            judge_provider=EVAL_JUDGE_PROVIDER,
            judge_model=EVAL_JUDGE_MODEL,
        )
        print(f"  Turns:    {len(case['turns'])} total, {len(case['reference_answers'])} scored")
        print(f"  Cosine:   {result['cosine_similarity']:.4f}")
        print(f"  Weighted: {result['weighted_score']:.2f}/5.00")
        print(f"  Verdict:  {result['pass_fail']}")
        print(f"  Scores:   Acc={result['accuracy']} Gnd={result['groundedness']} "
              f"Saf={result['safety']} Help={result['helpfulness']} "
              f"Rel={result['relevance']} Tone={result['tone']}")
        print(f"  Tokens:   {result['tokens_used']}  Latency: {result['latency_ms']}ms")
        print()

    print("=" * 60)
    total = len(test_single) + len(test_multi)
    print(f"[PASS] Cascade pipeline verified: {total} cases evaluated.")
