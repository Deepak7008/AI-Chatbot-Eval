# Lessons Learned

## General / Process Architecture
1. **Pacing and Learning:** 
   - *Issue:* Writing an entire phase of code at once prevents interactive learning.
   - *Solution:* For every phase, stop, explain the concepts, brainstorm the approach, and WAIT for explicit approval before creating any files.
   - *Why:* This is a learning project. Breaking work down interactively provides better visibility and education.
2. **Verify Work Artifact vs. Workspace Code:**
   - *Issue:* It is easy to check off tasks in tracking documents (`todo.md`, `task.md`) and assume the work is done, while forgetting to actually modify the real workspace files.
   - *Solution:* Never mark a task complete in tracking documents until the workspace files have actually been modified and visually verified.
   - *Why:* Maintaining alignment between planning artifacts and actual codebase state is critical to prevent compounding architectural debt.

## agents/router.py
3. **Handling Out of Scope Queries:**
   - *Issue:* When the LLM couldn't parse the JSON or failed, it was silently routing to `faq`. This caused hallucinations because the FAQ agent lacked context.
   - *Solution:* Added an `out_of_scope` intent. If the LLM doesn't know the answer, or if parsing fails, it safely falls back to `out_of_scope`.
   - *Why:* Honest signaling. Instead of pretending to know the answer and hallucinating, the system escalates safely.
4. **Silent Fallbacks & Confidence Clamping:**
   - *Issue:* If the LLM returned an invalid intent, the code changed it to `faq` but kept the high confidence score. This masked the failure.
   - *Solution:* If the intent is invalid, force it to `out_of_scope` and explicitly reset the confidence to `0.0`. Added a clamp to ensure confidence is always between `0.0` and `1.0`.
   - *Why:* Fallbacks should never inherit the confidence of the failed prediction.
5. **Separation of Concerns (Router vs Dispatcher):**
   - *Issue:* The Router was returning an `escalation_message` on failure paths, mixing UI logic with classification logic.
   - *Solution:* Removed the UI string from the Router. It only returns `intent`, `confidence`, and `raw_response`.
   - *Why:* The Router should be a pure classification function.

## agents/llm_client.py
6. **Decoupling Model Selection from Environment Variables:**
   - *Issue:* Hardcoding models in `.env` forces the Chatbot and Judge to use the same model, making testing hard.
   - *Solution:* Created a `MODEL_REGISTRY` in `llm_client.py`. The `.env` file now only stores API keys.
   - *Why:* This enables A/B testing, cross-provider evaluation, and defensive programming.
7. **Temperature and Token Constraints:**
   - *Issue:* `temperature=0.0` for everything makes the chatbot sound robotic, while `1.0` makes it hallucinate. Uncapped tokens waste money.
   - *Solution:* Tailored parameters per component (Router: 0.0, Chatbot: 0.2). Capped tokens per agent.
   - *Why:* Optimizing parameters per component balances cost, safety, user experience, and evaluation reliability.
8. **Exponential Backoff for API Resiliency:**
   - *Issue:* Hitting provider rate limits caused the application to crash instantly.
   - *Solution:* Implemented exponential backoff for API retries.
   - *Why:* Exponential backoff gives the provider time to recover and provides robust "self-healing" capabilities.

## agents/guardrails.py
9. **Defense in Depth (Guardrails):**
   - *Issue:* A chatbot without protections is vulnerable to prompt injection and accidental data leakage.
   - *Solution:* Implemented a two-layer guardrail system: `check_input` (blocks malicious inputs) and `check_output` (catches leakages).
   - *Why:* Catching attacks early saves compute. Catching output leaks protects internal rules.
10. **Denial of Wallet Protection:**
    - *Issue:* LLMs charge by the token. Massive input texts incur massive API costs.
    - *Solution:* Implemented a strict 1000-character heuristic block before the LLM.
    - *Why:* Heuristics are free and instant. Always put free checks in front of paid LLM calls.
11. **Fail-Open vs Fail-Closed in Security:**
    - *Issue:* What should the system do if the security LLM API goes down?
    - *Solution:* A JSON parse failure fails closed, but a hard network error fails open.
    - *Why:* A network blip shouldn't stop customers from getting help. Security must not degrade core availability.

## agents/pipeline.py
12. **Parallel Orchestration for Multi-Intent Queries:**
    - *Issue:* Complex user queries spanning multiple domains would either fail or be poorly answered by a single specialized agent.
    - *Solution:* Implemented a parallel `ThreadPoolExecutor` to route sub-intents to their respective specialist agents concurrently. A new Synthesizer agent was created to merge these independent responses.
    - *Why:* This achieves maximum accuracy by letting narrow experts handle data retrieval, while maintaining low latency through concurrent execution.

## agents/specialists.py
13. **Specialist Grounding over Premature Escalation:**
    - *Issue:* The bot was escalating valid queries about product availability instead of checking the provided catalog.
    - *Solution:* Updated specialist prompts to explicitly prioritize answering from context schemas before defaulting to an escalation fallback.
    - *Why:* Grounding instructions prevent false-positive escalations and improve resolution rates.
14. **Context Management & History Pruning:**
    - *Issue:* Passing the entire unbounded conversation history wastes tokens and blows up the context window.
    - *Solution:* Implemented a rolling window of history (e.g., only pass the last N turns).
    - *Why:* Sending long conversation histories increases costs exponentially. Pruning is mandatory for cost and latency control.

## agents/entity_extractor.py
15. **Strict Scoping for Entity Extraction:**
    - *Issue:* Email and complex entity extraction was running across all intents unnecessarily.
    - *Solution:* Entity extraction logic is now strictly scoped to only execute when the intent requires it.
    - *Why:* Minimizing data processing reduces latency, saves API tokens, and limits PII attack surface.
16. **Fast Regex + LLM Fallback (Hybrid Extraction):**
    - *Issue:* Calling an LLM to extract highly structured data is slow and expensive.
    - *Solution:* Use fast Regular Expressions (Regex) as the first line of defense. Only trigger the LLM if Regex fails.
    - *Why:* This hybrid pattern provides zero-cost instantaneous extraction for 90% of cases, and semantic flexibility for the 10% edge cases.

## data/ (policies.json & dataset_multi.json)
17. **Escalation Policy in Data Layer (`policies.json`):**
    - *Issue:* The instruction to "escalate" was hardcoded in the agent's system prompt.
    - *Solution:* Added an `escalation_policy` block to `data/policies.json`.
    - *Why:* Separating business logic from code allows product managers to update triggers easily and saves context window space.
18. **Categorizing Multi-Intent Queries (`dataset_multi.json`):**
    - *Issue:* Queries like "Where is my refund, and do you take Bitcoin?" were incorrectly tagged as `adversarial`.
    - *Solution:* Created a new `complex` category with an `edge_case` tag.
    - *Why:* Asking multiple genuine questions is complex user behavior, not an attack.
19. **Multi-Crescendo Attacks & Expected Outcomes (`dataset_multi.json`):**
    - *Issue:* Multi-turn evaluations only checked if a task was completed, which is ambiguous for adversarial attacks.
    - *Solution:* Added multi-turn adversarial cases with an `expected_outcome` field (e.g., `bot_maintains_boundary`).
    - *Why:* This gives the LLM Judge a precise, unambiguous metric to score against.

## evals/db.py
20. **Resource Leak Prevention in Database Connections:**
    - *Issue:* Database connections were being opened but not properly closed in some code paths, leading to deletion of the data ran during evals testing if any issue occured during the run, it was just wiped out with no trace of what went wrong or what could have been done.
    - *Solution:* Use context managers (`with get_connection() as conn:`) consistently and ensure all database functions properly close connections even on error paths.
    - *Why:* Resource leaks accumulate over time, especially in long-running evaluation jobs, and can cause connection pool exhaustion and application crashes.

## evals/judge.py
21. **Robust JSON Parsing for LLM-as-Judge:**
    - *Issue:* Judge LLMs often output malformed JSON (missing commas, unclosed quotes) causing parsing failures that incorrectly penalized good chatbot answers.
    - *Solution:* Implement multi-layered parsing: standard json.loads() → json_repair → regex extraction → manual fallback, with logging for error diagnosis.
    - *Why:* Evaluation systems must be robust to LLM output inconsistencies to avoid false negative assessments of chatbot performance.

## evals/cascade.py
22. **Weighted Score Calculation Integrity:**
    - *Issue:* Parsing errors causing all 1s dragged down weighted averages, failing conversations with otherwise perfect answers.
    - *Solution:* Detect parsing errors via reasoning patterns and exclude affected turns from weighted score calculations.
    - *Why:* Score calculations should reflect actual chatbot performance, not judge LLM output formatting issues.

## Evaluation vs Production Pipeline Mode

24. **Dual-Mode Architecture for Testing vs Security:**
    - *Issue:* Evaluation test cases contained orders belonging to multiple users (Alice, Bob, Charlie). Hardcoding `user_email="alice@example.com"` in evaluation caused test cases about Bob's and Charlie's orders to fail because Alice couldn't see their orders.
    - *Solution:* Implemented dual-mode architecture:
      - **Production mode** (`user_email` provided): Orders scoped to authenticated user only (IDOR protection)
      - **Evaluation mode** (`user_email=None`): Orders looked up by `order_id` directly (bypasses user scoping)
    - *Why:* Production must respect user boundaries (Alice can't see Bob's orders). Evaluation must test across all test cases (need to answer about any order). The dual-mode architecture ensures correct testing while maintaining production security.

25. **Order Lookup vs User Scoping in Pipeline:**
    - *Issue:* Pipeline's `_fetch_context` for order intent only fetched orders belonging to the authenticated user. Test cases referencing orders belonging to other users would always fail.
    - *Solution:* Added eval mode branch: when no `user_email` provided but `order_id` is extracted, look up the order directly by ID regardless of user ownership.
    - *Why:* Enables testing order-handling capability across the entire dataset without compromising production security model.

26. **Dynamic Context vs Static Test Cases in Groundedness Eval:**
    - *Issue:* Hardcoding reference answers in test cases restricted the evaluation of multi-turn conversational agents that dynamically query databases. Groundedness checks were failing or inaccurate because the reference answer didn't perfectly match the exact text the bot generated.
    - *Solution:* Pass the dynamic pipeline context (e.g., fetched database records or retrieved policy text) directly to the LLM Judge so it can evaluate "Groundedness" against the actual state of the system, rather than a hardcoded string.
    - *Why:* True groundedness should measure if the agent correctly represents the actual data it pulled (avoiding hallucination), not if it memorized a static test script.

27. **Multi-Turn Judge Output Structuring:**
    - *Issue:* Debug evaluation scripts and dashboards crashed with `KeyError: 'accuracy'` because multi-turn evaluations returned reasoning keys on a per-turn basis (e.g., `turn_1_accuracy`) instead of global dimension keys.
    - *Solution:* Implemented safe `.get()` fallbacks and explicitly parsed per-turn structures in the evaluation reports and Streamlit dashboard to cleanly accommodate multi-turn judge schemas.
    - *Why:* Evaluation schemas change shape between single-turn and multi-turn workflows. Defensive dynamic key handling prevents reporting pipeline crashes.

28. **Handling Explicit Nulls in Structured LLM JSON:**
    - *Issue:* The router threw a `TypeError: 'NoneType' object is not iterable` because the LLM explicitly returned `null` for `sub_intents` instead of an empty list `[]`. 
    - *Solution:* Implemented robust fallback parsing (`result.get("sub_intents") or []`) to handle explicit `None` values returned by the model.
    - *Why:* Even with strict structured JSON generation, models may still return `null` for empty fields rather than omitting the key or returning an empty array. Defensive defaulting is required when iterating over LLM lists.