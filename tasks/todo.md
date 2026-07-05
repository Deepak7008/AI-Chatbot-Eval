# ChatBot+Eval — Task Tracker (v3)

## Phase 1: Foundation & Data Layer
- [x] Create project directories (`data/`, `agents/`, `evals/`, `app/`, `app/pages/`)
- [x] Create `requirements.txt` and `.env.example`
- [x] Create `data/mock_db.json` (orders, users, products)
- [x] Create `data/policies.json` (store policies)
- [x] Create `data/dataset_single.json` (50 single-turn cases, 12 adversarial)
- [x] Create `data/dataset_multi.json` (15 multi-turn cases)
- [x] Initialize `data/dataset_extended.json` (empty list)
- [x] Set up SQLite schema in `evals/db.py` (chat_logs, eval_runs, eval_results) with `router_reasoning` and `escalated`

## Phase 2: Agent Layer
- [x] Build `agents/llm_client.py` (Groq/Gemini client + token tracking)
- [x] Build `agents/entity_extractor.py` (regex + LLM fuzzy matching)
- [x] Build `agents/guardrails.py` (input/output checks + "WellSaid" bypass)
- [x] Build `agents/router.py` (intent + confidence scoring)
- [x] Build `agents/specialists.py` (Policy, Order, FAQ agents + escalation handler)
- [x] Test the full agent pipeline in isolation

## Phase 2.5: Multi-Intent Upgrade
- [x] Update `agents/config.py` for multi_intent and synthesizer budgets
- [x] Update `agents/router.py` to route to multi_intent and sub_intents
- [x] Update `agents/entity_extractor.py` to handle multi_intent sub-intents
- [x] Build `run_synthesizer()` in `agents/specialists.py`
- [x] Implement parallel processing & synthesis in `agents/pipeline.py`
- [x] Update all system documents in `Documents/` and `Documents/Architectural Diagram/`
- [x] Verify using end-to-end test query

## Phase 3: Evaluation Pipeline
- [x] Build `evals/embeddings.py` (cosine similarity with `sentence-transformers`)
- [x] Build `evals/judge.py` (6-dimension rubric + structured reasoning)
- [x] Build `evals/cascade.py` (single-turn + multi-turn pipeline + cost tracking)
- [x] Build `evals/metrics.py` (Spearman, Cohen's d/kappa, bootstrap CI, t-test, calibration)
- [x] Build `evals/bias_check.py` (position bias detection)
- [x] Test eval pipeline on 5 cases

## Phase 4: Streamlit Dashboard (4-Page App)
- [x] Build `app/main.py` (entry point + setup)
- [x] Build `app/pages/1_💬_Chat.py` (live UI, entity tags, guardrail trace, bypass view, SQLite logging)
- [x] Build `app/pages/2_🧪_Eval.py` (eval runner, dataset selector, reasoning expanders)
- [x] Build `app/pages/3_📊_Dashboard.py` (Plotly charts, calibration curve, stats)
- [x] Build `app/pages/4_📋_Chat_History.py` (browse logs, full trace, promote to test cases)

## Phase 5: Polish & Verification
- [x] Run full 65-case eval suite
- [x] Promote 1 real chat into extended dataset and re-run "Both"
- [x] Write `README.md` with setup instructions and concept explanations
- [x] Final end-to-end verification walkthrough
