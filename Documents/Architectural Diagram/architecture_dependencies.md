# Architecture: File Dependencies

This diagram shows **which files import what** — making it easy to trace how a change in one module ripples through the system.

**Color Legend:**
- 🟠 **Orange** — Orchestrator (entry point that wires everything together)
- 🔵 **Blue** — Agent modules (contain LLM calls and business logic)
- ⚪ **Grey** — Shared utilities (no LLM, used by many modules)
- 🟢 **Green** — Data files (JSON on disk, read at runtime)

```mermaid
graph TD
    classDef orchestrator fill:#FF8C00,stroke:#333,stroke-width:2px,color:#fff;
    classDef agent fill:#4A90E2,stroke:#333,stroke-width:2px,color:#fff;
    classDef shared fill:#607D8B,stroke:#333,stroke-width:2px,color:#fff;
    classDef data fill:#27AE60,stroke:#333,stroke-width:2px,color:#fff;

    pipeline["pipeline.py\n(Orchestrator)"]:::orchestrator

    guardrails["guardrails.py"]:::agent
    router["router.py"]:::agent
    extractor["entity_extractor.py"]:::agent
    specialists["specialists.py"]:::agent

    config["config.py"]:::shared
    utils["utils.py"]:::shared
    llm_client["llm_client.py"]:::shared

    mock_db["mock_db.json"]:::data
    policies["policies.json"]:::data

    %% Pipeline imports all agents
    pipeline -->|"check_input, check_output"| guardrails
    pipeline -->|"route_query"| router
    pipeline -->|"extract_entities"| extractor
    pipeline -->|"handle_query, run_synthesizer"| specialists

    %% Pipeline imports shared
    pipeline -->|"ESCALATION_MESSAGE, BYPASS_KEYWORD"| config
    pipeline -->|"load_json"| utils

    %% Agents import llm_client
    guardrails -->|"call_llm"| llm_client
    router -->|"call_llm"| llm_client
    extractor -->|"call_llm"| llm_client
    specialists -->|"call_llm"| llm_client

    %% Agents import config
    guardrails -->|"TOKEN_BUDGET, BYPASS_KEYWORD, MAX_INPUT_LENGTH"| config
    router -->|"VALID_INTENTS, ROUTABLE_INTENTS, LOW_CONFIDENCE_THRESHOLD"| config
    extractor -->|"TOKEN_BUDGET"| config
    specialists -->|"MAX_HISTORY_TURNS, TOKEN_BUDGET, ESCALATION_MESSAGE, ROUTABLE_INTENTS"| config

    %% Agents import utils
    guardrails -->|"extract_json"| utils
    router -->|"extract_json"| utils
    extractor -->|"extract_json"| utils

    %% Data accessed at runtime via load_json
    utils -.->|"reads at runtime"| mock_db
    utils -.->|"reads at runtime"| policies
```

---

## Quick Reference: What Each Shared Module Provides

### config.py
| Constant | Used By |
|----------|---------|
| `VALID_INTENTS` | router |
| `ROUTABLE_INTENTS` | router, specialists |
| `LOW_CONFIDENCE_THRESHOLD` | router |
| `ESCALATION_MESSAGE` | pipeline, specialists |
| `BYPASS_KEYWORD` | pipeline, guardrails |
| `MAX_INPUT_LENGTH` | guardrails |
| `TOKEN_BUDGET` | guardrails, router, extractor, specialists |
| `MAX_HISTORY_TURNS` | specialists |
| `EVAL_PASS_THRESHOLD` | cascade.py |
| `EVAL_WARN_THRESHOLD` | cascade.py |

### utils.py
| Function | Used By | Purpose |
|----------|---------|---------|
| `extract_json()` | guardrails, router, extractor | Safe JSON parsing with 3-tier fallback |
| `load_json()` | pipeline | Cached file reader with `@lru_cache` |

### llm_client.py
| Function | Used By | Purpose |
|----------|---------|---------|
| `call_llm()` | guardrails, router, extractor, specialists | Unified multi-provider API wrapper with backoff retries |
| `call_judge_llm()` | judge.py | Separate judge model call with backoff retries |
| Client Getters | internal | Singletons (`_get_groq_client`, etc.) for connection pooling |
