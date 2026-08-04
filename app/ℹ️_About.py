import streamlit as st
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.ui_utils import load_fluent_css

st.set_page_config(
    page_title="About",
    page_icon="ℹ️",
    layout="wide"
)

load_fluent_css()

st.title("Agentic Chatbot with Robust Eval Framework")
st.markdown(
    "A production-ready AI agent pipeline demonstrating robust conversational routing, guardrails and comprehensive evaluation framework — "
    "using LLM-as-a-judge to evaluate the output across several dimensions"
    
)

stat_col1, stat_col2 = st.columns(2)
with stat_col2:
    st.metric("Judge Dimensions", "6")
with stat_col1:
    st.metric("Architecture Layers", "4")

st.divider()

# ============================================================
# Project Description
# ============================================================
st.subheader("Project Overview")

desc_col1, desc_col2, desc_col3 = st.columns(3)

with desc_col1:
    st.markdown("#### 🤖 Agent Pipeline")
    st.markdown(
        """
        - **Multi-Intent Router:** Classifies user inputs, assigns confidence scores, and
          routes to appropriate specialists (Policy, Order, FAQ,ChitChat,Multi-Intent).
        - **Entity Extractor:** Fuzzy matching and LLM extraction for product names,
          order IDs, and emails.
        - **Safety Guardrails:** Real-time input/output filtering for prompt injection,
          crisis escalation, and data leaks.
        - **Synthesizer:** Parallel processing of multi-intent queries, merging responses
          seamlessly.
        """
    )

with desc_col2:
    st.markdown("#### ⚖️ Evaluation Engine")
    st.markdown(
        """
        - **LLM-as-a-Judge:** Strict 6-dimension rubric (Accuracy, Groundedness, Safety,
          Helpfulness, Relevance, Tone) with hard-cut gates.
        - **Semantic Similarity:** Cosine similarity between generated and reference
          answers using sentence-transformers.
        - **Statistical Rigor:** Spearman correlation, Cohen's d/kappa, and bootstrap
          confidence intervals.
        - **Bias Detection:** Built-in checks for position bias in judge outputs.
        """
    )

with desc_col3:
    st.markdown("#### 🎛️ UI/UX")
    st.markdown(
        """
        - **Setup & Config:** Dynamically switch between Groq, Gemini, OpenRouter, or
          local models.
        - **Live Chat & Trace:** Chat with the bot and view a real-time debug trace of
          the entire pipeline (routing, entities, guardrails, latency).
        - **Evaluation Runner:** Single or bulk test cases with Run A vs Run B comparison.
        - **Data Flywheel:** Promote real chat logs directly into the evaluation dataset.
        """
    )

st.divider()

# ============================================================
# High-Level Architecture
# ============================================================
st.subheader("Architecture at a Glance")

st.markdown(
    "The system is organized into four layers. The UI orchestrates the Agent and "
    "Evaluation layers, both backed by a shared Data layer."
)

layer_row1 = st.columns(4)

layer_cards = [
    (
        "🟠 UI Layer",
        "",
        "Streamlit multi-page interface: Setup, Chat, History, Evaluation, Dashboard, "
        "Lessons. Entry point for all user interaction and visualization."
    ),
    (
        "🔵 Agent Layer",
        "",
        "LLM-powered pipeline: intent router, entity extractor, safety guardrails, "
        "specialists (Policy/Order/FAQ), and synthesizer, orchestrated by the pipeline."
    ),
    (
        "🟢 Data Layer",
        "",
        "Persistent stores: mock customer/order data, store policies, frozen + extended "
        "eval datasets, and an SQLite database for chat logs and eval results."
    ),
    (
        "🟣 Evaluation Layer",
        "",
        "LLM-as-a-Judge scoring, semantic similarity, statistical metrics, bias checks, "
        "and cascade orchestration that saves results back to SQLite."
    ),
]

for col, (title, path, desc) in zip(layer_row1, layer_cards):
    with col:
        st.markdown(f"**{title}**")
        #st.code(path, language="text")
        st.caption(desc)

st.markdown("")

flow_col1, flow_col2 = st.columns(2)

with flow_col1:
    with st.expander("💬 Live Chat Flow (7 steps)", expanded=False):
        st.markdown(
            """
            1. `guardrails.py` — checks input length, token-stuffing, and prompt-injection checks
            2. `router.py` — route_query based on intent classification + confidence
            3. `entity_extractor.py` — extract_entities based on user order IDs, emails, products
            4. `pipeline.py` — fetch_context for parallel data loading
            5. `specialists.py` — run different agents based on user query
            6. `synthesizer.py` — Merges parallel answers from different agent to summarize
            7. `guardrails.py` — check_output response for leak/PII scan before reply

            """
        )

with flow_col2:
    with st.expander("⚖️ Evaluation Flow (6 steps)", expanded=False):
        st.markdown(
            """
            1. Load test cases from `dataset_*.json`
            2. Run the chat pipeline on each query
            3. Compute cosine similarity vs reference answer
            4. Grade response with the LLM judge (6-dimension rubric)
            5. Aggregate metrics (hard cuts, weighted score, pass/fail)
            6. Persist results to SQLite for dashboard analytics
            """
        )

st.markdown("")


st.divider()

# ============================================================
# Getting Started
# ============================================================

st.title("**Getting Started**")
# ==============================================================================
# Live Chat
# ==============================================================================

st.subheader("`Live Chat`")

st.write(
    "*Interact with the AI chatbot and inspect its complete reasoning workflow.*"
)

st.markdown("""
1. Go to the **Setup** page and configure the **Chatbot** and **Judge** models. *(Password protected, as changing models and configurations can incur token costs.)*
2. Open the **Chat** page to ask questions, receive AI-generated responses, and inspect the execution/debug trace.
3. Visit the **Chat History** page to review previous conversations, examine the agents involved, and understand how each request was processed.
""")

# ==============================================================================
# Evaluation
# ==============================================================================

st.subheader("`Evaluation`")

st.write(
    "*Evaluate and compare the performance of different LLMs using a predefined test dataset.*"
)

st.markdown("""
1. Navigate to the **Evaluation** page and run the evaluation pipeline.
2. Open the **Dashboard** to analyze key metrics such as **Cosine Similarity**, **Latency**, **Token Usage**, and compare the generated responses with the reference answers.
3. Use the **A/B Testing** page to compare multiple evaluation runs and identify which LLM or configuration performs best for your use case.
""")