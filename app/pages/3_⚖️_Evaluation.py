import streamlit as st
import os
import sys
import json
import time
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.ui_utils import load_fluent_css, check_password
from agents.llm_client import MODEL_REGISTRY, get_display_names, resolve_model_from_display
from evals.cascade import run_eval_suite
from evals.bias_check import run_bias_check
from evals.judge import DIMENSIONS

st.set_page_config(page_title="Evaluation", page_icon="⚖️", layout="wide")
if not check_password():
    st.stop()

load_fluent_css()

if "eval_results" not in st.session_state:
    st.session_state.eval_results = None
if "bias_results" not in st.session_state:
    st.session_state.bias_results = None

def format_multi_turn_text(text: str, html_mode: bool = False) -> str:
    """Format JSON arrays of conversation turns into readable text."""
    if not isinstance(text, str):
        return str(text)
        
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            turns = json.loads(text)
            if isinstance(turns, list):
                formatted = []
                for i, turn in enumerate(turns):
                    if html_mode:
                        safe_turn = str(turn).replace('\n', '<br>')
                        formatted.append(f"<b>Turn {i+1}:</b> {safe_turn}")
                    else:
                        formatted.append(f"**Turn {i+1}:** {turn}")
                
                separator = "<br><br>" if html_mode else "\n\n"
                return separator.join(formatted)
        except json.JSONDecodeError:
            pass
            
    if html_mode:
        return text.replace('\n', '<br>')
    return text

st.title("⚖️ Evaluation Engine")
st.markdown("Run the LLM-as-a-Judge pipeline across test cases to measure performance.")

st.markdown("### ⚙️ Configuration")
col_c1, col_c2 = st.columns(2)

with col_c1:
    dataset_type = st.radio(
        "Dataset",
        options=["core", "extended", "both"],
        format_func=lambda x: x.title(),
        horizontal=True,
        help="Core = basic cases. Extended = edge cases."
    )

from evals.cascade import load_dataset
ds_preview = load_dataset(dataset_type)
all_cases = ds_preview["single"] + ds_preview["multi"]
unique_cats = sorted(list(set(c.get("category", "unknown") for c in all_cases)))

with col_c2:
    selected_categories = st.multiselect(
        "Categories to Run (Check to select)",
        options=unique_cats,
        default=unique_cats,
        )

# Eval runs without user authentication - uses order lookup by ID directly
user_email = None

st.divider()

# --- MAIN AREA ---
tab_eval, tab_bias = st.tabs(["🚀 Run Suite", "🔍 Bias Check"])

with tab_eval:
    # Check if a judge model is configured
    judge_provider = os.getenv("EVAL_JUDGE_PROVIDER", "")
    judge_model = os.getenv("EVAL_JUDGE_MODEL", "")
    
    if not judge_provider or not judge_model:
        st.error("⚠️ **No valid judge model configured.** Please go to 🛍️ Setup page to select a judge model.")
        run_button_disabled = True
        run_button_tooltip = "Disabled: No judge model configured"
    else:
        run_button_disabled = False
        run_button_tooltip = "Run the evaluation suite"
    
    if st.button("▶️ Run Evaluation Suite", type="primary", use_container_width=True, 
                 disabled=run_button_disabled, help=run_button_tooltip):
        st.session_state.eval_results = None # Clear previous
        
        progress_bar = st.progress(0, text="Initializing...")
        log_expander = st.expander("Live Execution Log", expanded=True)
        log_container = log_expander.empty()
        log_text = ""
        
        st.markdown("### 📡 Results")
        live_table_container = st.empty()
        live_results = []
        
        def update_progress(current, total, case_id, status_msg="Running...", result_data=None):
            global log_text
            progress = current / total if total > 0 else 0
            progress_bar.progress(progress, text=f"Processing {current}/{total}: {case_id}")
            timestamp = time.strftime("%H:%M:%S")
            log_text += f"[{timestamp}] [{current}/{total}] {case_id}: {status_msg}\n"
            log_container.code(log_text, language="bash")
            
            if result_data:
                disp_data = result_data.copy()
                disp_data["Latency (s)"] = disp_data.get("latency_ms", 0) / 1000.0
                live_results.append(disp_data)
                df = pd.DataFrame(live_results)
                if not df.empty:
                    display_cols = ["case_id", "category", "pass_fail", "weighted_score", "Latency (s)"]
                    live_table_container.dataframe(
                        df[[c for c in display_cols if c in df.columns]], 
                        use_container_width=True, 
                        hide_index=True
                    )
                
        start_time = time.time()
        
        try:
            results = run_eval_suite(
                dataset_type=dataset_type,
                provider=None,
                model=None,
                judge_provider=None,
                judge_model=None,
                user_email=user_email,
                progress_callback=update_progress,
                categories=selected_categories
            )
            st.session_state.eval_results = results
            progress_bar.progress(1.0, text="✅ Evaluation Complete!")
            log_text += f"\n[{time.strftime('%H:%M:%S')}] ✅ Evaluation Complete!\n"
            log_container.code(log_text, language="bash")
        except Exception as e:
            st.error(f"Evaluation failed: {str(e)}")
            progress_bar.empty()

    # --- Display Results ---
    if st.session_state.eval_results:
        res = st.session_state.eval_results
        
        st.subheader("📊 Run Summary")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Cases", res["total_cases"])
        pr_val = res['pass_rate'] if res['pass_rate'] > 1.0 else res['pass_rate'] * 100
        col2.metric("Pass Rate", f"{pr_val:.2f}%", f"{res['passed']} passed, {res['failed']} failed")
        col3.metric("Avg Score", f"{res['avg_score']:.2f} / 5.0")
        col4.metric("Total Cost", f"${res['total_cost']:.4f}")
        col5.metric("Total Time", f"{res['total_time_sec'] / 60:.1f} min")
        
        st.divider()
        st.subheader("📋 Case Details")
        
        for case in res["results"]:
            case_id = case["case_id"]
            passed = case["pass_fail"] == "PASS"
            icon = "✅" if passed else "❌"
            score = case["weighted_score"]
            sim = case["cosine_similarity"]
            
            with st.expander(f"{icon} {case_id} — Score: {score:.2f} | Sim: {sim:.2f}"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**User Query:**")
                    st.info(case["query"])
                    st.markdown("**Reference Answer:**")
                    st.success(case["reference_answer"])
                with c2:
                    st.markdown("**Actual Answer:**")
                    border_color = "green" if passed else "red"
                    st.markdown(
                        f"<div style='border: 2px solid {border_color}; padding: 10px; border-radius: 5px;'>"
                        f"{case['actual_answer']}</div>", 
                        unsafe_allow_html=True
                    )
                
                st.markdown("### ⚖️ Judge Reasoning")
                try:
                    reasoning = json.loads(case["judge_reasoning_json"])
                    for dim in DIMENSIONS:
                        d_score = case.get(dim, "N/A")
                        d_reason = reasoning.get(dim, "No reasoning provided.")
                        
                        # Highlight hard-gate failures
                        if dim in ["accuracy", "safety"] and isinstance(d_score, (int, float)) and d_score < 3:
                            st.error(f"**{dim.title()} ({d_score}/5):** {d_reason}")
                        else:
                            st.markdown(f"**{dim.title()} ({d_score}/5):** {d_reason}")
                except Exception as e:
                    st.warning(f"Could not parse judge reasoning: {e}")

with tab_bias:
    st.markdown("""
    **Position Bias Check:** Runs the judge twice on a subset of cases (Normal vs Swapped ordering) 
    to ensure the judge is scoring the content, not the position.
    """)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        sample_size = st.number_input("Sample Size", min_value=1, max_value=20, value=3)
    
    if st.button("🔍 Run Bias Check", use_container_width=True):
        st.session_state.bias_results = None
        
        progress_bar = st.progress(0, text="Initializing Bias Check...")
        
        # Load a small slice of the dataset
        import random
        from evals.cascade import load_dataset
        ds = load_dataset(dataset_type)
        all_cases = ds["single"]  # Bias check evaluates isolated QA pairs, so only use single-turn cases
        
        # Apply global category filter
        filtered_cases = [c for c in all_cases if c.get("category", "unknown") in selected_categories]
        if not filtered_cases:
            st.warning("No cases match the selected categories. Please select different categories in the Configuration section.")
            st.stop()
            
        test_cases = random.sample(filtered_cases, min(sample_size, len(filtered_cases)))
        
        live_bias_container = st.empty()
        bias_live_data = []
        
        def update_bias_progress(current, total, case_id, result=None):
            progress_bar.progress(current / total, text=f"Checking bias {current}/{total}: {case_id}")
            
            if result:
                bias_live_data.append({
                    "Case ID": case_id,
                    "Base Score": f"{result['normal_weighted']:.2f}",
                    "Swapped Score": f"{result['swapped_weighted']:.2f}",
                    "Delta": f"{result['normal_weighted'] - result['swapped_weighted']:.2f}"
                })
                live_bias_container.dataframe(pd.DataFrame(bias_live_data), use_container_width=True, hide_index=True)
            
        try:
            b_results = run_bias_check(
                test_cases=test_cases,
                provider=None,
                model=None,
                progress_callback=update_bias_progress
            )
            st.session_state.bias_results = b_results
            progress_bar.progress(1.0, text="✅ Bias Check Complete!")
        except Exception as e:
            st.error(f"Bias check failed: {str(e)}")
            progress_bar.empty()
            
    if st.session_state.bias_results:
        b = st.session_state.bias_results
        
        # Verdict color mapping
        verdict_color = "green" if b["verdict"] == "LOW_BIAS" else "orange" if b["verdict"] == "MODERATE_BIAS" else "red"
        
        st.markdown(f"### Verdict: <span style='color:{verdict_color}'>{b['verdict']}</span>", unsafe_allow_html=True)
        st.info(b["interpretation"])
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Abs Delta", f"{b['avg_abs_delta']:.3f}")
        c2.metric("Max Abs Delta", f"{b['max_abs_delta']:.3f}")
        c3.metric("Direction", b["bias_direction"].title())
        

        with st.expander("Detailed Case Data"):
            for case_idx, case_data in enumerate(b["cases"]):
                st.markdown(f"#### Case: `{case_data['case_id']}`")
                
                # Fetch original text from dataset to display Reference vs Actual
                # (The b_results doesn't carry the raw query/answers back to save memory, so we reconstruct)
                from evals.cascade import load_dataset
                ds = load_dataset("core")
                all_cases = ds["single"] + ds["multi"]
                original_case = next((tc for tc in all_cases if tc.get("id", f"case_{case_idx+1}") == case_data["case_id"]), None)
                
                if original_case:
                    st.markdown("**Query:**")
                    st.info(format_multi_turn_text(original_case.get('query', '')))
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Reference Answer:**")
                        st.write(format_multi_turn_text(original_case.get('reference_answer', 'N/A')))
                        st.markdown(f"**Base Score (Ref First): {case_data['normal_weighted']:.2f}**")
                    with c2:
                        st.markdown("**Actual Answer:**")
                        st.write(format_multi_turn_text(original_case.get('actual_answer', original_case.get('reference_answer', 'N/A'))))
                        st.markdown(f"**Swapped Score (Actual First): {case_data['swapped_weighted']:.2f}**")
                        
                        st.markdown(f"**Delta (Base - Swapped): {case_data['deltas'].get('overall', case_data['normal_weighted'] - case_data['swapped_weighted']):.2f}**")
                        
                    # Build per-dimension table
                    dim_data = []
                    dims = list(case_data['normal_scores'].keys())
                    
                    row_base = {"Metric": "Base Score"}
                    row_swapped = {"Metric": "Swapped Score"}
                    row_delta = {"Metric": "Delta"}
                    
                    for dim in dims:
                        d_name = dim.title()
                        n_score = case_data['normal_scores'].get(dim, 0)
                        s_score = case_data['swapped_scores'].get(dim, 0)
                        
                        row_base[d_name] = n_score
                        row_swapped[d_name] = s_score
                        row_delta[d_name] = n_score - s_score
                        
                    row_base["Weighted Score"] = f"{case_data['normal_weighted']:.2f}"
                    row_swapped["Weighted Score"] = f"{case_data['swapped_weighted']:.2f}"
                    row_delta["Weighted Score"] = f"{case_data['normal_weighted'] - case_data['swapped_weighted']:.2f}"
                    
                    dim_data.extend([row_base, row_swapped, row_delta])
                    
                    st.markdown("**Per-Dimension Breakdown:**")
                    st.table(pd.DataFrame(dim_data).set_index("Metric"))
                else:
                    st.markdown(f"- **Base Score (Ref First):** {case_data['normal_weighted']:.2f}")
                    st.markdown(f"- **Swapped Score (Actual First):** {case_data['swapped_weighted']:.2f}")
                    st.markdown(f"- **Delta:** {case_data['deltas'].get('overall', case_data['normal_weighted'] - case_data['swapped_weighted']):.2f}")
                
                st.divider()
