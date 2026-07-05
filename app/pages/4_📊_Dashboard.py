import streamlit as st
import os
import sys
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.ui_utils import load_fluent_css
from evals.db import get_eval_runs, get_eval_results
from evals.judge import DIMENSIONS
from evals.metrics import spearman_correlation, cohens_d, paired_ttest, bootstrap_ci

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
load_fluent_css()

st.title("📊 Analytics Dashboard")

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
                        # For HTML blocks (like Reference Answer), convert newlines to <br>
                        safe_turn = str(turn).replace('\n', '<br>')
                        formatted.append(f"<b>Turn {i+1}:</b> {safe_turn}")
                    else:
                        # For Markdown blocks, just use the raw newlines and bold syntax
                        formatted.append(f"**Turn {i+1}:** {turn}")
                
                separator = "<br><br>" if html_mode else "\n\n"
                return separator.join(formatted)
        except json.JSONDecodeError:
            pass
            
    # Single turn fallback
    if html_mode:
        return text.replace('\n', '<br>')
    return text


# Fetch past runs
runs = get_eval_runs()

if not runs:
    st.info("No evaluation runs found. Go to the **Evaluation** page to run your first test suite!")
    st.stop()

# Build a friendly run display name
run_options = {}
for r in runs:
    chat_model = r['model'].split('/')[-1] if '/' in r['model'] else r['model']
    eval_model = r['judge_model'].split('/')[-1] if '/' in r['judge_model'] else r['judge_model']
    pr_val = r['pass_rate'] if r['pass_rate'] > 1.0 else r['pass_rate'] * 100
    display = f"Run {r['id']} | eval:{eval_model} | chat:{chat_model} | {pr_val:.2f}% pass"
    run_options[r['id']] = display

# --- TAB SETUP ---
tab_overview, tab_compare = st.tabs(["📈 Run Overview", "⚔️ A/B Comparison"])

with tab_overview:
    # Select Run
    selected_run_id = st.selectbox("Select Run to Analyze", options=list(run_options.keys()), format_func=lambda x: run_options[x])
    
    # Get run metadata and results
    run_meta = next(r for r in runs if r["id"] == selected_run_id)
    results = get_eval_results(selected_run_id)
    df = pd.DataFrame(results)
    
    # --- Top KPIs ---
    st.markdown("### Key Performance Indicators")
    c1, c2, c3, c4, c5 = st.columns(5)
    pr_val = run_meta['pass_rate'] if run_meta['pass_rate'] > 1.0 else run_meta['pass_rate'] * 100
    c1.metric("Pass Rate", f"{pr_val:.2f}%", f"{run_meta['passed_cases']}/{run_meta['total_cases']} cases")
    
    # Format CI if it exists
    ci_text = ""
    if run_meta.get('ci_lower') and run_meta.get('ci_upper'):
        ci_text = f"CI: [{run_meta['ci_lower']:.2f}, {run_meta['ci_upper']:.2f}]"
    c2.metric("Avg Score", f"{run_meta['avg_score']:.2f} / 5", ci_text)
    
    c3.metric("Dataset", run_meta['dataset_used'].title())
    c4.metric("Total Cost", f"${run_meta['total_cost']:.4f}")
    c5.metric("Total Time", f"{run_meta.get('total_time_sec', 0) / 60:.1f} min")
    
    st.divider()
    
    if not df.empty:
        # --- Charts Row 1 ---
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.markdown("#### Category Performance (Pass Rate)")
            # Calculate pass rate per category
            cat_df = df.groupby('category').apply(
                lambda x: pd.Series({
                    'Pass Rate (%)': (x['pass_fail'] == 'PASS').mean() * 100,
                    'Cases': len(x)
                })
            ).reset_index()
            
            fig_cat = px.bar(
                cat_df, 
                x="category", 
                y="Pass Rate (%)",
                hover_data=["Cases"],
                color="Pass Rate (%)",
                color_continuous_scale=[[0, "#F87171"], [0.5, "#FBBF24"], [1, "#34D399"]],
                range_color=[0, 100]
            )
            fig_cat.update_layout(
                xaxis_title="Category", yaxis_title="Pass Rate (%)",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#EAEAF0", family="Inter"),
                xaxis=dict(gridcolor="#2A2A3C"), yaxis=dict(gridcolor="#2A2A3C")
            )
            st.plotly_chart(fig_cat, use_container_width=True)
            
        with col_c2:
            st.markdown("#### Score Distribution")
            fig_hist = px.histogram(
                df, x="weighted_score", nbins=10,
                color="pass_fail",
                color_discrete_map={"PASS": "#34D399", "FAIL": "#F87171"}
            )
            fig_hist.update_layout(
                xaxis_title="Weighted Score", yaxis_title="Count", barmode='stack',
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#EAEAF0", family="Inter"),
                xaxis=dict(gridcolor="#2A2A3C"), yaxis=dict(gridcolor="#2A2A3C")
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
        # --- Charts Row 2 ---
        st.divider()
        col_c3, col_c4 = st.columns(2)
        
        with col_c3:
            st.markdown("#### Cosine Similarity vs LLM Judge Score")
            fig_scatter = px.scatter(
                df, x="cosine_similarity", y="weighted_score",
                color="pass_fail", hover_data=["case_id", "category"],
                color_discrete_map={"PASS": "#34D399", "FAIL": "#F87171"}
            )
            # Add quadrants (rough heuristic)
            fig_scatter.add_hline(y=3.0, line_dash="dash", line_color="#555566", annotation_text="Judge Pass Threshold")
            fig_scatter.add_vline(x=0.8, line_dash="dash", line_color="#555566", annotation_text="Cosine Threshold")
            fig_scatter.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#EAEAF0", family="Inter"),
                xaxis=dict(gridcolor="#2A2A3C"), yaxis=dict(gridcolor="#2A2A3C")
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        with col_c4:
            st.markdown("#### Dimension Radar (Average Scores)")
            # Calculate average for each dimension
            dim_avgs = []
            for dim in DIMENSIONS:
                if dim in df.columns:
                    dim_avgs.append(df[dim].mean())
                else:
                    dim_avgs.append(0)
                    
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=dim_avgs + [dim_avgs[0]], # Close the loop
                theta=[d.title() for d in DIMENSIONS] + [DIMENSIONS[0].title()],
                fill='toself',
                mode='lines+markers+text',
                text=[f"{val:.1f}" for val in dim_avgs] + [f"{dim_avgs[0]:.1f}"],
                textposition="top center",
                textfont=dict(color="#EAEAF0", size=12),
                line_color='#3B82F6',
                fillcolor='rgba(59, 130, 246, 0.15)'
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[1, 5], gridcolor="#2A2A3C", color="#8888A0"),
                    angularaxis=dict(gridcolor="#2A2A3C", color="#EAEAF0")
                ),
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#EAEAF0", family="Inter")
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # --- Raw Data Explorer ---
        st.divider()
        st.markdown("### 🗄️ Raw Data Explorer")
        st.write("Filter test cases by score and explore the raw inputs and outputs.")
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            score_op = st.selectbox("Score Filter", options=["ALL", ">", "<", "=="])
        with col_f2:
            score_val = st.number_input("Score Value", min_value=0.0, max_value=5.0, value=3.0, step=0.1, disabled=(score_op == "ALL"))
            
        filtered_df = df.copy()
        if score_op == ">":
            filtered_df = filtered_df[filtered_df["weighted_score"] > score_val]
        elif score_op == "<":
            filtered_df = filtered_df[filtered_df["weighted_score"] < score_val]
        elif score_op == "==":
            filtered_df = filtered_df[filtered_df["weighted_score"] == score_val]
            
        if not filtered_df.empty:
            # Sort by case id
            filtered_df = filtered_df.sort_values("case_id", ascending=True)
            
            # Select columns to display
            display_cols = ["case_id", "category", "weighted_score", "pass_fail", "query", "actual_answer"]
            event = st.dataframe(
                filtered_df[display_cols],
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                column_config={
                    "case_id": "Test Case",
                    "category": "Category",
                    "weighted_score": st.column_config.NumberColumn("Score", format="%.2f"),
                    "pass_fail": "Status",
                    "query": "User Query",
                    "actual_answer": "Bot Answer"
                }
            )
            
            st.markdown("#### 🔍 Deep Dive")
            
            selected_rows = event.selection.rows
            
            if selected_rows:
                case_data = filtered_df.iloc[selected_rows[0]]
                
                def display_text(text):
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, list):
                            for i, t in enumerate(parsed):
                                st.markdown(f"**Turn {i+1}:** {t}")
                        else:
                            st.write(text)
                    except:
                        st.write(text)
                        
                st.markdown("**Query:**")
                display_text(case_data['query'])
                st.markdown("**Reference:**")
                display_text(case_data['reference_answer'])
                st.markdown("**Actual Answer:**")
                display_text(case_data['actual_answer'])
                
                st.markdown("##### Performance Metrics")
                metrics_data = {
                    "Weighted Score": f"{case_data['weighted_score']:.2f}",
                    "Cosine Sim": f"{case_data.get('cosine_similarity', 0):.2f}",
                    "Tokens": case_data.get('tokens_used', 0),
                    "Latency (s)": f"{case_data.get('latency_ms', 0) / 1000.0:.2f}",
                }
                for dim in DIMENSIONS:
                    metrics_data[dim.title()] = case_data.get(dim, "N/A")
                    
                st.dataframe(pd.DataFrame([metrics_data]), hide_index=True, use_container_width=True)
                
                st.markdown("##### Judge Reasoning")
                try:
                    reasoning_json = json.loads(case_data["judge_reasoning_json"])
                    if case_data.get("is_multi_turn", 0):
                        max_turns = max([int(k.split('_')[1]) for k in reasoning_json.keys() if k.startswith("turn_")] + [0])
                        for t in range(1, max_turns + 1):
                            st.markdown(f"###### Turn {t}")
                            for dim in DIMENSIONS:
                                key = f"turn_{t}_{dim}"
                                if key in reasoning_json:
                                    score = reasoning_json.get(f"{key}_score", "N/A")
                                    st.info(f"**{dim.title()} (Score: {score}/5):** {reasoning_json[key]}")
                    else:
                        for dim in DIMENSIONS:
                            if dim in reasoning_json:
                                score = case_data.get(dim, "N/A")
                                st.info(f"**{dim.title()} (Score: {score}/5):** {reasoning_json[dim]}")
                except Exception as e:
                    st.error(f"Could not parse judge reasoning: {e}")
                    st.code(case_data["judge_reasoning_json"])
            else:
                st.info("Select a test case in the table above to view deep dive details.")
                    
        else:
            st.info("No test cases match your filter criteria.")


with tab_compare:
    st.markdown("### ⚔️ Model A/B Comparison")
    st.write("Compare two runs statistically to see if one model is truly better than another.")
    
    c1, c2 = st.columns(2)
    with c1:
        run_a_id = st.selectbox("Select Run A (Latest / Challenger)", options=list(run_options.keys()), format_func=lambda x: run_options[x], key="run_a")
    with c2:
        run_b_id = st.selectbox("Select Run B (Old / Baseline)", options=list(run_options.keys()), format_func=lambda x: run_options[x], key="run_b")
        
    if run_a_id and run_b_id:
        if run_a_id == run_b_id:
            st.warning("Please select two different runs to compare.")
        else:
            df_a = pd.DataFrame(get_eval_results(run_a_id))
            df_b = pd.DataFrame(get_eval_results(run_b_id))
            
            # Find common cases
            common_cases = set(df_a['case_id']).intersection(set(df_b['case_id']))
            if len(common_cases) < 5:
                st.error(f"Not enough overlapping test cases ({len(common_cases)}) between these runs for valid statistical comparison.")
            else:
                st.success(f"Comparing performance across {len(common_cases)} matching test cases.")
                
                # Filter to matching cases and align them
                df_a_match = df_a[df_a['case_id'].isin(common_cases)].sort_values('case_id')
                df_b_match = df_b[df_b['case_id'].isin(common_cases)].sort_values('case_id')
                
                scores_a = df_a_match['weighted_score'].tolist()
                scores_b = df_b_match['weighted_score'].tolist()
                
                # Compute Stats
                d_result = cohens_d(scores_a, scores_b)
                t_result = paired_ttest(scores_a, scores_b)
                spearman_result = spearman_correlation(scores_a, scores_b)
                
                d_stat = d_result["d"]
                d_interp = d_result["interpretation"]
                t_stat = t_result["t_stat"]
                p_val = t_result["p_value"]
                t_interp = t_result["interpretation"]
                spearman_corr = spearman_result["rho"]
                
                # Get bootstrap CIs from run metadata
                run_a_meta = next(r for r in runs if r["id"] == run_a_id)
                run_b_meta = next(r for r in runs if r["id"] == run_b_id)
                
                # Format CI text for display
                ci_a_text = ""
                if run_a_meta.get('ci_lower') is not None and run_a_meta.get('ci_upper') is not None:
                    ci_a_text = f"[{run_a_meta['ci_lower']:.2f}, {run_a_meta['ci_upper']:.2f}]"
                else:
                    # Compute fresh CI if not in metadata
                    ci_a = bootstrap_ci(scores_a)
                    ci_a_text = f"[{ci_a['lower']:.2f}, {ci_a['upper']:.2f}]"
                    run_a_meta['ci_lower'] = ci_a['lower']
                    run_a_meta['ci_upper'] = ci_a['upper']
                
                ci_b_text = ""
                if run_b_meta.get('ci_lower') is not None and run_b_meta.get('ci_upper') is not None:
                    ci_b_text = f"[{run_b_meta['ci_lower']:.2f}, {run_b_meta['ci_upper']:.2f}]"
                else:
                    # Compute fresh CI if not in metadata
                    ci_b = bootstrap_ci(scores_b)
                    ci_b_text = f"[{ci_b['lower']:.2f}, {ci_b['upper']:.2f}]"
                    run_b_meta['ci_lower'] = ci_b['lower']
                    run_b_meta['ci_upper'] = ci_b['upper']
                
                # Compute bootstrap CI for difference (scores_a - scores_b)
                delta_scores = (np.array(scores_a) - np.array(scores_b)).tolist()
                diff_ci = bootstrap_ci(delta_scores)
                
                # Interpretation logic for difference CI
                diff_ci_lower = diff_ci["lower"]
                diff_ci_upper = diff_ci["upper"]
                diff_ci_width = diff_ci["width"]
                
                if diff_ci_lower > 0:
                    diff_interp = "Run A is significantly better (CI entirely positive)"
                    diff_color = "green"
                elif diff_ci_upper < 0:
                    diff_interp = "Run B is significantly better (CI entirely negative)"
                    diff_color = "red"
                elif diff_ci_lower <= 0 <= diff_ci_upper:
                    diff_interp = "No significant difference (CI includes zero)"
                    diff_color = "gray"
                else:
                    diff_interp = "Inconclusive"
                    diff_color = "orange"
                
                st.markdown("#### 🔬 Statistical Analysis")
                mc1, mc2, mc3 = st.columns(3)
                
                # Effect Size
                d_color = "normal" if d_stat > 0 else ("inverse" if d_stat < 0 else "off")
                mc1.metric("Effect Size (Cohen's d)", f"{d_stat:.2f}", d_interp, delta_color=d_color)
                
                # P-Value
                sig_text = "Statistically Significant" if p_val < 0.05 else "Not Significant (Noise)"
                mc2.metric("P-Value (T-Test)", f"{p_val:.4f}", sig_text, delta_color="off")
                
                # Correlation
                mc3.metric("Spearman Correlation", f"{spearman_corr:.2f}", "How similarly they ranked cases", 
                          help=f"Spearman's rho: {spearman_corr:.2f}. Measures rank agreement between runs")
                
                st.markdown("#### 📊 Confidence Intervals")
                dc1, dc2, dc3 = st.columns(3)
                
                with dc1:
                    # Difference CI
                    diff_ci_display = f"[{diff_ci_lower:.2f}, {diff_ci_upper:.2f}]"
                    diff_help = f"{diff_interp}. CI width: {diff_ci_width:.2f}. Lower = {diff_ci_lower:.2f}, Upper = {diff_ci_upper:.2f}"
                    diff_delta_color = "normal" if diff_color == "green" else "inverse" if diff_color == "red" else "off"
                    dc1.metric("Difference CI (A - B)", diff_ci_display, diff_interp, delta_color=diff_delta_color, help=diff_help)
                
                with dc2:
                    # Individual Run CIs
                    run_a_help = f"Run A average score: {run_a_meta.get('avg_score', 0):.2f} with 95% CI"
                    run_b_help = f"Run B average score: {run_b_meta.get('avg_score', 0):.2f} with 95% CI"
                    dc2.metric("Run A Avg CI", f"{run_a_meta.get('avg_score', 0):.2f}", ci_a_text, help=run_a_help)
                
                with dc3:
                    dc3.metric("Run B Avg CI", f"{run_b_meta.get('avg_score', 0):.2f}", ci_b_text, help=run_b_help)
                
                # # CI Width interpretation
                # st.markdown("#### 📏 Precision Analysis")
                # pa1, pa2 = st.columns(2)
                
                # with pa1:
                #     # CI width interpretation
                #     if diff_ci_width < 0.5:
                #         width_interp = "Precise estimate"
                #         width_color = "green"
                #     elif diff_ci_width < 1.0:
                #         width_interp = "Moderate precision"
                #         width_color = "orange"
                #     else:
                #         width_interp = "Low precision"
                #         width_color = "red"
                    
                #     pa1.metric("CI Width", f"{diff_ci_width:.2f}", width_interp, delta_color="off", 
                #                help=f"Narrower CI = more precise estimate of the true difference. Width < 0.5 is good")
                
                # with pa2:
                #     # Sample size adequacy
                #     n_cases = len(common_cases)
                #     if n_cases >= 30:
                #         sample_interp = "Adequate sample"
                #         sample_color = "green"
                #     elif n_cases >= 10:
                #         sample_interp = "Moderate sample"
                #         sample_color = "orange"
                #     else:
                #         sample_interp = "Small sample"
                #         sample_color = "red"
                    
                #     pa2.metric("Sample Size", n_cases, sample_interp, delta_color="off",
                #               help=f"Number of overlapping test cases. More cases = more reliable comparison")
                
                # Chart
                st.markdown("#### 📊 Score Delta (Run B - Run A)")
                delta_df = pd.DataFrame({
                    "case_id": df_a_match['case_id'],
                    "delta": df_b_match['weighted_score'].values - df_a_match['weighted_score'].values
                })
                # Sort by delta
                delta_df = delta_df.sort_values("delta")
                delta_df["color"] = delta_df["delta"].apply(lambda x: "green" if x > 0 else ("red" if x < 0 else "gray"))
                
                fig_bar = px.bar(
                    delta_df, x="case_id", y="delta", 
                    color="color", color_discrete_map={"green": "#34D399", "red": "#F87171", "gray": "#555566"}
                )
                fig_bar.update_layout(
                    showlegend=False, xaxis_title="Test Case", yaxis_title="Score Change (Positive = Run B Better)",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#EAEAF0", family="Inter"),
                    xaxis=dict(gridcolor="#2A2A3C"), yaxis=dict(gridcolor="#2A2A3C")
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # # Confidence Interval Visualization
                # st.markdown("#### 📈 Confidence Interval Comparison")
                
                # # Create data for CI visualization
                # ci_data = pd.DataFrame({
                #     "Group": ["Run A Average", "Run B Average", "Difference (B - A)"],
                #     "Value": [
                #         run_a_meta.get('avg_score', 0),
                #         run_b_meta.get('avg_score', 0),
                #         run_b_meta.get('avg_score', 0) - run_a_meta.get('avg_score', 0)
                #     ],
                #     "CI_Lower": [
                #         run_a_meta.get('ci_lower', run_a_meta.get('avg_score', 0)),
                #         run_b_meta.get('ci_lower', run_b_meta.get('avg_score', 0)),
                #         diff_ci_lower
                #     ],
                #     "CI_Upper": [
                #         run_a_meta.get('ci_upper', run_a_meta.get('avg_score', 0)),
                #         run_b_meta.get('ci_upper', run_b_meta.get('avg_score', 0)),
                #         diff_ci_upper
                #     ],
                #     "Color": ["blue", "purple", diff_color]
                # })
                
                # # Create CI plot
                # fig_ci = go.Figure()
                
                # # Add CI error bars
                # for i, row in ci_data.iterrows():
                #     # Add point estimate
                #     fig_ci.add_trace(go.Scatter(
                #         x=[row["Group"]],
                #         y=[row["Value"]],
                #         mode="markers",
                #         marker=dict(
                #             size=12,
                #             color=row["Color"],
                #             symbol="diamond"
                #         ),
                #         name=row["Group"],
                #         error_y=dict(
                #             type="data",
                #             symmetric=False,
                #             array=[row["CI_Upper"] - row["Value"]],
                #             arrayminus=[row["Value"] - row["CI_Lower"]],
                #             color=row["Color"],
                #             thickness=2,
                #             width=8
                #         ),
                #         hovertemplate=f"<b>{row['Group']}</b><br>" +
                #                      f"Value: {row['Value']:.2f}<br>" +
                #                      f"95% CI: [{row['CI_Lower']:.2f}, {row['CI_Upper']:.2f}]<br>" +
                #                      f"CI Width: {row['CI_Upper'] - row['CI_Lower']:.2f}<extra></extra>"
                #     ))
                
                # # Add zero line for difference plot reference
                # fig_ci.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5,
                #                 annotation_text="Zero Line", annotation_position="bottom right")
                
                # fig_ci.update_layout(
                #     title="95% Confidence Intervals Comparison",
                #     xaxis_title="Metric",
                #     yaxis_title="Value",
                #     showlegend=True,
                #     hovermode="x unified"
                # )
                
                # st.plotly_chart(fig_ci, use_container_width=True)




                # -------------------------------------------------------------
                # 🚨 A/B Divergence Explorer
                # -------------------------------------------------------------
                st.markdown("---")
                st.markdown("### 🚨 A/B Divergence Explorer")
                st.markdown("Investigate specific test cases where the two models disagreed on the evaluation score.")
                
                # Filters
                div_col1, div_col2, div_col3 = st.columns(3)
                
                # Get categories
                all_cats = ["All Categories"] + sorted(list(set(df_a_match['category'])))
                with div_col1:
                    sel_cat = st.selectbox("Category Filter", all_cats)
                    
                with div_col2:
                    div_type = st.selectbox("Divergence Type", [
                        "Most Controversial (Absolute Delta)",
                        "Top Improvements (Run A > Run B)",
                        "Top Regressions (Run A < Run B)"
                    ])
                    
                with div_col3:
                    div_limit = st.selectbox("Limit", [5, 10, 20, "All"])
                    
                # Merge data for detailed view
                div_df = pd.merge(
                    df_a_match, df_b_match,
                    on="case_id", suffixes=('_A', '_B')
                )
                div_df['delta'] = div_df['weighted_score_A'] - div_df['weighted_score_B']
                div_df['abs_delta'] = div_df['delta'].abs()
                
                # Apply Category Filter
                if sel_cat != "All Categories":
                    div_df = div_df[div_df['category_A'] == sel_cat]
                    
                # Apply Sort
                if div_type == "Most Controversial (Absolute Delta)":
                    div_df = div_df.sort_values("abs_delta", ascending=False)
                elif div_type == "Top Improvements (Run A > Run B)":
                    div_df = div_df.sort_values("delta", ascending=False)
                else: # Regressions
                    div_df = div_df.sort_values("delta", ascending=True)
                    
                # Apply Limit
                if div_limit != "All":
                    div_df = div_df.head(int(div_limit))
                    
                if div_df.empty:
                    st.info("No divergent cases found for these filters.")
                else:
                    for _, row in div_df.iterrows():
                        delta_val = row['delta']
                        icon = "🟢" if delta_val > 0 else "🔴" if delta_val < 0 else "⚪"
                        
                        with st.expander(f"{icon} {row['case_id']} | Delta: {delta_val:+.2f} | A: {row['weighted_score_A']:.2f} ➔ B: {row['weighted_score_B']:.2f}"):
                            st.markdown("**Query:**")
                            st.info(format_multi_turn_text(row['query_A']))
                            
                            st.markdown("**Reference Answer:**")
                            ref_formatted = format_multi_turn_text(row['reference_answer_A'], html_mode=True)
                            st.markdown(f"<div style='background-color: rgba(128, 128, 128, 0.15); padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;'>{ref_formatted}</div>", unsafe_allow_html=True)
                            
                            ans_c1, ans_c2 = st.columns(2)
                            with ans_c1:
                                st.markdown("##### Run A Answer")
                                ans_a_formatted = format_multi_turn_text(row['actual_answer_A'])
                                if delta_val < 0:
                                    st.success(ans_a_formatted)
                                elif delta_val > 0:
                                    st.error(ans_a_formatted)
                                else:
                                    st.write(ans_a_formatted)

                                st.markdown("**Judge Reasoning (Run A):**")
                                try:
                                    reasoning_a = json.loads(row['judge_reasoning_json_A'])
                                    for dim, text in reasoning_a.items():
                                        st.caption(f"**{dim}:** {text}")
                                except:
                                    st.caption(str(row['judge_reasoning_json_A']))
                                    
                            with ans_c2:
                                st.markdown("##### Run B Answer")
                                ans_b_formatted = format_multi_turn_text(row['actual_answer_B'])
                                if delta_val > 0:
                                    st.success(ans_b_formatted)
                                elif delta_val < 0:
                                    st.error(ans_b_formatted)
                                else:
                                    st.write(ans_b_formatted)

                                st.markdown("**Judge Reasoning (Run B):**")
                                try:
                                    reasoning_b = json.loads(row['judge_reasoning_json_B'])
                                    for dim, text in reasoning_b.items():
                                        st.caption(f"**{dim}:** {text}")
                                except:
                                    st.caption(str(row['judge_reasoning_json_B']))
