import streamlit as st
import sqlite3
import pandas as pd
import json
import os
import sys
from app.ui_utils import load_fluent_css, check_password

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

st.set_page_config(page_title="Chat History", page_icon="📋", layout="wide")
if not check_password():
    st.stop()

load_fluent_css()
st.title("📋 Chat History & Data Flywheel")
st.markdown("View past interactions, analyze traces, and promote failed conversations to the eval dataset.")

db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/eval_results.db'))

if not os.path.exists(db_path):
    st.warning("Database not found. Please initiate a chat first to create the database.")
    st.stop()

# Load Data
@st.cache_data(ttl=5) # Refresh every 5 seconds
def load_data():
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM chat_logs ORDER BY timestamp DESC", conn)
    conn.close()
    return df

df = load_data()

if df.empty:
    st.info("No chat logs found.")
    st.stop()

# Filters
col1, col2 = st.columns(2)
with col1:
    intent_filter = st.selectbox("Filter by Intent", ["All"] + list(df['router_intent'].unique()))
with col2:
    escalated_filter = st.selectbox("Filter by Escalated Status", ["All", "Escalated", "Not Escalated"])

# Apply Filters
filtered_df = df.copy()
if intent_filter != "All":
    filtered_df = filtered_df[filtered_df['router_intent'] == intent_filter]
if escalated_filter != "All":
    is_esc = True if escalated_filter == "Escalated" else False
    filtered_df = filtered_df[filtered_df['escalated'] == is_esc]

# Display Table
action_placeholder = st.empty()

event = st.dataframe(
    filtered_df[['id', 'timestamp', 'user_message', 'router_intent', 'escalated', 'latency_ms', 'tokens_used']],
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="multi-row"
)

selected_rows = event.selection.rows

if selected_rows:
    selected_ids = filtered_df.iloc[selected_rows]['id'].tolist()
    with action_placeholder:
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"🗑️ Delete {len(selected_ids)} Selected Chats", type="primary", use_container_width=True):
                from evals.db import delete_chat_logs
                delete_chat_logs(selected_ids)
                load_data.clear()
                st.rerun()
        
        with col2:
            if st.button(f"📤 Promote {len(selected_ids)} Selected Chats", type="secondary", use_container_width=True):
                st.session_state.show_promotion = True
                st.session_state.selected_promotion_ids = selected_ids
                st.rerun()

# Initialize promotion state
if "show_promotion" not in st.session_state:
    st.session_state.show_promotion = False
if "selected_promotion_ids" not in st.session_state:
    st.session_state.selected_promotion_ids = []
if "reference_answers" not in st.session_state:
    st.session_state.reference_answers = {}

# Promotion Interface
if st.session_state.show_promotion and st.session_state.selected_promotion_ids:
    st.divider()
    st.subheader("📤 Promote Chats to Dataset")
    
    # Load the selected chat logs
    from evals.promote import get_chat_logs_by_ids, preview_promotion
    selected_chats = get_chat_logs_by_ids(st.session_state.selected_promotion_ids)
    
    if not selected_chats:
        st.error("No chat logs found for the selected IDs.")
        st.session_state.show_promotion = False
        st.rerun()
    
    # Generate preview of test cases
    preview_cases = preview_promotion(st.session_state.selected_promotion_ids)
    
    st.markdown(f"**Preview of {len(selected_chats)} chat(s) to be promoted:**")
    st.info("Edit the reference answers below. These will become part of the extended evaluation dataset.")
    
    # Display each chat for editing
    for i, (chat, preview_case) in enumerate(zip(selected_chats, preview_cases)):
        chat_id = chat['id']
        test_case_id = preview_case['id']
        
        with st.expander(f"Chat #{i+1}: {test_case_id} - {preview_case['category']}", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original User Query:**")
                st.code(chat['user_message'], language=None)
                
                st.markdown("**Original Bot Response:**")
                st.info(chat['bot_response'])
                
                # Show metadata inline (not too much data)
                st.markdown("**Metadata:**")
                st.code(f"""
Test Case ID: {test_case_id}
Category: {preview_case['category']}
Tags: {', '.join(preview_case['tags'])}
Difficulty: {preview_case['difficulty']}
Original Time: {chat['timestamp']}
Router Intent: {chat['router_intent']}
""", language=None)
            
            with col2:
                st.markdown("**Reference Answer (Required):**")
                
                # Initialize reference answer with bot response if not already set
                if chat_id not in st.session_state.reference_answers:
                    st.session_state.reference_answers[chat_id] = chat['bot_response']
                
                # Text area for editing reference answer
                reference_answer = st.text_area(
                    "Edit the reference answer that will be used for evaluation:",
                    value=st.session_state.reference_answers[chat_id],
                    height=150,
                    key=f"ref_{chat_id}",
                    label_visibility="collapsed"
                )
                
                # Update session state
                st.session_state.reference_answers[chat_id] = reference_answer
                
                # Character count
                char_count = len(reference_answer)
                st.caption(f"Characters: {char_count}")
                
                # Validation warning if empty
                if not reference_answer.strip():
                    st.warning("⚠️ Reference answer cannot be empty")
    
    # Check for duplicates
    from evals.promote import get_existing_queries
    existing_queries = get_existing_queries()
    
    duplicate_warnings = []
    for chat in selected_chats:
        from evals.promote import find_similar_queries
        similar = find_similar_queries(chat['user_message'], existing_queries, threshold=0.8)
        if similar:
            duplicate_warnings.append(f"**CHAT-{chat['id']}**: Similar to existing query: '{similar[0][0][:50]}...' (similarity: {similar[0][1]:.2f})")
    
    if duplicate_warnings:
        st.warning("### ⚠️ Duplicate Detection")
        for warning in duplicate_warnings:
            st.markdown(f"- {warning}")
        st.info("You can still promote these, but consider editing the reference answer to make it unique.")
    
    # Action buttons
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.show_promotion = False
            st.session_state.reference_answers = {}
            st.rerun()
    
    with col3:
        # Check if all reference answers are filled
        all_filled = True
        for chat_id in st.session_state.selected_promotion_ids:
            if chat_id not in st.session_state.reference_answers or not st.session_state.reference_answers[chat_id].strip():
                all_filled = False
                break
        
        if st.button("✅ Confirm Promotion", type="primary", use_container_width=True, disabled=not all_filled):
            try:
                # Prepare reference answers dictionary
                ref_answers_dict = {}
                for chat_id in st.session_state.selected_promotion_ids:
                    if chat_id in st.session_state.reference_answers:
                        ref_answers_dict[chat_id] = st.session_state.reference_answers[chat_id]
                
                # Call promotion function
                from evals.promote import promote_chats_to_dataset
                success_count, messages = promote_chats_to_dataset(
                    st.session_state.selected_promotion_ids,
                    ref_answers_dict
                )
                
                if success_count > 0:
                    st.success(f"✅ Successfully promoted {success_count} chat(s) to the extended dataset!")
                    
                    # Show additional messages
                    for msg in messages[1:]:  # Skip the first success message
                        st.info(msg)
                    
                    # Clear state
                    st.session_state.show_promotion = False
                    st.session_state.selected_promotion_ids = []
                    st.session_state.reference_answers = {}
                    
                    # Refresh data
                    load_data.clear()
                    
                    # Show next steps
                    st.info("💡 **Next Steps:** Go to the Evaluation page and select 'extended' or 'both' dataset type to evaluate these new test cases.")
                    
                    # Small delay before rerun
                    import time
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error(f"Failed to promote chats: {messages[0] if messages else 'Unknown error'}")
                    
            except Exception as e:
                st.error(f"Error during promotion: {str(e)}")
                st.exception(e)

st.divider()

# Detail View
st.subheader("🔍 Debugger")

if selected_rows:
    # Get the row directly from the filtered dataframe using the visual index of the FIRST selection
    row_idx = selected_rows[0]
    row = filtered_df.iloc[row_idx]
    
    with st.chat_message("user"):
        st.write(row['user_message'])
        
    with st.chat_message("assistant"):
        st.write(row['bot_response'])
    
    with st.expander("Raw Trace Data (JSON)", expanded=True):
        try:
            trace_json = json.loads(row['trace_data'])
            st.json(trace_json)
        except:
            st.text(row['trace_data'])
