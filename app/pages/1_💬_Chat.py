import streamlit as st
import time
import os
import sys
import uuid

# Ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agents.pipeline import run_pipeline
from evals.db import log_chat
from app.ui_utils import load_fluent_css

st.set_page_config(page_title="Chat", page_icon="💬", layout="wide")
load_fluent_css()
# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Header and Controls
col1, col2 = st.columns([8, 2])
with col1:
    st.title("💬 Support Bot")
    st.markdown("Start a conversation with our AI assistant. ")
with col2:
    st.write("") # Spacing
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# Display Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        content = msg["content"]
        if msg["role"] == "assistant":
            content = content.replace('$', r'\$')
        st.write(content)
        
        # Render the debug trace if it's an assistant message and has trace data
        if msg["role"] == "assistant" and "trace" in msg:
            trace = msg["trace"]
            with st.expander("🛠️ Debug Trace", expanded=False):
                st.markdown(f"**Intent:** `{trace.get('intent', 'N/A')}`")
                st.markdown(f"**Confidence:** `{trace.get('confidence', 0):.2f}`")
                st.markdown(f"**Router Reasoning:** {trace.get('reasoning', 'N/A')}")
                st.markdown(f"**Entities:** {trace.get('entities', {})}")
                
                if trace.get("steps"):
                    st.markdown("**Step-by-Step Processing:**")
                    st.table(trace["steps"])
                st.markdown(f"**Latency:** `{trace.get('latency_ms', 0)} ms`")
                st.markdown(f"**Tokens:** `{trace.get('tokens_used', 0)}`")

# Chat Input
if prompt := st.chat_input("Type your message here..."):
    # Render user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
        
    # Get history limit from env
    history_limit = int(os.getenv("CHAT_HISTORY_LIMIT", 5))
    
    # We only send past messages up to the limit (excluding the current prompt)
    raw_history = st.session_state.messages[-(history_limit*2+1):-1] 
    
    # Clean the history so it only contains 'role' and 'content'. 
    # LLM APIs will crash with a 400 error if we send them custom keys like 'trace'.
    history = [{"role": m["role"], "content": m["content"]} for m in raw_history]
    
    user_email = os.environ.get("CURRENT_USER_EMAIL", "")
    
    # Run the pipeline
    with st.chat_message("assistant"):
        with st.status("🧠 Processing...", expanded=True) as status:
            def update_status(step_info):
                status.write(f"✅ **{step_info['step']}**: {step_info['result']} *(Tokens: {step_info['tokens_used']}, {step_info['latency_ms']}ms)*")
                
            pipeline_result = run_pipeline(prompt, history=history, status_callback=update_status, user_email=user_email)
            status.update(label="✨ Response Ready!", state="complete", expanded=False)
            
        final_text = pipeline_result["text"]
        st.write(final_text.replace('$', r'\$'))
        
        # Build the trace dictionary
        trace = {
            "intent": pipeline_result.get("intent"),
            "confidence": pipeline_result.get("confidence"),
            "entities": pipeline_result.get("entities"),
            "latency_ms": pipeline_result.get("latency_ms"),
            "tokens_used": pipeline_result.get("tokens_used"),
            "guardrail_input_safe": pipeline_result.get("guardrail_input_safe"),
            "guardrail_input_reason": pipeline_result.get("guardrail_input_reason"),
            "guardrail_bypassed": pipeline_result.get("guardrail_bypassed"),
            "escalated": pipeline_result.get("escalated"),
            "reasoning": pipeline_result.get("reasoning"),
            "context": pipeline_result.get("context"),
            "raw_response": pipeline_result.get("raw_response"),
            "steps": pipeline_result.get("steps", [])
        }
        
        # Render trace
        with st.expander("🛠️ Debug Trace", expanded=True):
            st.markdown(f"**Model:** `{os.getenv('LLM_PROVIDER', 'groq').upper()} / {os.getenv('LLM_MODEL', 'llama-3.3-70b-versatile')}`")
            st.markdown(f"**Intent:** `{trace.get('intent', 'N/A')}`")
            st.markdown(f"**Confidence:** `{trace.get('confidence', 0):.2f}`")
            st.markdown(f"**Router Reasoning:** {trace.get('reasoning', 'N/A')}")
            st.markdown(f"**Entities:** {trace.get('entities', {})}")
            
            if trace.get("steps"):
                st.markdown("**Step-by-Step Processing:**")
                st.table(trace["steps"])
                
            st.markdown(f"**Latency:** `{trace.get('latency_ms', 0)} ms`")
            st.markdown(f"**Tokens:** `{trace.get('tokens_used', 0)}`")
                
        # Append to state
        st.session_state.messages.append({
            "role": "assistant", 
            "content": final_text,
            "trace": trace
        })
        
        provider = os.getenv("LLM_PROVIDER", "groq")
        model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

        # Save to database
        trace_data = {
            "provider": provider,
            "model_id": model,
            "router_intent": trace.get("intent"),
            "router_confidence": trace.get("confidence"),
            "router_reasoning": trace.get("reasoning"),
            "agent_used": trace.get("intent"), # using intent as proxy for agent
            "entities": trace.get("entities"),
            "guardrail_input_safe": trace.get("guardrail_input_safe"),
            "guardrail_input_reason": trace.get("guardrail_input_reason"),
            "guardrail_bypassed": trace.get("guardrail_bypassed"),
            "tokens_used": trace.get("tokens_used"),
            "latency_ms": trace.get("latency_ms"),
            "escalated": trace.get("escalated"),
            "raw_response": trace.get("raw_response"),
            "steps": trace.get("steps")
        }
        
        log_chat(
            session_id=st.session_state.session_id,
            user_message=prompt,
            bot_response=final_text,
            trace_data=trace_data
        )
