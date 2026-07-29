import streamlit as st
import os
import sys

# Ensure the root directory is in the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evals.db import init_db
from agents.llm_client import get_display_names, resolve_model_from_display, MODEL_REGISTRY
from agents.utils import load_json
from app.ui_utils import load_fluent_css, check_password

st.set_page_config(
    page_title="Setup",
    page_icon="⚙️",
    layout="wide"
)

load_fluent_css()

# Initialize database on first load
@st.cache_resource
def setup_db():
    init_db()

setup_db()

# ============================================================
# Header + Password Panel
# ============================================================

st.markdown(
    """
    <style>
    /* Reduce spacing below page title */
    .block-container h1{
        margin-bottom:0.2rem;
    }

    /* Remove extra space before H2 headers */
    .block-container h2{
        margin-top:0.4rem;
    }

    /* Reduce paragraph spacing */
    .block-container p{
        margin-bottom:0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# # Top Row
# title_col, password_col = st.columns([4.8, 2.2], vertical_alignment="top")

# with title_col:

st.title("Full-Stack AI Customer Support Chatbot + Evaluation Framework")


left_col, right_col = st.columns([4.5, 2], vertical_alignment="top")

with left_col:

    st.subheader("⚙️ System Configuration")

    st.caption(
        """
        Configure your LLM settings, evaluation model,
        chatbot behavior and application controls.
        """
    )

with right_col:
    st.markdown(
        """
        <div style="margin-top:-10px;">
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 🔒 Unlock Operations")
    import hmac

    if "password_feedback" not in st.session_state:
        st.session_state.password_feedback = None

    if st.session_state.get("access_unlocked"):

        st.success("✅ Token Operations Unlocked")

    else:

        feedback_text = ""

        if st.session_state.password_feedback == "wrong":
            feedback_text = "❌ Wrong password"

        elif st.session_state.password_feedback == "correct":
            feedback_text = "✅ Correct"

        with st.form("password_unlock_form", clear_on_submit=True):

            password = st.text_input(
                "",
                type="password",
                placeholder="Enter password...",
                key="setup_password_input",
                label_visibility="collapsed",
            )

            button_col, status_col = st.columns([1, 1.4])

            with button_col:
                submitted = st.form_submit_button(
                    "🔓 Unlock",
                    use_container_width=True,
                    type="primary",
                )

            with status_col:
                if feedback_text:
                    st.caption(feedback_text)

            if submitted and password:

                expected = st.secrets.get("APP_PASSWORD", "")

                if expected and hmac.compare_digest(password, expected):
                    st.session_state.access_unlocked = True
                    st.session_state.password_feedback = "correct"
                    st.rerun()

                elif not expected:
                    st.session_state.access_unlocked = True
                    st.session_state.password_feedback = None
                    st.rerun()

                else:
                    st.session_state.password_feedback = "wrong"

        if password and st.session_state.password_feedback == "wrong":
            st.session_state.password_feedback = None
    st.markdown("</div>", unsafe_allow_html=True)
# --- ROW 1: LLM Selection ---
st.subheader("LLM Selection")
top_col1, top_col2 = st.columns(2)

providers = list(dict.fromkeys(m["provider"] for m in MODEL_REGISTRY))

with top_col1:
    st.markdown("**Chatbot Model**")
    current_provider = os.getenv("LLM_PROVIDER", MODEL_REGISTRY[0]["provider"])
    if current_provider not in providers:
        current_provider = providers[0]
        
    selected_provider = st.selectbox("Chatbot Provider", options=providers, index=providers.index(current_provider))
    provider_models = [m for m in MODEL_REGISTRY if m["provider"] == selected_provider]
    display_names = [m["display_name"] for m in provider_models]
    current_model = os.getenv("LLM_MODEL", provider_models[0]["model_id"])
    
    default_idx = 0
    for i, m in enumerate(provider_models):
        if m["model_id"] == current_model:
            default_idx = i
            break
            
    selected_display = st.selectbox("Chatbot Active Model", options=display_names, index=default_idx)
    selected_model_dict = next(m for m in provider_models if m["display_name"] == selected_display)
    
    os.environ["LLM_PROVIDER"] = selected_model_dict["provider"]
    os.environ["LLM_MODEL"] = selected_model_dict["model_id"]
    
    # Store in session state for judge dropdown filtering
    st.session_state["chatbot_model"] = selected_model_dict

with top_col2:
    st.markdown("**Eval Judge Model**")
    current_judge_provider = os.getenv("EVAL_JUDGE_PROVIDER", providers[0] if providers else "")
    if current_judge_provider not in providers:
        current_judge_provider = providers[0]
        
    selected_judge_provider = st.selectbox("Judge Provider", options=providers, index=providers.index(current_judge_provider))
    
    # Filter judge models: exclude the exact chatbot model to prevent self-evaluation bias
    # Use chatbot model from session state (set in previous column)
    chatbot_model_dict = st.session_state.get("chatbot_model", None)
    
    if chatbot_model_dict is None:
        # First run - get from environment or use default
        chatbot_provider = os.getenv("LLM_PROVIDER", MODEL_REGISTRY[0]["provider"])
        chatbot_model = os.getenv("LLM_MODEL", MODEL_REGISTRY[0]["model_id"])
        # Find the matching dict
        for m in MODEL_REGISTRY:
            if m["provider"] == chatbot_provider and m["model_id"] == chatbot_model:
                chatbot_model_dict = m
                break
        if chatbot_model_dict is None:
            chatbot_model_dict = MODEL_REGISTRY[0]
    
    judge_provider_models = [
        m for m in MODEL_REGISTRY 
        if m["provider"] == selected_judge_provider 
        # Exclude if exact same provider AND model_id as chatbot
        and not (m["provider"] == chatbot_model_dict["provider"] and m["model_id"] == chatbot_model_dict["model_id"])
    ]
    
    # Handle empty dropdown scenario (e.g., chatbot uses only model from provider)
    if not judge_provider_models:
        st.error(f"⚠️ **No valid judge models available** for provider '{selected_judge_provider}'. Please select a different judge provider or change your chatbot model.")
        # Set empty environment variables to indicate no judge configured
        os.environ["EVAL_JUDGE_PROVIDER"] = ""
        os.environ["EVAL_JUDGE_MODEL"] = ""
        # Show disabled placeholder
        st.selectbox("Judge Active Model", 
                     options=["⚠️ No valid models - select different provider"], 
                     disabled=True)
        # Don't set judge model in environment (already set to empty)
    else:
        judge_display_names = [m["display_name"] for m in judge_provider_models]
        current_judge_model = os.getenv("EVAL_JUDGE_MODEL", judge_provider_models[0]["model_id"] if judge_provider_models else "")
        
        default_judge_idx = 0
        for i, m in enumerate(judge_provider_models):
            if m["model_id"] == current_judge_model:
                default_judge_idx = i
                break
                
        selected_judge_display = st.selectbox("Judge Active Model", options=judge_display_names, index=default_judge_idx)
        selected_judge_dict = next(m for m in judge_provider_models if m["display_name"] == selected_judge_display)
        
        os.environ["EVAL_JUDGE_PROVIDER"] = selected_judge_dict["provider"]
        os.environ["EVAL_JUDGE_MODEL"] = selected_judge_dict["model_id"]
        
        # Remove the warning - filtering already prevents self-evaluation bias
        # Keep this check as defensive programming
        chatbot_model_dict = st.session_state.get("chatbot_model", selected_model_dict)
        if chatbot_model_dict["provider"] == selected_judge_dict["provider"] and chatbot_model_dict["model_id"] == selected_judge_dict["model_id"]:
            st.error("⚠️ **CRITICAL ERROR**: Self-evaluation bias detected despite filtering. This should not happen. Please report this bug.")

st.divider()

# --- ROW 2: Context & Limits ---
mid_col1, mid_col2, mid_col3 = st.columns(3)

with mid_col1:
    st.subheader("👤 Persona")
    mock_db = load_json("mock_db.json")
    users = mock_db.get("users", [])
    user_options = ["None (Guest)"] + [f"{u['name']} ({u['email']})" for u in users]
    
    current_persona = st.session_state.get("current_persona", "None (Guest)")
    selected_persona = st.selectbox("Simulate Login As:", options=user_options, index=user_options.index(current_persona) if current_persona in user_options else 0)
    
    st.session_state["current_persona"] = selected_persona
    
    if selected_persona != "None (Guest)":
        import re
        email_match = re.search(r'\((.*?)\)', selected_persona)
        if email_match:
            os.environ["CURRENT_USER_EMAIL"] = email_match.group(1)
    else:
        os.environ["CURRENT_USER_EMAIL"] = ""

with mid_col2:
    st.subheader("Chatbot Tokens")
    chat_token_options = [256, 512, 1024, 2048, 4096]
    current_chat = int(os.getenv("CHAT_MAX_TOKENS", 1024))
    if current_chat not in chat_token_options:
        chat_token_options.append(current_chat)
        chat_token_options.sort()
    chat_tokens = st.selectbox(
        "Max Output Tokens", 
        options=chat_token_options, 
        index=chat_token_options.index(current_chat)
    )
    os.environ["CHAT_MAX_TOKENS"] = str(chat_tokens)

with mid_col3:
    st.subheader("Guardrail Tokens")
    guardrail_token_options = [50, 100, 150, 200, 250, 500]
    current_guardrail = int(os.getenv("GUARDRAIL_MAX_TOKENS", 150))
    if current_guardrail not in guardrail_token_options:
        guardrail_token_options.append(current_guardrail)
        guardrail_token_options.sort()
    guardrail_tokens = st.selectbox(
        "Scanner Space", 
        options=guardrail_token_options, 
        index=guardrail_token_options.index(current_guardrail)
    )
    os.environ["GUARDRAIL_MAX_TOKENS"] = str(guardrail_tokens)

st.divider()

# --- ROW 3: Controls ---
st.subheader("🎛️ Controls")
bot_col1, bot_col2 = st.columns(2)

with bot_col1:
    st.markdown("**Context Management**")
    history_limit = st.slider("Chat History Limit (Turns)", min_value=1, max_value=20, value=int(os.getenv("CHAT_HISTORY_LIMIT", 5)), step=1)
    os.environ["CHAT_HISTORY_LIMIT"] = str(history_limit)

with bot_col2:
    st.markdown("**Temperature (Creativity vs Determinism)**")
    current_temp = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    temperature = st.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=current_temp, step=0.1)
    os.environ["LLM_TEMPERATURE"] = str(temperature)
    


