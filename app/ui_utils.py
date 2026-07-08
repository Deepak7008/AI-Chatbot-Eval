import streamlit as st
import os

def load_fluent_css():
    """
    Loads the custom anthropic_style.css and injects it into the Streamlit app.
    Call this function right after st.set_page_config() on every page.
    """
    # Resolve the absolute path to app/assets/anthropic_style.css
    # Since this file (ui_utils.py) is in app/, assets is a sibling directory.
    css_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'assets', 'anthropic_style.css'))
    
    try:
        with open(css_path, 'r', encoding='utf-8') as f:
            css_content = f.read()
            st.markdown(f'<style>{css_content}</style>', unsafe_allow_html=True)
    except Exception as e:
        print(f"[UI] Warning: Could not load anthropic_style.css: {e}")

def check_password():
    """
    Returns `True` if the user had a correct password.
    If the password is not set in secrets, it assumes no protection is needed.
    """
    import hmac

    # Safely try to access secrets
    try:
        if "APP_PASSWORD" not in st.secrets:
            return True
        expected_password = st.secrets["APP_PASSWORD"]
    except FileNotFoundError:
        return True

    def login_form():
        """Form with widgets to collect password"""
        with st.form("Credentials"):
            st.text_input("Passcode", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], expected_password):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password in session state
        else:
            st.session_state["password_correct"] = False

    # Return True if the passcode was already validated
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password
    login_form()
    
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
        
    return False
