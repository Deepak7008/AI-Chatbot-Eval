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
