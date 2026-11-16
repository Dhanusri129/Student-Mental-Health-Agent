import streamlit as st
import joblib
import numpy as np
import pandas as pd

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="MindCare | Student Wellness",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# LOAD EXISTING PROJECT FILES
# Keep these files in the same folder as app.py:
# mental_health_model.pkl, scaler.pkl,
# label_encoders.pkl, Student Mental health.csv
# -------------------------------------------------
@st.cache_resource
def load_model_files():
    model = joblib.load("mental_health_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("label_encoders.pkl")
    return model, scaler, encoders

@st.cache_data
def load_dataset():
    return pd.read_csv("Student Mental health.csv")

try:
    model, scaler, encoders = load_model_files()
    df = load_dataset()
except Exception as error:
    st.error("Project files could not be loaded.")
    st.info(
        "Check that mental_health_model.pkl, scaler.pkl, "
        "label_encoders.pkl and Student Mental health.csv "
        "are present in the same folder as app.py."
    )
    st.caption(f"Technical details: {error}")
    st.stop()

# -------------------------------------------------
# CUSTOM DESIGN
# -------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #f7f5ff 0%, #f3f8ff 55%, #ffffff 100%);
        color: #30265f !important;
    }

    /* Force readable dark text when the Streamlit account theme is dark. */
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] label,
    [data-testid="stAppViewContainer"] li,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stCaptionContainer"] {
        color: #39335f !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #39335f !important;
    }

    [data-testid="stSidebar"] {
        background: #f0edff;
        border-right: 1px solid #e5defd;
    }

    .hero {
        padding: 28px 30px;
        border-radius: 22px;
        background: linear-gradient(115deg, #e8e1ff 0%, #e3efff 100%);
        border: 1px solid #ded7fb;
        margin-bottom: 22px;
    }

    .hero h1 {
        color: #30265f;
        font-size: 2.25rem;
