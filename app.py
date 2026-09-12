"""
app.py - House Price Prediction - Streamlit Web Application
Author  : House Price Prediction Project
Purpose : Production-ready interactive UI to predict house prices
          using the trained scikit-learn pipeline (best_model.pkl).
"""

import os
import sys

# Make sure the project root is on the Python path so joblib can resolve the Model package namespaces
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="House Price Prediction System",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "Model", "best_model.pkl")

BINARY_OPTIONS = ["No", "Yes"]
FURNISHING_OPTIONS = ["Unfurnished", "Semi-Furnished", "Furnished"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading prediction model…")
def load_model(model_path: str):
    """Load the serialised sklearn pipeline from disk (cached across sessions)."""
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


def yes_no(option: str) -> str:
    """Convert UI 'Yes'/'No' label back to dataset encoding 'yes'/'no'."""
    return option.lower()


def furnish_label(option: str) -> str:
    """Convert UI furnishing label to dataset encoding."""
    return option.lower()


def format_price(value: float) -> str:
    """Format price as a human-readable currency string."""
    if value >= 1_000_000:
        return f"₹ {value / 1_000_000:.2f} Million"
    return f"₹ {value:,.0f}"


def build_input_df(
    area, bedrooms, bathrooms, stories, parking,
    mainroad, guestroom, basement, hotwaterheating,
    airconditioning, prefarea, furnishingstatus
) -> pd.DataFrame:
    """Assemble raw input dictionary into a single-row DataFrame matching training schema."""
    data = {
        "area":            [int(area)],
        "bedrooms":        [int(bedrooms)],
        "bathrooms":       [int(bathrooms)],
        "stories":         [int(stories)],
        "mainroad":        [yes_no(mainroad)],
        "guestroom":       [yes_no(guestroom)],
        "basement":        [yes_no(basement)],
        "hotwaterheating": [yes_no(hotwaterheating)],
        "airconditioning": [yes_no(airconditioning)],
        "parking":         [int(parking)],
        "prefarea":        [yes_no(prefarea)],
        "furnishingstatus":[furnishingstatus.lower()],
    }
    return pd.DataFrame(data)


def validate_inputs(area, bedrooms, bathrooms, stories, parking) -> list[str]:
    """Return a list of validation error messages (empty list = valid)."""
    errors = []
    if area < 500:
        errors.append("🔴 Area must be at least 500 sq ft.")
    if area > 20_000:
        errors.append("🔴 Area seems too large (> 20,000 sq ft). Please verify.")
    if bedrooms < 1:
        errors.append("🔴 Number of bedrooms must be at least 1.")
    if bathrooms < 1:
        errors.append("🔴 Number of bathrooms must be at least 1.")
    if bathrooms > bedrooms * 2:
        errors.append("⚠️ Bathrooms count seems unusually high relative to bedrooms.")
    return errors


# ---------------------------------------------------------------------------
# Custom CSS – clean, professional, minimal
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---- Global font ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] .stSlider > label,
    [data-testid="stSidebar"] .stSelectbox > label { color: #94a3b8 !important; }

    /* ---- Prediction card ---- */
    .price-card {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(59, 130, 246, 0.35);
        margin-top: 1rem;
    }
    .price-card h2 { color: #bfdbfe; font-size: 1rem; font-weight: 600; margin: 0 0 .4rem 0; letter-spacing: .05em; text-transform: uppercase; }
    .price-card h1 { color: #ffffff; font-size: 2.8rem; font-weight: 700; margin: 0; }

    /* ---- Metric cards ---- */
    .metric-row { display: flex; gap: 1rem; margin-top: 1.2rem; }
    .metric-card {
        flex: 1;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-card .metric-label { font-size: .75rem; color: #64748b; text-transform: uppercase; letter-spacing: .05em; margin-bottom: .25rem; }
    .metric-card .metric-value { font-size: 1.25rem; font-weight: 700; color: #1e293b; }

    /* ---- Footer ---- */
    .footer {
        margin-top: 3rem;
        padding: 1.2rem;
        border-top: 1px solid #e2e8f0;
        text-align: center;
        color: #94a3b8;
        font-size: .82rem;
    }

    /* ---- Section headers ---- */
    h1 { color: #0f172a !important; }
    h3 { color: #1e293b !important; border-bottom: 2px solid #3b82f6; padding-bottom: .3rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
pipeline = load_model(MODEL_PATH)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
col_title, col_logo = st.columns([5, 1])
with col_title:
    st.title("🏠 House Price Prediction System")
    st.markdown(
        "Enter the house details in the **sidebar** and click **Predict Price** to get an instant estimate "
        "from our trained Machine Learning pipeline."
    )
with col_logo:
    st.markdown("<br>", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# Sidebar – Input Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📋 House Details")
    st.markdown("Fill in the characteristics of the house below.")
    st.markdown("---")

    st.markdown("### 📐 Physical Attributes")
    area = st.slider("Total Area (sq ft)", min_value=500, max_value=16_200, value=5_000, step=100)
    bedrooms  = st.selectbox("Bedrooms",  options=[1, 2, 3, 4, 5, 6], index=2)
    bathrooms = st.selectbox("Bathrooms", options=[1, 2, 3, 4],        index=0)
    stories   = st.selectbox("Stories",   options=[1, 2, 3, 4],        index=1)
    parking   = st.selectbox("Parking Spaces", options=[0, 1, 2, 3],   index=0)

    st.markdown("---")
    st.markdown("### 🏡 Amenities")
    mainroad        = st.radio("Main Road Access",      BINARY_OPTIONS, horizontal=True, index=0)
    guestroom       = st.radio("Guest Room",            BINARY_OPTIONS, horizontal=True, index=1)
    basement        = st.radio("Basement",              BINARY_OPTIONS, horizontal=True, index=1)
    hotwaterheating = st.radio("Hot Water Heating",     BINARY_OPTIONS, horizontal=True, index=1)
    airconditioning = st.radio("Air Conditioning",      BINARY_OPTIONS, horizontal=True, index=1)
    prefarea        = st.radio("Preferred Locality",    BINARY_OPTIONS, horizontal=True, index=1)

    st.markdown("---")
    st.markdown("### 🛋️ Furnishing Status")
    furnishingstatus = st.selectbox("Furnishing", FURNISHING_OPTIONS, index=1)

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Price", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Main panel – Output
# ---------------------------------------------------------------------------
if pipeline is None:
    st.error(
        "⚠️ Model file not found at `Model/best_model.pkl`. "
        "Please run the Jupyter Notebook first to train and save the model.",
        icon="🚨",
    )
    st.stop()

if predict_btn:
    # --- Validation ---
    errors = validate_inputs(area, bedrooms, bathrooms, stories, parking)
    if errors:
        for err in errors:
            st.warning(err)
        st.stop()

    # --- Inference ---
    input_df = build_input_df(
        area, bedrooms, bathrooms, stories, parking,
        mainroad, guestroom, basement, hotwaterheating,
        airconditioning, prefarea, furnishingstatus,
    )

    with st.spinner("Running prediction…"):
        predicted_price = float(pipeline.predict(input_df)[0])
        predicted_price = max(predicted_price, 0)   # guard against negative predictions

    # --- Output ---
    st.markdown(
        f"""
        <div class="price-card">
            <h2>Estimated House Price</h2>
            <h1>{format_price(predicted_price)}</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Supporting metrics ---
    area_per_bed = area / bedrooms
    luxury_score = sum([
        1 if mainroad == "Yes" else 0,
        1 if guestroom == "Yes" else 0,
        1 if basement == "Yes" else 0,
        1 if hotwaterheating == "Yes" else 0,
        1 if airconditioning == "Yes" else 0,
        1 if prefarea == "Yes" else 0,
    ])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Area per Bedroom", f"{area_per_bed:,.0f} sq ft")
    with c2:
        st.metric("Luxury Score", f"{luxury_score} / 6")
    with c3:
        st.metric("Total Rooms", bedrooms + bathrooms + stories)
    with c4:
        st.metric("Bath / Bed Ratio", f"{bathrooms / bedrooms:.2f}")

    # --- Input Summary ---
    with st.expander("📊 View Input Summary", expanded=False):
        st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

    # --- Disclaimer ---
    st.info(
        "ℹ️ **Disclaimer:** This prediction is based on historical data patterns. "
        "Actual market prices may vary depending on local market conditions, property condition, and current demand.",
        icon="ℹ️",
    )

else:
    # Idle state – show instructions
    st.markdown("### 👈 Fill in the sidebar and click **Predict Price**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1**\n\nEnter the physical details of the house (area, bedrooms, stories).")
    with col2:
        st.info("**Step 2**\n\nSelect the amenities available (AC, parking, guestroom, etc.).")
    with col3:
        st.info("**Step 3**\n\nClick **Predict Price** to get an instant ML-based price estimate.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
model_name = type(pipeline.named_steps["model"]).__name__
st.markdown(
    f"""
    <div class="footer">
        🏠 House Price Prediction System &nbsp;|&nbsp;
        Model: <strong>{model_name}</strong> &nbsp;|&nbsp;
        Built with Scikit-Learn, XGBoost &amp; Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
