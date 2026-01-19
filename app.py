import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="DiamondsDynamics", page_icon="💎", layout="centered")

st.title("💎 DiamondsDynamics")
st.write("Price prediction + market segmentation (clustering)")

# -----------------------------
# Load models (cached)
# -----------------------------
@st.cache_resource
def load_artifacts():
    reg_model = joblib.load("models/XGBoost.pkl")
    kmeans = joblib.load("models/Kmean_k2_market_segment.pkl")
    scaler = joblib.load("models/cluster_scaler.pkl")
    return reg_model, kmeans, scaler

reg_model, kmeans_model, cluster_scaler = load_artifacts()

# -----------------------------
# Encoding maps (ordinal)
# -----------------------------
cut_order = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
color_order = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
clarity_order = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}
carat_category_order = {'Light': 0, 'Medium': 1, 'Heavy': 2}

def get_carat_category(carat: float) -> str:
    if carat < 0.5:
        return "Light"
    elif carat <= 1.5:
        return "Medium"
    return "Heavy"

# -----------------------------
# Helper: cluster naming for ANY K
# -----------------------------
def get_cluster_label(cluster_id: int, k: int):
    """
    Returns (segment_name, description) for a given cluster id.
    Works for K=2 and for larger K values (fallback names).
    """
    # Best-case: if K=2 (your intended segmentation)
    if k == 2:
        mapping = {
            0: ("Affordable Small Diamonds", "Budget-friendly segment with smaller sizes and moderate grades."),
            1: ("Premium Heavy Diamonds", "High-value segment with larger sizes and premium grades.")
        }
        return mapping.get(cluster_id, (f"Cluster {cluster_id}", "Unmapped cluster."))

    # If K > 2, show meaningful but generic names
    generic_names = [
        ("Affordable Small Diamonds", "Lower size and value-focused segment."),
        ("Mid-range Balanced Diamonds", "Balanced size and quality segment."),
        ("Premium Heavy Diamonds", "High size and premium segment."),
        ("High-Quality Medium Diamonds", "Medium size with higher quality grades."),
        ("Everyday Value Diamonds", "Common market segment with practical value."),
        ("Luxury Showcase Diamonds", "Top-end segment with standout attributes.")
    ]

    if cluster_id < len(generic_names):
        return generic_names[cluster_id]

    return (f"Cluster {cluster_id}", "This cluster is not yet mapped to a business-friendly name.")

# -----------------------------
# Inputs
# -----------------------------
st.subheader("🔧 Enter Diamond Attributes")

col1, col2 = st.columns(2)

with col1:
    carat = st.number_input("Carat", min_value=0.01, value=0.70, step=0.01)
    table = st.number_input("Table", min_value=0.0, value=57.0, step=0.1)

with col2:
    cut = st.selectbox("Cut", list(cut_order.keys()), index=4)
    color = st.selectbox("Color", ['D','E','F','G','H','I','J'], index=3)
    clarity = st.selectbox(
        "Clarity",
        ['IF','VVS1','VVS2','VS1','VS2','SI1','SI2','I1'],
        index=4
    )

# -----------------------------
# Feature builder
# -----------------------------
def build_features(carat, table, cut, color, clarity):
    carat_log = np.log1p(carat)
    cut_ord = cut_order[cut]
    color_ord = color_order[color]
    clarity_ord = clarity_order[clarity]

    carat_cat = get_carat_category(carat)
    carat_category_ord = carat_category_order[carat_cat]

    X = pd.DataFrame([{
        "carat_log": carat_log,
        "table": table,
        "cut_ord": cut_ord,
        "color_ord": color_ord,
        "clarity_ord": clarity_ord,
        "carat_category_ord": carat_category_ord
    }])

    return X, carat_cat, carat_log, cut_ord, color_ord, clarity_ord, carat_category_ord

# -----------------------------
# Buttons
# -----------------------------
st.divider()
btn1, btn2 = st.columns(2)

with btn1:
    if st.button("🎯 Predict Price", use_container_width=True):
        X, carat_cat, *_ = build_features(carat, table, cut, color, clarity)

        pred_log = float(reg_model.predict(X)[0])
        pred_price = np.expm1(pred_log)

        st.success(f"💰 Diamond Price: **{pred_price:,.2f} INR**")
        # st.caption(f"(Model predicts log-price internally and converts back for display | Carat Category: {carat_cat})")

with btn2:
    if st.button("🧩 Predict Market Segment", use_container_width=True):
        X, carat_cat, carat_log, cut_ord, color_ord, clarity_ord, carat_category_ord = build_features(
            carat, table, cut, color, clarity
        )

        # Scale and predict cluster
        X_scaled = cluster_scaler.transform(X)
        cluster_id = int(kmeans_model.predict(X_scaled)[0])

        # Show K for transparency (helps debug "Cluster 5" issues)
        k = int(getattr(kmeans_model, "n_clusters", 0))
        segment_name, segment_desc = get_cluster_label(cluster_id, k)

        st.success(f"🧩 Market Segment: **{segment_name}**")
        # st.write(f"**Cluster ID:** {cluster_id}  |  **K (Total clusters):** {k}")
        st.caption(segment_desc)

        # st.markdown("### 🔎 Your Diamond Summary")
        # st.write(f"- **Carat:** {carat}  (**{carat_cat}**)")
        # st.write(f"- **Cut:** {cut} (ord={cut_ord})")
        # st.write(f"- **Color:** {color} (ord={color_ord})")
        # st.write(f"- **Clarity:** {clarity} (ord={clarity_ord})")
        # st.write(f"- **Table:** {table}")
        # st.write(f"- **carat_log:** {carat_log:.4f}")

