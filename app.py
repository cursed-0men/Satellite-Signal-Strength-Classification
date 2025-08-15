import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------- PAGE SETTINGS -------------------
st.set_page_config(page_title="📊 Model Results Dashboard", layout="wide")

# Internal CSS with better contrast
st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background: linear-gradient(to right, #f4f4f4, #dfe6e9);
        color: #2d3436 !important;
    }
    h1, h2, h3, h4, h5, h6, p, label {
        color: #2d3436 !important;
    }
    .main-title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 20px;
    }
    .accuracy-card {
        background: #FF5252;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
        transition: transform 0.2s ease-in-out;
        color: #2d3436;
    }
    .accuracy-card:hover {
        transform: scale(1.02);
    }
    .accuracy-badge {
        font-size: 26px;
        font-weight: bold;
        padding: 8px 16px;
        border-radius: 8px;
        display: inline-block;
    }
    .high { background-color: #4CAF50; color: white; } /* Green */
    .medium { background-color: #FFA500; color: black; } /* Orange */
    .low { background-color: #F44336; color: white; } /* Red */
    .section-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 20px;
        transition: transform 0.2s ease-in-out;
        color: #2d3436;
    }
    .section-card:hover {
        transform: scale(1.01);
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- MODEL CONFIG -------------------
MODELS = {
    "Decision Tree": {"pred_file": "predictions/decision_tree_predictions.csv"},
    "KNN": {"pred_file": "predictions/knn_predictions.csv"},
    "Logistic Regression": {"pred_file": "predictions/logistic_regression_predictions.csv"}
}

# ------------------- SIDEBAR -------------------
st.markdown("""
    <style>
        /* Sidebar background */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #FF9D3D, #FF826B);
            color: white;
        }

        /* Sidebar title text */
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3, 
        [data-testid="stSidebar"] h4 {
            color: white !important;
            font-weight: bold;
        }

        /* Radio button labels */
        [data-testid="stSidebar"] label {
            color: white !important;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("⚙️ Model Selection")
selected_model = st.sidebar.radio("Choose a model", list(MODELS.keys()))

# ------------------- MAIN PAGE -------------------
st.markdown(f"<div class='main-title'>📈 Results for <span style='color:#0984e3;'>{selected_model}</span></div>", unsafe_allow_html=True)

# Load CSV
pred_path = MODELS[selected_model]["pred_file"]
try:
    df_pred = pd.read_csv(pred_path)
except FileNotFoundError:
    st.error(f"❌ Prediction file not found: {pred_path}")
    st.stop()

# Validate columns
if not {"True_Label", "Predicted_Label"}.issubset(df_pred.columns):
    st.error("❌ CSV must contain 'True_Label' and 'Predicted_Label' columns.")
    st.stop()

# ---------- Accuracy ----------
accuracy = np.mean(df_pred["True_Label"] == df_pred["Predicted_Label"]) * 100
if accuracy >= 80:
    color_class = "high"
elif accuracy >= 60:
    color_class = "medium"
else:
    color_class = "low"

with st.container():
    st.markdown(f"""
        <div class="accuracy-card">
            <h3>✅ Accuracy</h3>
            <div class="accuracy-badge {color_class}">{accuracy:.2f}%</div>
            <div style="background:#ddd; border-radius:10px; height:20px; margin-top:10px;">
                <div style="background:{'#4CAF50' if color_class=='high' else '#FFA500' if color_class=='medium' else '#F44336'};
                            height:20px; border-radius:10px; width:{accuracy}%;"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)


# ---------- Confusion Matrix ----------
labels = sorted(df_pred["True_Label"].unique())
conf_matrix = np.zeros((len(labels), len(labels)), dtype=int)

label_to_index = {label: idx for idx, label in enumerate(labels)}
for t, p in zip(df_pred["True_Label"], df_pred["Predicted_Label"]):
    conf_matrix[label_to_index[t], label_to_index[p]] += 1

with st.container():
    st.markdown("<div class='section-card'><h3>📊 Confusion Matrix</h3>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="flare",
                xticklabels=labels, yticklabels=labels, ax=ax,
                cbar=False, linewidths=1, linecolor='white', annot_kws={"size": 14, "color": "black"})
    ax.set_xlabel("Predicted", fontsize=12, color="#2d3436")
    ax.set_ylabel("True", fontsize=12, color="#2d3436")
    plt.xticks(rotation=45, color="#2d3436")
    plt.yticks(rotation=0, color="#2d3436")
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Predictions Preview ----------
with st.container():
    st.markdown("<div class='section-card'><h3>📄 Predictions Preview</h3>", unsafe_allow_html=True)
    st.dataframe(df_pred.head(20), use_container_width=True, height=300)
    st.markdown("</div>", unsafe_allow_html=True)
