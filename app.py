# import streamlit as st
# import pickle
# import numpy as np

# # Model aur scaler load karo
# with open('slp_regressor.pkl', 'rb') as f:
#     model = pickle.load(f)

# with open('scaler.pkl', 'rb') as f:
#     scaler = pickle.load(f)

# st.set_page_config(page_title="LPA Predictor", page_icon="💼")
# st.title("🎓 CGPA & IQ se LPA Predictor")
# st.write("Apna CGPA aur IQ daalo, model predict karega expected LPA (package)")
# iq = st.number_input("IQ", min_value=50, max_value=200, value=100, step=1)
# cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5, step=0.01)


# if st.button("Predict LPA"):
#     input_data = np.array([[iq, cgpa]])
#     scaled_input = scaler.transform(input_data)
#     prediction = model.predict(scaled_input)
    
#     lpa_value = float(prediction[0][0]) if prediction.ndim > 1 else float(prediction[0])
#     st.success(f"Predicted LPA: ₹{lpa_value:.2f} LPA")
import streamlit as st
import pickle
import numpy as np

# ---------------- Page Config ----------------
st.set_page_config(page_title="LPA Predictor", page_icon="🎯", layout="centered")

# ---------------- Load Model & Scaler ----------------
with open('slp_regressor.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
    /* Overall app background */
    .stApp {
        background-color: #0d1117;
    }

    /* Center content, limit width, responsive */
    .main .block-container {
        max-width: 480px;
        padding-top: 2.5rem;
        padding-bottom: 2rem;
    }

    /* Card wrapper */
    .card {
        background-color: #161b22;
        border: 1px solid #2a3140;
        border-radius: 18px;
        padding: 28px 26px 20px 26px;
    }

    /* Header row */
    .header-row {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 18px;
    }
    .icon-circle {
        width: 52px;
        height: 52px;
        border-radius: 50%;
        background-color: #21262d;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 26px;
        flex-shrink: 0;
    }
    .header-text h1 {
        font-size: 22px;
        font-weight: 700;
        color: #f0f0f0;
        margin: 0;
        line-height: 1.3;
    }
    .header-text p {
        font-size: 13.5px;
        color: #8b949e;
        margin: 2px 0 0 0;
    }

    hr.divider {
        border: none;
        border-top: 1px solid #2a3140;
        margin: 18px 0 20px 0;
    }

    /* Field labels */
    .field-label {
        font-size: 12.5px;
        font-weight: 700;
        letter-spacing: 0.06em;
        color: #e3a33e;
        margin-bottom: 6px;
        text-transform: uppercase;
    }
    .field-hint {
        font-size: 12.5px;
        color: #8b949e;
        margin-top: 4px;
        margin-bottom: 18px;
    }

    /* Number input styling */
    div[data-testid="stNumberInput"] input {
        background-color: #0d1117 !important;
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
        color: #f0f0f0 !important;
        font-size: 20px !important;
        padding: 10px 14px !important;
    }
    div[data-testid="stNumberInput"] button {
        display: none;
    }

    /* Predict button */
    div.stButton > button {
        width: 100%;
        background-color: #21262d;
        color: #e3a33e;
        font-weight: 700;
        font-size: 15.5px;
        border: 1px solid #3a4150;
        border-radius: 12px;
        padding: 12px 0;
        margin-top: 6px;
        transition: 0.2s;
    }
    div.stButton > button:hover {
        background-color: #2a3140;
        border-color: #e3a33e;
        color: #e3a33e;
    }

    /* Result box */
    .result-box {
        background-color: #10221e;
        border: 1px solid #1f4a3f;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        margin-top: 18px;
    }
    .result-label {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: #4fd1a5;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .result-value {
        font-size: 32px;
        font-weight: 800;
        color: #f5f5f5;
        font-family: Georgia, serif;
    }

    .footer-note {
        text-align: center;
        font-size: 12px;
        color: #6e7681;
        margin-top: 18px;
        line-height: 1.5;
    }

    /* Hide Streamlit default chrome */
    #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------- Card Header ----------------
st.markdown("""
<div class="card">
    <div class="header-row">
        <div class="icon-circle">🎯</div>
        <div class="header-text">
            <h1>LPA Predictor</h1>
            <p>Estimate expected package from IQ &amp; CGPA</p>
        </div>
    </div>
    <hr class="divider">
""", unsafe_allow_html=True)

# ---------------- IQ Field ----------------
st.markdown('<div class="field-label">IQ Score</div>', unsafe_allow_html=True)
iq = st.number_input("IQ", min_value=50, max_value=200, value=120, step=1, label_visibility="collapsed")
st.markdown('<div class="field-hint">Typical range: 80–160</div>', unsafe_allow_html=True)

# ---------------- CGPA Field ----------------
st.markdown('<div class="field-label">CGPA</div>', unsafe_allow_html=True)
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5, step=0.01, label_visibility="collapsed")
st.markdown('<div class="field-hint">On a 10-point scale</div>', unsafe_allow_html=True)

# ---------------- Predict Button ----------------
predict_clicked = st.button("Predict Package")

# ---------------- Result ----------------
if predict_clicked:
    input_data = np.array([[iq, cgpa]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)
    lpa_value = float(prediction[0][0]) if prediction.ndim > 1 else float(prediction[0])

    st.markdown(f"""
    <div class="result-box">
        <div class="result-label">Predicted Package</div>
        <div class="result-value">{lpa_value:.2f} LPA</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- Footer + Card Close ----------------
st.markdown("""
    <div class="footer-note">Prediction is an estimate from a trained regression model and is not a guarantee.</div>
</div>
""", unsafe_allow_html=True)