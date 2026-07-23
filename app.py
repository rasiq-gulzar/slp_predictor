import streamlit as st
import pickle
import numpy as np

# Model aur scaler load karo
with open('slp_regressor.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="LPA Predictor", page_icon="💼")
st.title("🎓 CGPA & IQ se LPA Predictor")
st.write("Apna CGPA aur IQ daalo, model predict karega expected LPA (package)")
iq = st.number_input("IQ", min_value=50, max_value=200, value=100, step=1)
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5, step=0.01)


if st.button("Predict LPA"):
    input_data = np.array([[iq, cgpa]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)
    
    lpa_value = float(prediction[0][0]) if prediction.ndim > 1 else float(prediction[0])
    st.success(f"Predicted LPA: ₹{lpa_value:.2f} LPA")