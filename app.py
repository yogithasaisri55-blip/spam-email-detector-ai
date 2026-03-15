import streamlit as st
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="AI Spam Detector",
    page_icon="📧",
    layout="centered"
)

# Load model safely
MODEL_PATH = "model/spam_model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    st.error("❌ Model files not found. Please check the 'model' folder in the repository.")
    st.stop()

model = pickle.load(open(MODEL_PATH, "rb"))
vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))

# Custom CSS styling
st.markdown("""
<style>

/* 3-Color Classy Gradient Background */
.stApp {
    background: linear-gradient(135deg,#a770ef,#cf8bf3,#fdb99b);
    background-attachment: fixed;
}

/* Title */
h1 {
    text-align:center;
    color:white;
    font-size:45px;
    font-weight:bold;
}

/* Textbox */
textarea {
    border-radius:12px !important;
    border:2px solid white !important;
    padding:10px !important;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg,#c33764,#1d2671);
    color:white;
    font-size:18px;
    border-radius:10px;
    padding:10px 25px;
    border:none;
}

.stButton>button:hover {
    background: linear-gradient(135deg,#03001e,#7303c0,#ec38bc);
}

</style>
""", unsafe_allow_html=True)

# Title
st.title("📧 AI Spam Email Detector")

# User input
message = st.text_area("Enter your message here:")

# Button
if st.button("Check Message"):

    if message.strip() != "":
        data = vectorizer.transform([message])
        prediction = model.predict(data)

        if prediction[0] == 1:
            st.error("🚨 This is a Spam Message")
        else:
            st.success("✅ This is Not Spam")
    else:
        st.warning("⚠ Please enter a message first!")