import streamlit as st
import pickle

# Load trained model and vectorizer
model = pickle.load(open("model/spam_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# Page configuration
st.set_page_config(
    page_title="AI Spam Detector",
    page_icon="📧",
    layout="centered"
)

# Custom CSS styling
st.markdown("""
<style>

/* 3-Color Classy Gradient Background */
.stApp {
    background: linear-gradient(135deg, #a770ef ,#cf8bf3 ,#fdb99b);
    background-attachment: fixed;
}

/* Title Styling */
h1 {
    text-align: center;
    color: white;
    font-size: 45px;
    font-weight: bold;
}

/* Text Area Styling */
textarea {
    border-radius: 12px !important;
    border: 2px solid white !important;
    padding: 10px !important;
}

/* Button Styling */
.stButton>button {
    background: linear-gradient(90deg, #c33764 ,#1d2671);
    color: white;
    font-size: 18px;
    border-radius: 10px;
    padding: 10px 25px;
    border: none;
}

/* Button Hover Effect */
.stButton>button:hover {
    background: linear-gradient(135deg, #03001e ,#7303c0 ,#ec38bc );
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