import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer

# ----------------------------------
# UI Styling (YouTube Light Mode)
# ----------------------------------
def apply_light_theme():
    st.markdown("""
    <style>
    .stApp {
        background-color: #f9f9f9;
        color: #0f0f0f;
    }
    div.block-container {
        background-color: #ffffff;
        padding: 2.5rem 3.5rem;
        border-radius: 16px;
        border: 1px solid #e5e5e5;
        margin-top: 40px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    }
    .yt-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 25px;
    }
    .stTextArea textarea {
        background-color: #fcfcfc !important;
        color: #0f0f0f !important;
        border: 1px solid #cccccc !important;
        border-radius: 8px !important;
        font-size: 16px;
    }
    div.stButton > button {
        background-color: #FF0000;
        color: white;
        padding: 14px 24px;
        border-radius: 30px;
        font-weight: 700;
        width: 100%;
    }
    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-top: 25px;
        background: #f1f1f1;
        border: 1px solid #e5e5e5;
    }
    </style>
    """, unsafe_allow_html=True)

apply_light_theme()

# ----------------------------------
# NLP + Model Loader
# ----------------------------------
@st.cache_resource
def load_resources():
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("punkt")
    nltk.download("punkt_tab")   # IMPORTANT FOR STREAMLIT CLOUD

    with open("log_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer

model, vectorizer = load_resources()

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ----------------------------------
# Text Preprocessing (SAFE)
# ----------------------------------
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)

    tokens = wordpunct_tokenize(text)   # SAFE TOKENIZER

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(tokens)

# ----------------------------------
# UI
# ----------------------------------
st.markdown("""
<div class="yt-header">
    <img src="https://upload.wikimedia.org/wikipedia/commons/e/ef/Youtube_logo.png" width="45">
    <h2 style="margin:0;font-weight:800;">
        Sentiment <span style="color:#FF0000;">AI</span>
    </h2>
</div>
""", unsafe_allow_html=True)

st.write("Analyze YouTube comment sentiment using NLP.")

comment = st.text_area("", placeholder="Enter comment for analysis...", height=130)

if st.button("RUN ANALYSIS"):
    if not comment.strip():
        st.warning("⚠️ Please enter a comment")
    else:
        with st.spinner("Analyzing..."):
            processed = preprocess_text(comment)
            vectorized = vectorizer.transform([processed])

            proba = model.predict_proba(vectorized)[0]
            prediction = model.predict(vectorized)[0]
            confidence = max(proba) * 100

        label = "POSITIVE" if prediction == 1 else "NEGATIVE"
        color = "#008000" if prediction == 1 else "#CC0000"
        icon = "📈" if prediction == 1 else "📉"

        st.markdown(f"""
        <div class="result-box">
            <h2 style="color:{color};">{icon} {label}</h2>
            <p><b>Confidence:</b> {confidence:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(confidence))

# ----------------------------------
# Footer
# ----------------------------------
st.markdown("""
<div style="margin-top:40px;text-align:center;color:#999;font-size:12px;">
    DEVELOPED BY <b>BAGADI <span style="color:#FF0000;">SANTHOSH KUMAR</span></b>
</div>
""", unsafe_allow_html=True)
