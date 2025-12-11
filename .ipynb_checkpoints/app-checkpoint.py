import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# ---------------------------------------------------------
# MUST BE THE FIRST STREAMLIT COMMAND
# ---------------------------------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# DOWNLOAD NLTK DATA
# ---------------------------------------------------------
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except:
        pass

download_nltk_data()

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.75rem;
        font-size: 1.1rem;
        border-radius: 10px;
        border: none;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .fake-result {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .real-result {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .title {
        text-align: center;
        color: #1f1f1f;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #000000;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin: 1.5rem 0;
    }
    .metric-box {
        text-align: center;
        padding: 1rem;
        background: #f5f5f5;
        border-radius: 8px;
        flex: 1;
        margin: 0 0.5rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1976d2;
    }
    .metric-label {
        color: #666;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# LOAD MODEL & VECTORIZER
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load('fake_news_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        return None, None

# ---------------------------------------------------------
# TEXT CLEANING FUNCTION
# ---------------------------------------------------------
@st.cache_data
def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())

    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return ' '.join(tokens)

# ---------------------------------------------------------
# PREDICTION FUNCTION
# ---------------------------------------------------------
def predict_news(text, model, vectorizer):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])

    prediction = model.predict(vectorized)[0]

    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(vectorized)[0]
        confidence = proba[prediction] * 100
    else:
        confidence = None

    result = "REAL" if prediction == 1 else "FAKE"
    return result, confidence

# ---------------------------------------------------------
# MAIN APP UI
# ---------------------------------------------------------
def main():

    st.markdown('<h1 class="title">📰 Fake News Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Powered by Machine Learning | Detect misinformation instantly</p>', unsafe_allow_html=True)

    model, vectorizer = load_model()

    if model is None or vectorizer is None:
        st.error("⚠️ Model files not found! Please ensure 'fake_news_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory.")
        st.stop()

    st.markdown("""
        <div class="info-box">
            <strong>ℹ️ How to use:</strong><br>
            1. Paste or type a news article title and content below<br>
            2. Click "Analyze News"<br>
            3. Get a REAL or FAKE prediction with confidence score
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📝 Enter News Article")

    title = st.text_input("News Title", placeholder="Enter the news headline here...")
    text = st.text_area("News Content", height=200, placeholder="Paste the full news article here...")

    # Sample examples
    with st.expander("📌 Try Sample News Articles"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Load Fake News Sample"):
                st.session_state['sample_title'] = "BREAKING: Scientists Discover Cure for All Diseases!"
                st.session_state['sample_text'] = (
                    "Researchers claim a miracle cure for all diseases. "
                    "The government is hiding the information. Share this quickly!"
                )
                st.rerun()
        with col2:
            if st.button("Load Real News Sample"):
                st.session_state['sample_title'] = "New Study Shows Climate Change Impact on Agriculture"
                st.session_state['sample_text'] = (
                    "A study published in Nature Climate Change reports significant agricultural impacts "
                    "due to rising global temperatures."
                )
                st.rerun()

    if 'sample_title' in st.session_state:
        title = st.session_state['sample_title']
        text = st.session_state['sample_text']
        del st.session_state['sample_title']
        del st.session_state['sample_text']

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 Analyze News"):
        if not title and not text:
            st.warning("⚠️ Please enter a title or content to analyze.")
        else:
            combined_text = f"{title} {text}"

            with st.spinner("🔄 Analyzing article with AI..."):
                result, confidence = predict_news(combined_text, model, vectorizer)

                st.markdown("### 📊 Analysis Results")

                if result == "FAKE":
                    st.markdown("""
                        <div class="fake-result">
                            <h2 style="color: #f44336;">🚫 FAKE NEWS DETECTED</h2>
                            <p>This article shows characteristics of misinformation.</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.warning("Verify this news from trusted sources.")
                else:
                    st.markdown("""
                        <div class="real-result">
                            <h2 style="color: #4caf50;">✅ LIKELY REAL NEWS</h2>
                            <p>This article appears to be legitimate.</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.success("This content seems reliable.")

                # Confidence metrics
                if confidence:
                    st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-box">
                                <div class="metric-value">{confidence:.1f}%</div>
                                <div class="metric-label">Confidence Score</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-value">{len(combined_text.split())}</div>
                                <div class="metric-label">Words Analyzed</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("""
                    <div style="text-align: center; color: #666;">
                        <strong>💡 Reminder:</strong> AI predictions are not perfect.
                        Always cross-check important information.
                    </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
