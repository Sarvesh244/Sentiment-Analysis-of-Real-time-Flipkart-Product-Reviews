import streamlit as st
import pickle
import re

st.set_page_config(
    page_title="Flipkart Sentiment Analyzer",
    page_icon="🛒",
    layout="centered"
)

st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #2874f0;
        font-family: sans-serif;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff9f00;
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background-color: #f09200;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    with open('sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

try:
    model, vectorizer = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found. Please run 'train_model.py' first.")
    st.stop()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("READ MORE", "")
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

with st.sidebar:
    st.image("https://www.vhv.rs/dpng/d/147-1473726_ganpati-icon-png-flipkart-logo-transparent-png.png", width=80)
    st.title("About")
    st.write("This tool uses AI to analyze customer reviews and determine if they are positive or negative.")
    st.markdown("---")
    st.caption("Built with Streamlit")

st.markdown("<h1 class='main-title'>Flipkart Review Analyzer</h1>", unsafe_allow_html=True)
st.write("Paste a product review below to detect the sentiment.")

user_input = st.text_area("Review Text", height=150, placeholder="Type here...")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze = st.button("🔍 Analyze Sentiment")

if analyze:
    if user_input:
        cleaned_input = clean_text(user_input)
        vectorized_input = vectorizer.transform([cleaned_input])
        
        prediction = model.predict(vectorized_input)[0]
        probability = model.predict_proba(vectorized_input)[0]
        
        st.divider()
        
        if prediction == 1:
            st.success("### Sentiment: Positive 😃")
            st.metric("Confidence Score", f"{probability[1]*100:.2f}%")
            st.progress(float(probability[1]))
        else:
            st.error("### Sentiment: Negative 😞")
            st.metric("Confidence Score", f"{probability[0]*100:.2f}%")
            st.progress(float(probability[0]))
            
    else:
        st.warning("Please enter some text to analyze.")
