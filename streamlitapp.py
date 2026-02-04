import streamlit as st
import pickle
import re
import time

st.set_page_config(
    page_title="Flipkart Sentiment Analyzer",
    page_icon="🛍️",
    layout="centered",
    initial_sidebar_state="expanded"
)

def local_css():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #f1f2f6, #e1eec3);
    }
    
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        color: #2874f0;
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        padding-bottom: 20px;
        text-shadow: 2px 2px 4px #cccccc;
    }
    
    .sub-text {
        text-align: center;
        color: #555;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }

    .stTextArea textarea {
        background-color: #ffffff;
        border: 2px solid #2874f0;
        border-radius: 10px;
        color: #333;
    }

    .stButton>button {
        background-color: #fb641b;
        color: white;
        border: none;
        border-radius: 25px;
        padding: 15px 30px;
        font-size: 18px;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        background-color: #d64b08;
        transform: scale(1.02);
        box-shadow: 0 6px 8px rgba(0,0,0,0.2);
    }
    
    .result-card {
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shopping-bag.png", width=80)
    st.title("About the App")
    st.info(
        """
        This app uses Machine Learning to analyze customer reviews from Flipkart.
        
        **How it works:**
        1. Enter the review text.
        2. Our model cleans and vectorizes the text.
        3. It predicts if the sentiment is **Positive** or **Negative**.
        """
    )
    st.markdown("---")
    st.write("Created with ❤️ using Streamlit")

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
    st.error("Model files not found! Please ensure sentiment_model.pkl and vectorizer.pkl are in the directory.")
    st.stop()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("READ MORE", "")
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

st.markdown('<div class="main-header">Flipkart Review Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Decode customer sentiment instantly with AI</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 8])
with col2:
    user_input = st.text_area("Paste the review below:", height=150, placeholder="E.g., The battery life is amazing, but the camera is average...")

_, mid_col, _ = st.columns([1, 2, 1])
with mid_col:
    analyze_btn = st.button("✨ Analyze Sentiment")

if analyze_btn:
    if user_input:
        with st.spinner('Analyzing patterns...'):
            time.sleep(0.8) 
            
            cleaned_input = clean_text(user_input)
            vectorized_input = vectorizer.transform([cleaned_input])
            
            prediction = model.predict(vectorized_input)[0]
            probability = model.predict_proba(vectorized_input)[0]
            
            neg_prob = probability[0]
            pos_prob = probability[1]

        st.markdown("---")
        
        if prediction == 1:
            st.balloons()
            st.markdown(
                f"""
                <div class="result-card" style="background-color: #d4edda; color: #155724; border: 2px solid #c3e6cb;">
                    <h2>😊 Positive Feedback</h2>
                    <p>The model is <b>{pos_prob*100:.2f}%</b> confident.</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="result-card" style="background-color: #f8d7da; color: #721c24; border: 2px solid #f5c6cb;">
                    <h2>😞 Negative Feedback</h2>
                    <p>The model is <b>{neg_prob*100:.2f}%</b> confident.</p>
                </div>
                """, 
                unsafe_allow_html=True
            )

        st.write("")
        m_col1, m_col2 = st.columns(2)
        
        with m_col1:
            st.metric(label="Positivity Score", value=f"{pos_prob:.2%}", delta="High Confidence" if pos_prob > 0.8 else None)
            st.progress(float(pos_prob))
            
        with m_col2:
            st.metric(label="Negativity Score", value=f"{neg_prob:.2%}", delta="-High Confidence" if neg_prob > 0.8 else None, delta_color="inverse")
            st.progress(float(neg_prob))

    else:
        st.warning("⚠️ Please enter some text to analyze.")
