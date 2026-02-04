import streamlit as st
import pickle
import re

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

st.title("Flipkart Review Sentiment Analyzer")
st.write("Enter a product review to determine if the sentiment is Positive or Negative.")

user_input = st.text_area("Review Text", "The product quality is amazing!")

if st.button("Analyze Sentiment"):
    if user_input:
        cleaned_input = clean_text(user_input)
        
        vectorized_input = vectorizer.transform([cleaned_input])
        
        prediction = model.predict(vectorized_input)[0]
        probability = model.predict_proba(vectorized_input)[0]
        
        if prediction == 1:
            st.success(f"Sentiment: **Positive** (Confidence: {probability[1]:.2f})")
        else:
            st.error(f"Sentiment: **Negative** (Confidence: {probability[0]:.2f})")
    else:
        st.warning("Please enter some text.")
