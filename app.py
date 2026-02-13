import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import re

# 1. Page Configuration
st.set_page_config(
    page_title="BrandPulse | Sentiment AI",
    page_icon="💬",
    layout="centered"
)

# 2. Exotic UI CSS
st.markdown("""
    <style>
    .main { background-color: #f4f6f9; }
    .stTextArea textarea { font-size: 16px; border-radius: 10px; border: 2px solid #4a90e2; }
    .stButton>button { width: 100%; background-color: #4a90e2; color: white; font-weight: bold; border-radius: 8px; padding: 12px; }
    .stButton>button:hover { background-color: #357abd; border: 2px solid #4a90e2; }
    </style>
    """, unsafe_allow_html=True)

# 3. Load Model and Word Dictionary
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('sentiment_rnn.h5')
    # We need the exact dictionary the AI was trained on to translate new text into numbers
    word_index = imdb.get_word_index()
    return model, word_index

try:
    model, word_index = load_assets()
except Exception as e:
    st.error("⚠️ Model missing. Ensure 'sentiment_rnn.h5' is in the repository.")
    st.stop()

# Text Preprocessing Function
def encode_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # Remove punctuation
    words = text.split()
    
    tokens = []
    for word in words:
        # Keras IMDB index is offset by 3
        idx = word_index.get(word, -3) + 3 
        if idx >= 10000: # Max words used in training
            idx = 2 # Treat as Out-Of-Vocabulary
        tokens.append(idx)
        
    padded = sequence.pad_sequences([tokens], maxlen=200)
    return padded

# 4. Main UI
st.title("💬 BrandPulse Sentiment AI")
st.markdown("### Powered by Recurrent Neural Networks (LSTM)")
st.write("Paste a customer review, tweet, or feedback below. The AI will read the context and determine the emotional sentiment.")

user_input = st.text_area("Customer Feedback Input:", height=150, placeholder="e.g., I absolutely loved this product, the quality is outstanding and delivery was fast!")

if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("AI is reading the sequence of words..."):
            # Process and Predict
            encoded_input = encode_text(user_input)
            prediction = model.predict(encoded_input)
            sentiment_score = float(prediction[0][0])
            
            st.markdown("---")
            st.markdown("### 📊 AI Analysis Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if sentiment_score >= 0.5:
                    st.success("🟢 POSITIVE SENTIMENT")
                    st.metric(label="Confidence Score", value=f"{sentiment_score:.1%}")
                else:
                    st.error("🔴 NEGATIVE SENTIMENT")
                    st.metric(label="Confidence Score", value=f"{(1 - sentiment_score):.1%}")
                    
            with col2:
                st.markdown("**Managerial Routing Action:**")
                if sentiment_score >= 0.5:
                    st.info("✅ **Action:** Route to Marketing team for potential use as a website testimonial.")
                else:
                    st.warning("⚠️ **Action:** High priority flag. Route immediately to Customer Success team for intervention.")
                    
            st.progress(sentiment_score)
            st.caption("0 = Extremely Negative | 1 = Extremely Positive")