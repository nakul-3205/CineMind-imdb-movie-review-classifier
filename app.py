# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model
model = load_model('model.h5')

# Step 2: Helper Functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

MAX_WORDS = 10000  # embedding layer vocab size

def preprocess_text(text):
    words = text.lower().split()
    # Ensure indices do not exceed embedding vocab
    encoded_review = [min(word_index.get(word, 2) + 3, MAX_WORDS - 1) for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# --- Streamlit UI/UX ---
st.set_page_config(page_title="Movie Sentiment Analyzer", page_icon="🎬", layout="centered")
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>🎬 IMDB Movie Review Sentiment Analysis</h1>
    <p style='text-align: center; color: #555;'>Enter a movie review below and see if it's positive or negative!</p>
    """, unsafe_allow_html=True
)

# Styled input box
user_input = st.text_area(
    'Your Movie Review:', 
    placeholder="Type your review here...", 
    height=150
)

# Center the button
if st.button('Classify Review', use_container_width=True):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review before clicking 'Classify'.")
    else:
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)
        sentiment = 'Positive 😄' if prediction[0][0] > 0.5 else 'Negative 😞'
        score = prediction[0][0]

        # Display results in a colorful container
        if sentiment.startswith("Positive"):
            st.success(f"Sentiment: {sentiment}")
        else:
            st.error(f"Sentiment: {sentiment}")
        
        # Show probability as a progress bar
        st.write("Prediction Confidence:")
        st.progress(float(score) if score > 0.5 else 1 - float(score))
else:
    st.info('Enter a movie review above and click "Classify Review" to get the sentiment!')
