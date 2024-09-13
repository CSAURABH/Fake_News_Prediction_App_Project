import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the trained model with any necessary custom objects
try:
    model = load_model('fake_news_detection.h5')  # Add custom_objects parameter if necessary
except Exception as e:
    st.write(f"Error loading model: {e}")

# Initialize tokenizer
max_words = 10000
max_len = 100
tokenizer = Tokenizer(num_words=max_words)

def preprocess_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=max_len)
    return padded

def main():
    st.title("Fake News Detection App")

    user_input = st.text_area("Enter news text", "Type here...")
    
    if st.button("Predict"):
        processed_text = preprocess_text(user_input)
        prediction = model.predict(processed_text)

        if prediction >= 0.5:
            st.write("The news is likely **REAL**")
        else:
            st.write("The news is likely **FAKE**")

if __name__ == "__main__":
    main()
