import streamlit as st
import joblib
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import io

# Set Streamlit page configuration
st.set_page_config(page_title="Sentiment Analysis Tool", layout="centered")

# Load Models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
vectorizer = joblib.load(os.path.join(BASE_DIR, "D:\\final year project\\models\\tfidf_vectorizer.pkl"))
text_model = joblib.load(os.path.join(BASE_DIR, "D:\\final year project\\models\\svm_model.pkl"))
image_model = tf.keras.models.load_model(os.path.join(BASE_DIR, "D:\\final year project\\optimized_sentiment_cnn_model.keras"))

# Function to process image
def preprocess_image(img):
    img = img.resize((128, 128))  # Resize to match CNN input size
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Map numerical sentiment to text
def get_sentiment_text(sentiment_value):
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_map.get(sentiment_value, "Unknown")

# Streamlit UI
st.title("📊 Sentiment Analysis Tool")
st.write("Analyze sentiment from **text** or **images**")

# --- TEXT SENTIMENT ANALYSIS ---
st.subheader("🔤 Analyze Text Sentiment")
text_input = st.text_area("Enter text:", placeholder="Type or paste text here...")

if st.button("Analyze Text"):
    if text_input.strip():
        text_features = vectorizer.transform([text_input])
        prediction = text_model.predict(text_features)[0]
        sentiment_text = get_sentiment_text(prediction)
        
        # Display sentiment result
        st.success(f"**Sentiment:** {sentiment_text}")
    else:
        st.warning("⚠️ Please enter some text for analysis.")

# --- IMAGE SENTIMENT ANALYSIS ---
st.subheader("🖼️ Analyze Image Sentiment")
uploaded_image = st.file_uploader("Upload an image:", type=["jpg", "png", "jpeg"])

if uploaded_image:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Analyze Image"):
        # Convert the uploaded image into PIL format
        img = Image.open(io.BytesIO(uploaded_image.read()))
        img_array = preprocess_image(img)
        
        prediction = image_model.predict(img_array)
        sentiment_label = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction[0]))  # Get the confidence score
        sentiment_text = get_sentiment_text(sentiment_label)
        
        # Display sentiment result
        st.success(f"**Sentiment:** {sentiment_text} (Confidence: {confidence:.2f})")

st.write("🚀 **Built using Streamlit & Machine Learning models**")

