📊 Real-Time Sentiment Analysis for Detecting Emotional Trends
This project implements a real-time sentiment analysis tool that monitors and analyzes emotional trends on social media platforms, focusing on mental health-related communities such as r/depression and r/mentalhealth. It utilizes a multi-model approach, combining machine learning, deep learning, and natural language processing (NLP) techniques to process both text and image data from Reddit.

🔍 Features
Real-time data collection from Reddit using the Pushshift API.

Text sentiment classification using VADER and ML models (SVM, Random Forest, Naïve Bayes, KNN) with TF-IDF.

Image sentiment classification using a CNN model (MobileNetV2).

Multimodal sentiment analysis integrating both text and image inputs.

Time-series data storage in InfluxDB.

Interactive visualization using Streamlit and Power BI dashboards.

🧠 Tech Stack
Languages: Python

ML/DL: Scikit-learn, TensorFlow

Database: InfluxDB

Visualization: Streamlit, Power BI

APIs: Pushshift API (Reddit)

🎯 Objective
To detect emotional trends in real time from social media data, aiming to support mental health monitoring and research.

