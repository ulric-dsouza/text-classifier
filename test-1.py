import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import requests

# Set your Gemini API key
api_key = "AIzaSyC8FSRI4MaX6hOrnwVDNNK0wcLHP_048-g"

import re

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])

    # Remove punctuation and other non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    return text

def get_gemini_embeddings(text):
    url = "https://api.gemini.com/embeddings"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"text": text}
    response = requests.post(url, headers=headers, json=data)
    embeddings = response.json()["embeddings"]
    return embeddings

def train_classifier(X_train, y_train):
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)

    # Train a Multinomial Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    return classifier, vectorizer

def classify_text(classifier, vectorizer, text):
    text = preprocess_text(text)
    embeddings = get_gemini_embeddings(text)
    X_test = vectorizer.transform([text])
    predicted_category = classifier.predict(X_test)[0]
    return predicted_category

def main():
    st.title("Text Classifier using Gemini GenAI Embeddings")

    # Upload data file
    uploaded_file = st.file_uploader("Upload a CSV or XLSX file")

    if uploaded_file is not None:
        # Read data from the file
        if uploaded_file.type == "text/csv":
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            data = pd.read_excel(uploaded_file)
        else:
            st.error("Please upload a CSV or XLSX file.")

        # Preprocess the text data
        data['text'] = data['text'].apply(preprocess_text)

        # Get Gemini embeddings
        data['embeddings'] = data['text'].apply(get_gemini_embeddings)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(data['embeddings'], data['category'], test_size=0.2, random_state=42)

        # Train the classifier
        classifier, vectorizer = train_classifier(X_train, y_train)

        # Evaluate the classifier
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write("Accuracy:", accuracy)

        # Classify new text
        new_text = st.text_input("Enter a new text to classify:")
        if new_text:
            predicted_category = classify_text(classifier, vectorizer, new_text)
            st.write("Predicted Category:", predicted_category)

if __name__ == '__main__':
    main()
