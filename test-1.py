import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
import re

# Download nltk data
nltk.download('stopwords')
from nltk.corpus import stopwords

# Set up Streamlit app
st.title("Text Classifier Using Gemini GenAI Embeddings")

# Input for Gemini API key
gemini_api_key = st.text_input("Enter your Gemini API Key", type="password")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV or XLSX file", type=["csv", "xlsx"])

if uploaded_file is not None and gemini_api_key:
    # Read the file
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # Check if data has at least two columns
    if len(data.columns) < 2:
        st.error("The data file must have at least two columns: 'Text' and 'Label'")
        st.stop()
    
    # Assuming first column is 'Text' and second column is 'Label'
    data.columns = ['Text', 'Label']

    # Preprocessing function
    def preprocess_text(text):
        # Lowercase
        text = text.lower()
        # Remove punctuation and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        text_tokens = text.split()
        filtered_text = [word for word in text_tokens if word not in stop_words]
        return ' '.join(filtered_text)

    # Preprocess the text data
    data['Processed_Text'] = data['Text'].apply(preprocess_text)

    # Generate embeddings using Gemini API
    def get_embedding(text, api_key):
        url = 'https://api.gemini.com/embeddings'
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        payload = {
            'text': text
        }
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            embedding = response.json().get('embedding')
            return embedding
        else:
            st.error(f"Error from Gemini API: {response.text}")
            return None

    # Generate embeddings
    st.info("Generating embeddings, please wait...")
    embeddings = []
    for idx, row in data.iterrows():
        embedding = get_embedding(row['Processed_Text'], gemini_api_key)
        if embedding is not None:
            embeddings.append(embedding)
        else:
            st.error("Failed to get embedding.")
            st.stop()

    # Convert embeddings to numpy array
    X = np.array(embeddings)
    y = data['Label']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build a simple classification model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict and compute accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.success(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Show the assigned categories
    st.write("Assigned Categories:")
    result_df = pd.DataFrame({'Text': data['Text'], 'Assigned_Category': model.predict(X)})
    st.dataframe(result_df)
else:
    st.warning("Please upload a data file and enter your Gemini API Key.")
