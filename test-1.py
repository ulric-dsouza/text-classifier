import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import requests

# Set API key
GEMINI_API_KEY = 'AIzaSyC8FSRI4MaX6hOrnwVDNNK0wcLHP_048-g'

# Function to generate embeddings using Gemini API
def get_gemini_embeddings(text):
    headers = {
        'Authorization': f'Bearer {GEMINI_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    # Replace this with the correct endpoint from Gemini API documentation
    url = "https://api.gemini.com/v1/embeddings"
    
    # Prepare payload
    data = {"text": text}
    
    # Make API request
    response = requests.post(url, headers=headers, json=data)
    
    # Log full response for debugging
    if response.status_code == 200:
        embedding = response.json().get("embedding")
        return embedding
    else:
        st.error(f"Failed to get embeddings from Gemini API. Status code: {response.status_code}")
        st.write("Response content:", response.text)  # To see the full error message
        return None

# Function to clean text data (remove extraneous words)
def preprocess_text(text):
    # Here we can add functions like removing stop words, punctuation, etc.
    return text.lower()

# Streamlit App Layout
st.title("Text Classifier Using Gemini GenAI Embeddings")

# Step 1: File upload
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Step 2: Read the file
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Display uploaded data
    st.write("Uploaded Data:", df.head())
    
    # Check for the required column
    if len(df.columns) != 1:
        st.error("The uploaded file should contain only one column.")
    else:
        column_name = df.columns[0]
        
        # Step 3: Preprocess text data
        st.write("Preprocessing text data...")
        df[column_name] = df[column_name].apply(preprocess_text)
        
        # Step 4: Generate embeddings using Gemini API
        st.write("Generating embeddings...")
        embeddings = []
        for text in df[column_name]:
            embedding = get_gemini_embeddings(text)
            if embedding:
                embeddings.append(embedding)
        
        if len(embeddings) == len(df):
            # Step 5: Create a dummy classification label for demo purposes (can be modified)
            df['category'] = df.index % 2  # Assigning labels for example purposes

            # Encoding labels
            le = LabelEncoder()
            df['category'] = le.fit_transform(df['category'])

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(embeddings, df['category'], test_size=0.2, random_state=42)

            # Step 6: Build a simple classification model (Logistic Regression)
            clf = LogisticRegression()
            clf.fit(X_train, y_train)
            
            # Predict on the test set
            y_pred = clf.predict(X_test)

            # Step 7: Show accuracy and predicted categories
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
            
            # Assign categories for new inputs
            st.write("Predicted categories for the test set:")
            st.write(pd.DataFrame({'Text': df[column_name].iloc[X_test.index], 'Predicted Category': y_pred}))

        else:
            st.error("Embedding generation failed for some texts.")
