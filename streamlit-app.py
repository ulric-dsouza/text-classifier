import streamlit as st
import pandas as pd
from gemini_client import GeminiClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

def preprocess_text(text):

    # Remove extraneous words and perform other preprocessing steps
    # Replace with your specific preprocessing logic
    return text

def generate_embeddings(texts):
    # Initialize Gemini client
    client = GeminiClient()

    # Generate embeddings
    embeddings = []
    for text in texts:
        embedding = client.generate_embeddings(text)
        embeddings.append(embedding)

    return embeddings

def build_classifier(X_train, y_train):
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Train Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(X_train_tfidf, y_train)

    return classifier, vectorizer

def classify_text(classifier, vectorizer, text):
    # Preprocess text
    text = preprocess_text(text)

    # Generate embedding
    embedding = generate_embeddings([text])

    # Transform embedding using TF-IDF vectorizer
    text_tfidf = vectorizer.transform(embedding)

    # Predict category
    predicted_category = classifier.predict(text_tfidf)[0]

    return predicted_category

def main():
    st.title("Text Classifier")

    # Upload data file
    uploaded_file = st.file_uploader("Upload CSV or XLSX file")

    if uploaded_file:
        # Read data from file
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(uploaded_file)

        else:
            st.error("Invalid   
 file format. Please upload a CSV or XLSX file.")
            return

        # Extract text data
        text_data = df["text_column"]  # Replace "text_column" with your actual column name

        # Preprocess text data
        preprocessed_text_data = text_data.apply(preprocess_text)

        # Generate embeddings
        embeddings = generate_embeddings(preprocessed_text_data)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(embeddings, text_data, test_size=0.2, random_state=42)

        # Build classifier
        classifier, vectorizer = build_classifier(X_train, y_train)

        # Evaluate classifier on test set
        y_pred = classifier.predict(vectorizer.transform(X_test))
        accuracy = accuracy_score(y_test, y_pred)
        st.write("Accuracy:", accuracy)

        # Classify new text
        new_text = st.text_input("Enter a new text to classify:")
        if new_text:
            predicted_category = classify_text(classifier, vectorizer, new_text)
            st.write("Predicted category:", predicted_category)

if __name__ == "__main__":
    main()
