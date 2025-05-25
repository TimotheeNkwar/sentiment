import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Text cleaning function
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)     # Remove mentions
    text = re.sub(r'#\w+', '', text)     # Remove hashtags
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove punctuation
    return text.lower().strip()

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("Tweets.csv")
    df['clean_text'] = df['text'].apply(clean_text)
    return df

df = load_data()

# Train model
@st.cache_resource
def train_model():
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['airline_sentiment']
    
    model = LogisticRegression()
    model.fit(X, y)
    
    return model, vectorizer

model, vectorizer = train_model()

# Streamlit UI
st.title("✈️ Sentiment Analysis: Airline Tweets")
st.write("Enter a tweet related to an airline to predict its sentiment.")

# User input
user_input = st.text_area("Enter a tweet here:", "")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vec_input = vectorizer.transform([cleaned])
        prediction = model.predict(vec_input)[0]
        st.success(f"Predicted Sentiment: **{prediction.upper()}**")

