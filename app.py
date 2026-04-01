import streamlit as st
import pandas as pd
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

st.title("📧 Spam Email Detection using NLP")

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

@st.cache_resource
def load_model():
    data = pd.read_csv("spam.csv", sep='\t', names=['label', 'message'])
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})

    def clean_text(text):
        text = text.lower()
        text = ''.join([char for char in text if char not in string.punctuation])
        words = text.split()
        words = [ps.stem(word) for word in words if word not in stop_words]
        return " ".join(words)

    data['message'] = data['message'].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        data['message'], data['label'], test_size=0.2, random_state=42
    )

    tfidf = TfidfVectorizer(max_features=3000)
    X_train_tfidf = tfidf.fit_transform(X_train)

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    return model, tfidf, clean_text

model, tfidf, clean_text = load_model()

st.subheader("Enter a message:")
user_input = st.text_area("Message")

if st.button("Check Spam"):
    if user_input.strip():
        cleaned = clean_text(user_input)
        vector = tfidf.transform([cleaned])
        result = model.predict(vector)[0]

        if result == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")
    else:
        st.warning("⚠️ Enter a message")
