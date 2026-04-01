# 📌 Step 1: Import Libraries
import pandas as pd
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords (only first time)
nltk.download('stopwords')

# 📌 Step 2: Load Dataset (FIXED)
data = pd.read_csv("spam.csv", sep='\t', names=['label', 'message'])

# Convert labels: ham=0, spam=1
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# 📌 Step 3: Text Preprocessing
ps = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    
    return " ".join(words)

data['message'] = data['message'].apply(clean_text)

# 📌 Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

# 📌 Step 5: Feature Extraction (TF-IDF)
tfidf = TfidfVectorizer(max_features=3000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 📌 Step 6: Model Training
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 📌 Step 7: Prediction
y_pred = model.predict(X_test_tfidf)

# 📌 Step 8: Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 📌 Step 9: Custom Prediction Function
def predict_spam(text):
    text = clean_text(text)
    vector = tfidf.transform([text])
    result = model.predict(vector)[0]
    
    if result == 1:
        return "Spam"
    else:
        return "Not Spam"

# 📌 Step 10: Test Example
print("\nTest Message:", predict_spam("Congratulations! You won a free ticket"))# 📌 Step 1: Import Libraries
