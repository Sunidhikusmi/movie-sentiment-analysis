# predict_sentiment.py

import re
from joblib import load
import nltk
from nltk.corpus import stopwords

# ----------------- STOPWORDS -----------------
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# ----------------- LOAD MODEL & TF-IDF -----------------
tfidf = load('tfidf_vectorizer.pkl')   # Change extension to .joblib if renamed
model = load('logreg_model.pkl')       # Change extension to .joblib if renamed

print("TF-IDF vectorizer and model loaded successfully!")

# ----------------- TEXT CLEANING -----------------
def clean_text(text):
    """
    Clean the input text:
    - Remove non-alphabetic characters
    - Convert to lowercase
    - Remove stopwords
    """
    text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    return ' '.join([word for word in text if word not in stop_words and len(word) > 1])

# ----------------- PREDICTION FUNCTION -----------------
def predict_sentiment(text):
    cleaned = clean_text(text)
    vectorized = tfidf.transform([cleaned])
    pred = model.predict(vectorized)[0]
    return 'Positive' if pred == 1 else 'Negative'

# ----------------- INTERACTIVE PREDICTION -----------------
def main():
    print("\n🎬 Movie Sentiment Predictor")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_input = input("Enter a movie review:\n").strip()

        if user_input.lower() == 'exit':
            print("Exiting predictor.")
            break

        if not re.search(r"[a-zA-Z]", user_input):
            print("Please enter a meaningful review.")
            continue

        result = predict_sentiment(user_input)
        print("Predicted Sentiment:", result, "\n")

# ----------------- RUN SCRIPT -----------------
if __name__ == "__main__":
    main()
