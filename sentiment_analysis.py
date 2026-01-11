# Standard library imports
import pandas as pd
import numpy as np
import os
import re   
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load

# Machine Learning imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Download stopwords 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Check current working directory and dataset files
print("Current Working Directory :", os.getcwd())  # e.g., project root
print("Files in dataset:", os.listdir("dataset"))  # list expected CSVs

# Load datasets into pandas DataFrames for later preprocessing/modeling
train_df = pd.read_csv("dataset/train.csv", header=None, names=['review','sentiment'], skiprows=1)
test_df = pd.read_csv("dataset/test.csv", header=None, names=['review','sentiment'], skiprows=1)

# Inspect first few rows
print("\nTrain dataset sample:")
print(train_df.head())

# Normalize and validate columns
train_df['review'] = train_df['review'].fillna('').astype(str)
test_df['review'] = test_df['review'].fillna('').astype(str)

# Convert sentiment labels to numeric 0/1. 
train_df['sentiment'] = pd.to_numeric(train_df['sentiment'].map({'positive':1,'negative':0}).fillna(train_df['sentiment']), errors='coerce')
test_df['sentiment'] = pd.to_numeric(test_df['sentiment'].map({'positive':1,'negative':0}).fillna(test_df['sentiment']), errors='coerce')

# Drop rows with missing required fields
train_df = train_df.dropna(subset=['review','sentiment'])
test_df = test_df.dropna(subset=['review','sentiment'])

# Ensure integer type
train_df['sentiment'] = train_df['sentiment'].astype(int)
test_df['sentiment'] = test_df['sentiment'].astype(int)

# Function to clean text data (Text Preprocessing)
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    return ' '.join([word for word in text if word not in stop_words and len(word) > 1])

# Apply text cleaning to the review columns
train_df["review"] = train_df["review"].apply(clean_text)
test_df["review"] = test_df["review"].apply(clean_text) 

print("Sample cleaned review (first 200 chars):")
print(train_df['review'].iloc[0][:200])  # Print the first cleaned review (truncated) for verification
print("Fitting TF-IDF on training data...")
   
#Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000)  # Limit to top 5000 important words
X_train = tfidf.fit_transform(train_df["review"])   # Fit on training data & transform
X_test = tfidf.transform(test_df["review"]) # Transform test data

# Labels
y_train = train_df["sentiment"]
y_test = test_df["sentiment"]   

print("\nTF-IDF vectorization complete. Shape of training data:", X_train.shape)

# Train Logistic Regression Model
model = LogisticRegression(max_iter=1000, solver='liblinear')
print("\nTraining LogisticRegression...")
model.fit(X_train, y_train)
print("Model training complete")

# Save model and vectorizer
dump(model, "logreg_model.pkl")
dump(tfidf, "tfidf_vectorizer.pkl")
print("Model and TF-IDF vectorizer saved!")

# Evaluate
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

conf_mat = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", conf_mat)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Visualize confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig("confusion_matrix.png", bbox_inches='tight')
print("Confusion matrix saved to: confusion_matrix.png")

# End of sentiment_analysis.py