# Movie Sentiment Analysis

## Project Overview
This project performs **sentiment analysis** on IMDb movie reviews using **Natural Language Processing (NLP)** and **Machine Learning (ML)**.  
It classifies reviews as **Positive** or **Negative** based on the textual content.

**Workflow:**

1. **NLP Preprocessing**  
   - Text cleaning (removing punctuation, numbers, special characters)  
   - Lowercasing  
   - Stopword removal using NLTK  

2. **TF-IDF Feature Extraction**  
   - Converts cleaned text into numerical features  
   - Captures the importance of words relative to all reviews  

3. **Machine Learning Classification**  
   - Logistic Regression is used for binary classification  

4. **Model Saving & Loading**  
   - Trained model and TF-IDF vectorizer are saved using Joblib  
   - Can be loaded later for predicting new reviews without retraining  

---

## Technologies & Libraries Used
- *Python* 
- *Pandas & NumPy* – Data manipulation  
- *NLTK* – Stopword removal and text preprocessing  
- *Scikit-learn*– TF-IDF, Logistic Regression, evaluation metrics  
- *Matplotlib* – Confusion matrix visualization  
- *Joblib* – Saving and loading trained models  

---

## Model Performance
- **Accuracy:** 88.1%  
- **Evaluation Metrics:** Confusion matrix, Precision, Recall, F1-score  

**Sample Confusion Matrix:**

| Predicted Positive | Predicted Negative |
|------------------|------------------|
| True Positive     | False Negative   |
| False Positive    | True Negative    |

---

## Project Structure
movie-sentiment-analysis/
- dataset/
    |-README.md
    |-test.csv
    |-train.csv
- confusion_matrix.png
- logreg_model.pkl
- new_reviews.csv
- predict_sentiment.py
- sentiment_analysis.py
- tfidf_vectorizer.pkl

## How to Run

1. Install dependencies:

```bash
pip install pandas numpy scikit-learn nltk joblib

Run:
python predict_sentiment.py

Type a movie review and get Positive/Negative prediction
Type exit to quit

## Dataset

- IMDb Movie Reviews Dataset (Kaggle)
