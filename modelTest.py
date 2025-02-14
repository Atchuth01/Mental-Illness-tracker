##Mental illeness tracker model testing

import joblib
import numpy as np
import re
import nltk
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load trained models
svm_model = joblib.load("models/sentiment_model_svm.pkl")  # SVM Model
bilstm_model = tf.keras.models.load_model("models/bilstm_sentiment_model.h5")  # BiLSTM Model
sentiment_label_encoder = joblib.load("models/sentiment_label_encoder.pkl")  # Label encoder
impact_model = joblib.load("models/impact_model.pkl")  # Regression Model
scaler = joblib.load("models/scaler.pkl")  # Scaler for behavioral data
word2vec_model = Word2Vec.load("models/word2vec_model.bin")  # Fine-tuned Word2Vec model

max_sequence_length = 100  # Must match training settings
alpha = 0.5  # Weight factor for ensemble (adjustable)

# Function to clean text while preserving negations
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english') or word in ["not", "never", "no"]]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens  # Returning list (needed for Word2Vec & BiLSTM)

# Convert cleaned text into Word2Vec vector
def text_to_vector(tokens):
    vectors = [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)  # Vector size = 100

# Convert cleaned text into BiLSTM input sequence
def tokens_to_sequence(tokens):
    vectors = [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv]
    vectors = vectors[:max_sequence_length]  # Trim if too long
    while len(vectors) < max_sequence_length:
        vectors.append(np.zeros(100))  # Pad if too short
    return np.array([vectors])  # Shape: (1, max_sequence_length, 100)

# Function to predict sentiment using ensemble model
def predict_sentiment(text):
    tokens = clean_text(text)
    text_vector = text_to_vector(tokens).reshape(1, -1)
    text_sequence = tokens_to_sequence(tokens)

    # Predict probabilities
    svm_prob = svm_model.predict_proba(text_vector)[0]  # Get probability distribution
    bilstm_prob = bilstm_model.predict(text_sequence)[0]  # BiLSTM already outputs softmax

    # Weighted probability fusion
    final_prob = alpha * svm_prob + (1 - alpha) * bilstm_prob

    # Final prediction based on highest probability
    final_pred = np.argmax(final_prob)
    return sentiment_label_encoder.inverse_transform([final_pred])[0]

# Function to predict mental health impact score
def predict_impact(age, gender, social_media_freq, self_reported_status):
    df_input = pd.DataFrame({
        'Age': [age],
        'Gender_Female': [1 if gender.lower() == "female" else 0],
        'Gender_Male': [1 if gender.lower() == "male" else 0],
        'Frequency of Social Media Interaction_Frequently': [1 if social_media_freq.lower() == "frequently" else 0],
        'Frequency of Social Media Interaction_Occasionally': [1 if social_media_freq.lower() == "occasionally" else 0],
        'Frequency of Social Media Interaction_Rarely': [1 if social_media_freq.lower() == "rarely" else 0],
        'Frequency of Social Media Interaction_Very Often': [1 if social_media_freq.lower() == "very often" else 0],
        'Self-reported Mental Health Status_Excellent': [1 if self_reported_status.lower() == "excellent" else 0],
        'Self-reported Mental Health Status_Fair': [1 if self_reported_status.lower() == "fair" else 0],
        'Self-reported Mental Health Status_Good': [1 if self_reported_status.lower() == "good" else 0],
        'Self-reported Mental Health Status_Poor': [1 if self_reported_status.lower() == "poor" else 0]
    })
    
    input_scaled = scaler.transform(df_input)
    impact_score = impact_model.predict(input_scaled)[0]
    return round(impact_score, 2)

# Test cases for sentiment classification
test_sentences = [
    "I feel empty, unmotivated, and nothing excites me anymore.",  
    "I don't see a way out of this pain. I'm done with everything.",  
    "My heart is racing, and my mind won’t stop worrying about everything.",  
    "One moment I feel like I can do anything, the next I just crash completely.",  
    "I have too much on my plate, and it's overwhelming. I can't focus on anything!",  
    "I don’t understand people, and I feel like no one understands me either."  
]

print("\n🔹 Sentiment Predictions:")
for i, text in enumerate(test_sentences, 1):
    print(f"Test {i} Prediction: {predict_sentiment(text)}")

# Test case for impact prediction
age = 23
gender = "Female"
social_media_freq = "Very Often"
self_reported_status = "Anxiety"
print(f"\n🔹 Predicted Mental Health Impact Score: {predict_impact(age, gender, social_media_freq, self_reported_status)}")
