##Mental-Illness-tracker
#AI-Driven Mental Health Analysis from Social Media

import pandas as pd
import numpy as np
import re
import nltk
import joblib
import tensorflow as tf

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from gensim.models import Word2Vec
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load balanced dataset
df_text = pd.read_csv("datasets/socialMedia_mentalHealth.csv")
df_behavior = pd.read_csv("datasets/mentalHealth_Behavior.csv")

# Encode sentiment labels
label_encoder = LabelEncoder()
df_text['sentiment_label'] = label_encoder.fit_transform(df_text['label'])

# Save label encoder
joblib.dump(label_encoder, "models/sentiment_label_encoder.pkl")

# Text Cleaning Function (keeping negations)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english') or word in ["not", "never", "no"]]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens  # Return list instead of string for Word2Vec

# Apply preprocessing
print("Starting text preprocessing...")
df_text['cleaned_tokens'] = df_text['text'].astype(str).apply(clean_text)
print("Text preprocessing completed!")

# Train Fine-Tuned Word2Vec model
word2vec_model = Word2Vec(sentences=df_text['cleaned_tokens'], vector_size=100, window=5, min_count=1, workers=4)
word2vec_model.save("models/word2vec_model.bin")

# Function to convert text to vector
def text_to_vector(tokens):
    vectors = [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)

# Convert text data into vectors
X_text = np.array([text_to_vector(tokens) for tokens in df_text['cleaned_tokens']])
y_text = df_text['sentiment_label']

# Split dataset
X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(X_text, y_text, test_size=0.2, random_state=42)

# Train SVM classifier
sentiment_model_svm = SVC(kernel='linear', probability=True)
sentiment_model_svm.fit(X_train_text, y_train_text)

# Save SVM model
joblib.dump(sentiment_model_svm, "models/sentiment_model_svm.pkl")

print("SVM Sentiment model trained successfully!")

# Train BiLSTM Model
max_sequence_length = 100

# Function to convert tokens into padded sequences for LSTM
def tokens_to_sequence(tokens):
    vectors = [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv]
    vectors = vectors[:max_sequence_length]  # Trim if too long
    while len(vectors) < max_sequence_length:
        vectors.append(np.zeros(100))  # Pad if too short
    return np.array(vectors)  # Shape: (max_sequence_length, 100)

# Convert entire dataset to sequences
# Convert tokens to word embeddings and pad sequences
X_lstm = [np.array([word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv]) for tokens in df_text['cleaned_tokens']]
X_lstm = [seq if len(seq) > 0 else np.zeros((1, 100)) for seq in X_lstm]  # Ensure all sequences have at least one vector
X_lstm = pad_sequences(X_lstm, maxlen=max_sequence_length, dtype='float32', padding='post', truncating='post')

# Ensure X_lstm is (samples, timesteps, features)
X_lstm = np.array(X_lstm)  # Convert list to NumPy array

# Split LSTM dataset
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_text, test_size=0.2, random_state=42)

# Define BiLSTM model
lstm_model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True, input_shape=(max_sequence_length, 100))),  # Set input shape explicitly
    Bidirectional(LSTM(32)),
    Dense(16, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

lstm_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train LSTM model
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10, validation_data=(X_test_lstm, y_test_lstm))

# Save LSTM model
lstm_model.save("models/bilstm_sentiment_model.h5")

print("BiLSTM Sentiment model trained successfully!")

# Encode behavioral dataset
categorical_cols = ['Gender', 'Frequency of Social Media Interaction', 'Self-reported Mental Health Status']
df_behavior_encoded = pd.get_dummies(df_behavior, columns=categorical_cols)

# Define input and output
X_behavior = df_behavior_encoded.drop(columns=['Respondent', 'Impact on Mental Health (Score)'])
y_behavior = df_behavior_encoded['Impact on Mental Health (Score)']

# Standardize data
scaler = StandardScaler()
X_behavior_scaled = scaler.fit_transform(X_behavior)

# Split dataset
X_train_behavior, X_test_behavior, y_train_behavior, y_test_behavior = train_test_split(
    X_behavior_scaled, y_behavior, test_size=0.2, random_state=42)

# Train regression model
impact_model = RandomForestRegressor(n_estimators=100, random_state=42)
impact_model.fit(X_train_behavior, y_train_behavior)

# Save models
joblib.dump(impact_model, "models/impact_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Impact prediction model trained successfully!")
