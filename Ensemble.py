import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Load and preprocess your data
# Replace this comment with your data loading and preprocessing steps
df = pd.read_csv('Final_Review_Dataset_Copy.csv')
df = df.iloc[:4000]
df1 = pd.read_csv('pos_neg.csv')
df1 = df1.iloc[:4000]


# 2. Data Preprocessing
df['Summary'].fillna('', inplace=True)
l1=len(df)
df1 = df1.iloc[:l1]
X = df['Summary']
y=df1['Sentiment']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define constants
max_seq_length = 100  # Update this with your actual maximum sequence length
vocab_size = 10000    # Update this with your actual vocabulary size

# Create Bi-LSTM model
def create_bilstm_model(input_shape, vocab_size, embedding_dim):
  input_layer = Input(shape=input_shape)
  embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input_layer)
  bilstm_layer = Bidirectional(LSTM(64, return_sequences=True))(embedding_layer)
  maxpool_layer = GlobalMaxPooling1D()(bilstm_layer)
  dense_layer = Dense(64, activation='relu')(maxpool_layer)
  output_layer = Dense(1, activation='sigmoid')(dense_layer)
  model = Model(inputs=input_layer, outputs=output_layer)
  return model

# Create CNN model
def create_cnn_model(input_shape, vocab_size, embedding_dim):
  input_layer = Input(shape=input_shape)
  embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input_layer)
  conv_layer = Conv1D(128, kernel_size=3, activation='relu')(embedding_layer)
  maxpool_layer = GlobalMaxPooling1D()(conv_layer)
  dense_layer = Dense(64, activation='relu')(maxpool_layer)
  output_layer = Dense(1, activation='sigmoid')(dense_layer)
  model = Model(inputs=input_layer, outputs=output_layer)
  return model

# Compile and train Bi-LSTM model
bilstm_model = create_bilstm_model(input_shape=(max_seq_length,), vocab_size=vocab_size, embedding_dim=100)
bilstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
bilstm_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val))

# Compile and train CNN model
cnn_model = create_cnn_model(input_shape=(max_seq_length,), vocab_size=vocab_size, embedding_dim=100)
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val))

# Ensemble: Combine predictions from both models
bilstm_preds = bilstm_model.predict(X_val)
cnn_preds = cnn_model.predict(X_val)
ensemble_preds = (bilstm_preds + cnn_preds) / 2  # Simple averaging

# Evaluate the ensemble
ensemble_accuracy = accuracy_score(y_val, ensemble_preds > 0.5)
ensemble_f1 = f1_score(y_val, ensemble_preds > 0.5)
print("Ensemble Accuracy:", ensemble_accuracy)
print("Ensemble F1 Score:", ensemble_f1)
