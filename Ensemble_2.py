import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Load and preprocess your data
df = pd.read_csv('Dataset_Kannada.csv')
df = df.iloc[:20000]

df1 = pd.read_csv('Pos_Neg_Final.csv')
df1 = df1.iloc[:20000]

# Text cleaning (consider more techniques like stemming/lemmatization)
X = df['Summary_Kannada'].fillna('').str.lower().replace('[^a-zA-Z0-9\s]', '', regex=True)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, df1['Sentiment'], test_size=0.2, random_state=42)

# Define constants
max_seq_length = 100  # Adjust this with your actual maximum sequence length
vocab_size = 10000  # Update this with your actual vocabulary size

# Create tokenizers for text-to-sequence conversion
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X_train)

# Convert text to sequences
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_val_sequences = tokenizer.texts_to_sequences(X_val)

# Pad sequences to the same length
X_train_padded = pad_sequences(X_train_sequences, maxlen=max_seq_length)
X_val_padded = pad_sequences(X_val_sequences, maxlen=max_seq_length)

# Function to create a Bi-LSTM model
def create_bilstm_model(input_shape, vocab_size, embedding_dim):
    input_layer = Input(shape=input_shape)
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input_layer)
    bilstm_layer = Bidirectional(LSTM(64, return_sequences=True))(embedding_layer)
    maxpool_layer = GlobalMaxPooling1D()(bilstm_layer)
    dense_layer = Dense(64, activation='relu')(maxpool_layer)
    output_layer = Dense(1, activation='sigmoid')(dense_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Function to create a CNN model
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
bilstm_history = bilstm_model.fit(X_train_padded, y_train, epochs=5, batch_size=32, validation_data=(X_val_padded, y_val))

# Print Bi-LSTM model training accuracy
bilstm_train_acc = bilstm_history.history['accuracy'][-1]
bilstm_val_acc = bilstm_history.history['val_accuracy'][-1]
print(f"Bi-LSTM Model Training Accuracy: {bilstm_train_acc:.4f}")
print(f"Bi-LSTM Model Validation Accuracy: {bilstm_val_acc:.4f}")

# Compile and train CNN model
cnn_model = create_cnn_model(input_shape=(max_seq_length,), vocab_size=vocab_size, embedding_dim=100)
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_history = cnn_model.fit(X_train_padded, y_train, epochs=5, batch_size=32, validation_data=(X_val_padded, y_val))

# Print CNN model training accuracy
cnn_train_acc = cnn_history.history['accuracy'][-1]
cnn_val_acc = cnn_history.history['val_accuracy'][-1]
print(f"CNN Model Training Accuracy: {cnn_train_acc:.4f}")
print(f"CNN Model Validation Accuracy: {cnn_val_acc:.4f}")

