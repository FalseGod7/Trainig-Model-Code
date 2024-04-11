from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.utils import to_categorical  # Import for one-hot encoding
from sklearn.preprocessing import LabelEncoder  # Import for label encoding (optional)
import pandas as pd

# Hyperparameters (you can adjust these based on your data)
max_len = 100  # Maximum length of a review (in words)
embedding_dim = 128  # Dimensionality of word embeddings
filters = 32  # Number of filters in the convolutional layer
kernel_size = 3  # Size of the kernel in the convolutional layer
num_classes = 3  # Number of sentiment classes (positive, neutral, negative)

# Load data and labels
df = pd.read_csv('Final_Review_Dataset_Copy.csv')
df1=pd.read_csv('Pos_Neg_Final.csv')
df=df.iloc[:40000]
df1=df1.iloc[:40000]
reviews = df['Summary_Tamil']
labels = df1['Sentiment']

# Convert float to string
for i in range(len(reviews)):
  if isinstance(reviews[i], float):
    reviews[i] = str(reviews[i])  

# Preprocess text data (replace with your preprocessing steps)
tokenizer = Tokenizer(num_words=40000)  # Limit vocabulary to 5000 most frequent words
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)
padded_reviews = pad_sequences(sequences, maxlen=max_len)

# Choose one of the following options for label conversion:

# Option A: Label Mapping (Dictionary) - uncomment if preferred
# Define a label mapping (adjust class labels and values based on your data)
# label_map = {"positive": 0, "neutral": 1, "negative": 2}

# Convert labels to integer values using the mapping
# integer_labels = [label_map[label] for label in labels]

# Option B: Label Encoding (scikit-learn) - uncomment if preferred
# Encode string labels to integer indices (recommended)
encoder = LabelEncoder()
integer_labels = encoder.fit_transform(labels)

# One-hot encode the integer labels (use num_classes based on encoded classes)
labels = to_categorical(integer_labels, num_classes=encoder.classes_.shape[0])

# Define the CNN model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=embedding_dim))  # Removed input_length argument
model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))  # Ensure output layer matches number of classes

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(padded_reviews, labels, epochs=5, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(padded_reviews, labels)
print('Accuracy:', accuracy)
