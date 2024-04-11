import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset (replace with actual data)
df = pd.read_csv('Final_Review_Dataset_Copy.csv')
df = df.iloc[:40000]
try:
  if 'Summary_Tamil' in df.columns:
    reviews = df['Summary_Tamil']
  else:
    print("Error: 'Summary_Tamil' column not found in the DataFrame.")
    exit()
except FileNotFoundError:
  print("Error: CSV file 'Final_Review_Dataset_Copy.csv' not found.")
  exit()

df1 = pd.read_csv('Pos_Neg_Final.csv')
df1 = df1.iloc[:40000]
df.fillna('', inplace=True)  # Fill missing values in the entire DataFrame

# Encode 'sentiment' as 0 (negative) or 1 (positive)
le = LabelEncoder()
df['Sentiment'] = le.fit_transform(df1['Sentiment'])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(reviews, df['Sentiment'], test_size=0.2, random_state=42)

# Vectorize text data (convert to bag-of-words representation)
vectorizer = CountVectorizer(max_features=1000)  # Adjust max_features as needed
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Apply TF-IDF transformation
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Predictions on test data
y_pred = model.predict(X_test_tfidf)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)
