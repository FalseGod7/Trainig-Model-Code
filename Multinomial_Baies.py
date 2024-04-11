import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset (assuming you have a CSV file with 'review' and 'sentiment' columns)
# Replace 'your_dataset.csv' with the actual path to your dataset
df = pd.read_csv('Final_Review_Dataset_Copy.csv')
df = df.iloc[:40000]
df.dropna(subset=['Summary_Tamil'], inplace=True)
# Assuming 'review' contains the text data and 'sentiment' contains labels (e.g., 'positive', 'negative')
X = df['Summary_Tamil']
y = df['Sentiment']

# Vectorize text data (convert to word counts)
vectorizer = CountVectorizer()
X_counts = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_counts, y, test_size=0.2, random_state=42)

# Train the MNB model
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report (precision, recall, F1-score)
print(classification_report(y_test, y_pred))
