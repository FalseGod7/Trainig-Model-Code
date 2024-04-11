import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load your labeled dataset (replace with your actual data)
df = pd.read_csv('Final_Review_Dataset_Copy.csv')
df=df.iloc[:40000]
df1 = pd.read_csv('pos_neg.csv')
df1=df1.iloc[:40000]

# Fill any missing values in the 'Summary_Tamil' column with an empty string
df['Summary'].fillna('', inplace=True)

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000)

# Extract features from the text data
X = tfidf_vectorizer.fit_transform(df['Summary'])

# Sentiment scores (positive/negative) from your second dataset
sentiment_score = df1['Sentiment']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, sentiment_score, test_size=0.2, random_state=42)

# Initialize the Random Forest regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict sentiment scores on the test data
y_pred = rf_model.predict(X_test)

# Evaluate model performance (Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
