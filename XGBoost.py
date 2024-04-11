# Import necessary libraries
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

# Load your review dataset (replace with your actual data)
# Assume you have a CSV file with columns: 'review_text' and 'sentiment'
df = pd.read_csv('Dataset_Kannada.csv')
df1=pd.read_csv('Pos_Neg_Final.csv')

df=df.iloc[:20000]
df1=df1.iloc[:20000]

df['Summary_Kannada'].fillna('', inplace=True)
# Preprocessing: Clean text, tokenize, and convert to lowercase
# You can use NLTK or spaCy for tokenization and other preprocessing steps

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    df['Summary_Kannada'], df1['Sentiment_1'], test_size=0.2, random_state=42
)

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_tfidf = tfidf_vectorizer.transform(X_val)

# Initialize XGBoost classifier
xgb_classifier = xgb.XGBClassifier(
    objective='multi:softmax',  # For multiclass classification
    num_class=len(df1['Sentiment'].unique()),  # Number of sentiment classes
    max_depth=5,  # Adjust hyperparameters as needed
    learning_rate=0.1,
    n_estimators=100
)

# Train the model
xgb_classifier.fit(X_train_tfidf, y_train)

# Predictions on validation set
y_val_pred = xgb_classifier.predict(X_val_tfidf)

# Evaluate model performance
print(f"Accuracy: {accuracy_score(y_val, y_val_pred)}")
print(classification_report(y_val, y_val_pred))

# Now you can deploy this model for inference on new reviews!
