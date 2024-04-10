import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

# Load your review dataset (replace with your actual data)
df = pd.read_csv("Final_Review_Dataset_Copy.csv")
df=df.dropna(axis=0, inplace=True)

# Assuming your DataFrame is named 'df'


# Create an imputer for mode imputation
#imputer = SimpleImputer(strategy='mean')  # Or any other strategy you want to use
imputer = SimpleImputer(strategy='constant', fill_value='missing')  # Adjust strategy and fill_value as needed

# Impute missing values in the 'Summary' column
#df['Summary'] = imputer.fit_transform(df[['Summary']])

# Repeat the same process for 'Summary_Tamil' if needed

# trial Version for imputing data
try:
  # Impute the 'Summary' column (assuming it has missing values)
  df['Summary'] = imputer.fit_transform(df[['Summary']])
except (KeyError, ValueError) as e:
  # Handle potential errors like missing column or invalid data types
  print(f"Error during imputation: {e}")
  
# Assume you have 'text' column containing review text and 'sentiment' column

X = df['Summary']
y = df['Sentiment']


# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000)

# Transform text data to TF-IDF features
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize logistic regression model
lr_model = LogisticRegression()

# Train the model
lr_model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = lr_model.predict(X_test_tfidf)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {accuracy:.4f}")

# Classification report (optional but informative)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Continue with your analysis:
# - Interpret feature importance (if applicable)
# - Investigate misclassified samples
# - Explore other metrics (precision, recall, F1-score, etc.)
# - Visualize results (e.g., confusion matrix, ROC curve)

# Example: Investigate misclassified samples
misclassified_samples = X_test[y_test != y_pred]
print("\nMisclassified Samples:")
for sample in misclassified_samples [:5] :
    print(sample)

# Feel free to adapt and extend the analysis based on your specific goals!
