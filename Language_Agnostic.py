# Install necessary libraries (if not already installed)
#!pip install transformers torch
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW

# Load pre-trained DistilBERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)  # 3 labels: positive, negative, neutral

# Load your dataset (reviews and sentiments)
# Assume 'reviews' contains your text data and 'labels' contains sentiment labels (0, 1, 2)
df = pd.read_csv('Final_Review_Dataset_Copy.csv')
df = df.iloc[:40]
df1 = pd.read_csv('pos_neg.csv')
df1 = df1.iloc[:40]

# Convert 'Summary' column to a list of strings
reviews_list = df['Summary_Tamil'].tolist()

# Tokenize and encode reviews
inputs = tokenizer(reviews_list, padding=True, truncation=True, return_tensors='pt', max_length=128)
labels = torch.tensor(df1['Sentiment'])

# Convert sentiment labels to one-hot encoded vectors
num_classes = 3  # Number of sentiment classes (positive, negative, neutral)
one_hot_labels = torch.zeros(len(labels), num_classes)
one_hot_labels.scatter_(1, labels.long().unsqueeze(1), 1)

# Set up optimizer and training parameters
optimizer = AdamW(model.parameters(), lr=1e-5)
epochs = 3

# Training loop
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(**inputs, labels=one_hot_labels)  # Use one-hot encoded labels
    loss = outputs.loss
    loss.backward()
    optimizer.step()

# Evaluate your model (similar to training loop)
# You can calculate accuracy, F1-score, etc. (not shown in this snippet)

# Save your trained model for inference
torch.save(model.state_dict(), 'my_sentiment_model.pth')

# Load the saved model for inference
loaded_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
loaded_model.load_state_dict(torch.load('my_sentiment_model.pth'))
loaded_model.eval()  # Set to evaluation mode

# Example: Predict sentiments for new reviews
new_reviews = ["This movie was fantastic!", "The service was terrible."]
for review in new_reviews:
    inputs = tokenizer(review, padding=True, truncation=True, return_tensors='pt', max_length=128)
    with torch.no_grad():
        outputs = loaded_model(**inputs)
        predicted_label = torch.argmax(outputs.logits, dim=1).item()
        sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
        print(f"Review: {review}")
        print(f"Predicted Sentiment: {sentiment_mapping[predicted_label]}")
        print("-" * 30)
        
# Now you'll see the predicted sentiments printed for each review
