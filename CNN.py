import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data

# Load your custom dataset from CSV
df = pd.read_csv('Dataset_Kannada.csv')
df1=pd.read_csv('Pos_Neg_Final.csv')
df=df.iloc[:2000]
df1=df1.iloc[:2000]

# Assuming your CSV has 'text' column for reviews and 'label' column for sentiment (0 or 1)
TEXT = data.Field(tokenize='spacy', batch_first=True)
LABEL = data.LabelField(dtype=torch.float)

fields = [('text', TEXT), ('label', LABEL)]
examples = [data.Example.fromlist([text, label], fields) for text, label in zip(df['Summary'], df1['Sentiment'])]
train_data = data.Dataset(examples, fields)

# Build vocabulary
TEXT.build_vocab(train_data, max_size=30_000, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)

# Create data iterators
BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator = data.BucketIterator(train_data, batch_size=BATCH_SIZE, device=device)

# Define your CNN model
class CNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.conv = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(3, embedding_dim))
        self.fc = nn.Linear(100, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved = self.conv(embedded).squeeze(3)
        pooled = nn.functional.max_pool1d(conved, conved.shape[2]).squeeze(2)
        return self.fc(pooled)

# Initialize model, loss function, and optimizer
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
OUTPUT_DIM = 1
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = CNNModel(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX)
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

model.to(device)
criterion.to(device)

# Training loop
def train(model, iterator, optimizer, criterion):
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        text, _ = batch.text
        predictions = model(text).squeeze(1)
        loss = criterion(predictions, batch.label)
        loss.backward()
        optimizer.step()

N_EPOCHS = 10
for epoch in range(N_EPOCHS):
    train(model, train_iterator, optimizer, criterion)

print("Training complete!")
