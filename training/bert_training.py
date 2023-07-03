!pip install transformers
from transformers import BertTokenizer, BertModel, BertForPreTraining, BertForSequenceClassification, EncoderDecoderModel, AdamW
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, classification_report
import matplotlib.pyplot as plt

"""# 1. Load data"""

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/DeClare/bert_set.csv').drop('Unnamed: 0', axis=1)  # a subset of FEVER dataset for training the bert model
df.head()

"""# 2. Tokenize input"""

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

df['input'] = df['claim'] + bert_tokenizer.sep_token + df['text']

lengths = df.input.apply(lambda x: len(x.split()))
print(lengths.min())
print(lengths.max())
print(lengths.mean())
print(lengths.median())
lengths.plot.hist()

def bert_encoder(seq, tokenizer, max_length=512):
  encoding = tokenizer.encode(seq, max_length=max_length, padding="max_length", truncation=True)
  return encoding
df['input_tokens'] = df.input.apply(bert_encoder, tokenizer=bert_tokenizer)
# df.head()

df.to_csv('/content/drive/MyDrive/DeClare/input_encoded_base_512.csv', index=False)

"""# 3. Train and fine tune the model"""

df = pd.read_csv('/content/drive/MyDrive/DeClare/input_encoded_base_512.csv')

df['input_tokens'] = df['input_tokens'].apply(eval)

input_tokens, label_tokens = df.input_tokens.to_list(), df.label.to_list()

train_input, val_test_input, train_label, val_test_label = train_test_split(input_tokens, label_tokens, test_size=0.2, random_state=42)
val_input, test_input, val_label, test_label = train_test_split(val_test_input, val_test_label, test_size=0.5, random_state=42)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device

train_input = torch.tensor(train_input).to(device)
train_label = torch.tensor(train_label).to(device)
val_input = torch.tensor(val_input).to(device)
val_label = torch.tensor(val_label).to(device)
test_input = torch.tensor(test_input).to(device)
test_label = torch.tensor(test_label).to(device)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1).to(device)
# model.train()

train_data = TensorDataset(train_input, train_label)
val_data = TensorDataset(val_input, val_label)
test_data = TensorDataset(test_input, test_label)

batch_size = 16   # 64 and 32 don't work for me :(
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

optimizer = AdamW(model.parameters(), lr=5e-5)  # Should I change the learning rate?

num_epochs = 3
train_loss_list = []
val_loss_list = []

for epoch in range(num_epochs):
    train_loss = 0
    for input_batch, label_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(input_ids=input_batch, labels=label_batch.float())
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_loss_list.append(train_loss)

    val_loss = 0
    with torch.no_grad():
        for input_batch, label_batch in val_loader:
            outputs = model(input_ids=input_batch, labels=label_batch.float())
            loss = outputs.loss
            val_loss += loss.item()
        val_loss /= len(val_loader)
        val_loss_list.append(val_loss)

    print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

torch.save(model.state_dict(), '/content/drive/MyDrive/DeClare/bert_model.pth')

model.load_state_dict(torch.load('/content/drive/MyDrive/DeClare/bert_model.pth'))

test_loss = 0
logits = torch.tensor([]).to(device)
with torch.no_grad():
    for input_batch, label_batch in test_loader:
        outputs = model(input_ids=input_batch, labels=label_batch.float())
        logits = torch.cat([logits, outputs.logits])
        loss = outputs.loss
        test_loss += loss.item()
    test_loss /= len(test_loader)

print(f'Test Loss = {test_loss:.4f}')
