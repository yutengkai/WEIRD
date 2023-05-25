import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import roc_curve, auc, precision_recall_curve, classification_report
import matplotlib.pyplot as plt
import pandas as pd

# Define device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load the BERT tokenizer and model
model_path = "/content/drive/MyDrive/DeClare/models/bert_model.pth"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
model.load_state_dict(torch.load(model_path, map_location=device))  # Load model to the device
model.to(device)  # Move model to the device
model.eval()

# Prepare the input data
df = pd.read_csv("/content/drive/MyDrive/DeClare/test_set.csv")
df['input'] = df['claim'] + tokenizer.sep_token + df['text']
inputs = [tokenizer.encode(x, max_length=512, padding="max_length", truncation=True) for x in df['input'].tolist()]
inputs = torch.tensor(inputs).to(device)  # Move inputs to the device

df['label'] = df['label'].map({'SUPPORTS': True, 'REFUTES': False})
labels = torch.tensor(df['label'].tolist()).to(device)  # Move labels to the device

# Create a TensorDataset and DataLoader
dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset, batch_size=64)

# Evaluate the model
probs = []
true_labels = []
with torch.no_grad():
    for input, label in dataloader:
        output = model(input)[0]
        prob = torch.sigmoid(output)
        probs.extend(prob.detach().cpu().numpy().tolist())  # Move probabilities to cpu before converting to numpy
        true_labels.extend(label.cpu().numpy().tolist())  # Move labels to cpu before converting to numpy

true_labels = [int(label) for label in true_labels]
probs = [prob[0] for prob in probs]

# Calculate ROC AUC
fpr, tpr, _ = roc_curve(true_labels, probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Calculate PR AUC
precision, recall, _ = precision_recall_curve(true_labels, probs)
pr_auc = auc(recall, precision)

# Plot PR curve
plt.figure()
plt.plot(recall, precision, color='darkorange', label='PR curve (area = %0.2f)' % pr_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc="lower right")
plt.show()

# Print classification report
print(classification_report(true_labels, [1 if p > 0.5 else 0 for p in probs]))
