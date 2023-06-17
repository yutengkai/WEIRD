pip install --upgrade openai

from google.colab import drive
import pandas as pd
import openai
import re
from sklearn.metrics import roc_curve, auc, precision_recall_curve, classification_report
import matplotlib.pyplot as plt

openai.api_key = "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"

drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/DeClare/test_set.csv').drop('Unnamed: 0', axis=1)

ground_truth_labels = df['label'].tolist()
ground_truth_labels = [1 if label==True else 0 for label in ground_truth_labels]
predicted_probs = []

for _, data_sample in df.iterrows():
    claim = data_sample['claim']
    evidence = data_sample['text']
    prompt = 'What is the possibility of the following claim to be true given the following text?' + '\nClaim: ' + claim + '\nText: ' + evidence + '\nOnly write your estimated probability percentage.'

    completions = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5
    )
    prediction = completions.choices[0].text
    prediction = prediction.strip()
    try:
      prediction = re.findall(r'\d+', prediction)[0]
      prediction = int(prediction)
    except:     # ChatGPT cannot estimate the probability!
      prediction = 50
    predicted_probs.append(float(prediction)/100)

predicted_labels = [1 if p > 0.5 else 0 for p in predicted_probs]

# Classification report
print(classification_report(ground_truth_labels, predicted_labels))

# Calculate ROC AUC
fpr, tpr, _ = roc_curve(ground_truth_labels, predicted_probs)
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
precision, recall, _ = precision_recall_curve(ground_truth_labels, predicted_probs)
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
