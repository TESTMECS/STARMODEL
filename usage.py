import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import torch
import json

# Load the CSV file
csv_file = "data.csv"
df = pd.read_csv(csv_file)

# Encode labels into numerical values
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Load the fine-tuned model and tokenizer
model_path = "./fine_tuned_model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Load the label mapping
with open(f"{model_path}/label_mapping.json", "r") as f:
    label_mapping = json.load(f)

# Function to predict the label of a given sentence
def predict(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    predicted_label = label_mapping[str(predicted_class_id)]
    return predicted_label

# Test the model on the data.csv file
df['predicted_label'] = df['sentence'].apply(predict)

# Print out a few predictions
print(df.head())
CUSTOM_TEXT = "I created the model, filled in the data, and trained it."
print(f"\nCustom text: {CUSTOM_TEXT}")
print(f"Predicted label: {predict(CUSTOM_TEXT)}")

TASK_TEST = "I was assigned this task to test the model."
print(f"\nCustom text: {TASK_TEST}")
print(f"Predicted label: {predict(TASK_TEST)}")



