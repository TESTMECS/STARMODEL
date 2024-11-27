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

# Usage tutorial
print("\nUsage Tutorial:")
print("1. Load the fine-tuned model and tokenizer using the following code:")
print("    from transformers import AutoTokenizer, AutoModelForSequenceClassification")
print("    model = AutoModelForSequenceClassification.from_pretrained('./fine_tuned_model')")
print("    tokenizer = AutoTokenizer.from_pretrained('./fine_tuned_model')")
print("2. Load the label mapping with:")
print("    import json")
print("    with open('./fine_tuned_model/label_mapping.json', 'r') as f:")
print("        label_mapping = json.load(f)")
print("3. Define a function to predict the label of a given sentence:")
print("    def predict(sentence):")
print("        inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=128)")
print("        outputs = model(**inputs)")
print("        logits = outputs.logits")
print("        predicted_class_id = torch.argmax(logits, dim=1).item()")
print("        predicted_label = label_mapping[str(predicted_class_id)]")
print("        return predicted_label")
print("4. Use the predict function to get predictions for new sentences.")
