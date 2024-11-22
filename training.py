import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import os

# Ensure the directory exists
os.makedirs("./results", exist_ok=True)

# Change permissions to 777
os.chmod("./results", 0o777)

# Load the CSV file
csv_file = "data.csv"
df = pd.read_csv(csv_file)

# Encode labels into numerical values
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Split data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['sentence'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)

# Create datasets
train_data = Dataset.from_dict({"sentence": train_texts, "label": train_labels})
val_data = Dataset.from_dict({"sentence": val_texts, "label": val_labels})

# Initialize the tokenizer
model_name = "bert-base-uncased"  # You can use any other pretrained model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize datasets
def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True, padding=True, max_length=128)

train_data = train_data.map(preprocess_function, batched=True)
val_data = val_data.map(preprocess_function, batched=True)

# Load the model
num_labels = len(label_encoder.classes_)  # Number of unique labels
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# Define data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Compute metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), axis=1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# Map labels back to text for future reference
label_mapping = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))
with open("./fine_tuned_model/label_mapping.json", "w") as f:
    import json
    json.dump(label_mapping, f)

print("Model training complete. Saved to './fine_tuned_model'.")
