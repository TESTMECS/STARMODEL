import pickle
from transformers import BertTokenizer, BertForSequenceClassification
import json
# Load the fine-tuned model and tokenizer
model_path = "./fine_tuned_model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
# Save the model and tokenizer
with open("model_tokenizer.pkl", "wb") as f:
    pickle.dump((model, tokenizer), f)
print("Model and tokenizer pickled.")
