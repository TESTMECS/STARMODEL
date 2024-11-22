from transformers import pipeline
import spacy

# Load a pre-trained NLP model
classifier = pipeline("text-classification", model="bert-base-uncased", return_all_scores=False)
nlp = spacy.load("en_core_web_sm")

# Input response
response = """The project faced delays. I streamlined the workflow. We finished early."""

# Step 1: Split response into sentences
doc = nlp(response)
sentences = [sent.text for sent in doc.sents]

# Step 2: Classify sentences
labels = {"Situation": 0, "Task": 0, "Action": 0, "Result": 0}
total_words = 0

for sentence in sentences:
    result = classifier(sentence)
    print(result)
    predicted_label = result[0]['label']  # Assuming the model predicts the label
    labels[predicted_label] += len(sentence.split())
    total_words += len(sentence.split())

# Step 3: Calculate percentages
percentages = {label: (count / total_words) * 100 for label, count in labels.items()}
print("STAR Percentages:", percentages)

# Step 4: Check Action percentage
if percentages["Action"] < 60:
    print(f"Warning: ACTION is only {percentages['Action']:.2f}%, ideally it should be 60%.")
else:
    print(f"ACTION is on target at {percentages['Action']:.2f}%.")


# Labels
# {"sentence": "I streamlined the workflow.", "label": "Action"}
# {"sentence": "The project faced delays.", "label": "Situation"}
