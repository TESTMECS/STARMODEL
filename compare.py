
import csv

# Input file paths
data_csv = "data.csv"  # The reference data file
response_csv = "sampleResponse.csv"  # The response file to evaluate

# Column names
sentence_column = "sentence"  # Column name for sentences
label_column = "label"  # Column name for labels

try:
    # Load DATA.csv into a dictionary
    data_dict = {}
    with open(data_csv, mode='r', newline='', encoding='utf-8') as data_file:
        data_reader = csv.DictReader(data_file)
        
        for row in data_reader:
            sentence = row[sentence_column].strip()
            label = row[label_column].strip()
            data_dict[sentence] = label
    
    # Compare RESPONSE.csv to DATA.csv
    correct = 0
    total = 0

    with open(response_csv, mode='r', newline='', encoding='utf-8') as response_file:
        response_reader = csv.DictReader(response_file)
        
        for row in response_reader:
            sentence = row[sentence_column].strip()
            label = row[label_column].strip()
            
            # Check if the sentence exists in DATA.csv and compare labels
            if sentence in data_dict:
                total += 1
                if data_dict[sentence] == label:
                    correct += 1

    # Calculate percentage correct
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"Correct: {correct}, Total: {total}, Accuracy: {accuracy:.2f}%")

except Exception as e:
    print(f"An error occurred: {e}")
