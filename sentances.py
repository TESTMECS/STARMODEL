
import csv

# Input and output file paths
input_csv = "data.csv"  # Replace with your CSV file name
output_txt = "output.txt"  # Replace with your desired output file name

# Column name or index of the sentence field in the CSV
sentence_column = "sentence"  # Replace with the actual column name for sentences

# Open the CSV and write the sentences to the text file
try:
    with open(input_csv, mode='r', newline='', encoding='utf-8') as csv_file, \
         open(output_txt, mode='w', encoding='utf-8') as txt_file:
        
        reader = csv.DictReader(csv_file)
        
        if sentence_column not in reader.fieldnames:
            raise ValueError(f"Column '{sentence_column}' not found in the CSV file.")
        
        for row in reader:
            sentence = row[sentence_column].strip()  # Get the sentence and strip extra spaces
            txt_file.write(sentence + '\n')  # Write to the text file, adding a newline
    
    print(f"Sentences successfully extracted to {output_txt}.")

except Exception as e:
    print(f"An error occurred: {e}")
