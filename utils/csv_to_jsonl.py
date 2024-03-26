import csv
import json

csv_file = './data/testsuite/testsuite.csv'  # Replace with your CSV file path
json_file = './data/testsuite/testsuite.jsonl'  # Replace with the desired JSON file path

data = []

with open(csv_file, 'r', encoding='utf-8', errors='replace') as csv_input:
    csv_reader = csv.DictReader(csv_input)
    for row in csv_reader:
        # Replace invalid characters with a placeholder (e.g., '?')
        cleaned_row = {k: v.replace('\ufffd', '?') for k, v in row.items()}
        data.append(cleaned_row)

with open(json_file, 'w', encoding='utf-8') as json_output:
    for item in data:
        json_output.write((json.dumps(item, ensure_ascii=False) + "\n"))
        # json.dump(item, json_output, ensure_ascii=False, indent=4)
    # json.dump(data, json_output, ensure_ascii=False, indent=4)