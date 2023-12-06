import json
import sys

def convert_to_json(input_file_path, output_file_path):
    # Read the text data from the file
    with open(input_file_path, "r", encoding="utf-8") as input_file:
        text_data = input_file.read()

    # Convert the text data to the desired format
    json_data = {"data": []}
    for line in text_data.split("\n"):
        if line.strip():  # Check if the line is not empty
            translation = json.loads(line)["translation"]
            json_data["data"].append({"de": translation["de"], "en": translation["en"]})

    # Save the data to a JSON file
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        json.dump(json_data, output_file, indent=2, ensure_ascii=False)

    print(f"Data has been converted and saved to {output_file_path}")

if __name__ == "__main__":
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python script.py input_file.txt output_file.json")
        sys.exit(1)

    # Get the input and output file paths from command-line arguments
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    # Call the conversion function
    convert_to_json(input_file_path, output_file_path)
