import json
from deep_translator import GoogleTranslator

def translate_text(text, dest_language="es"):
    """Translate a single text to the destination language (Spanish by default)."""
    try:
        translated = GoogleTranslator(source="en", target=dest_language).translate(text)
        return translated
    except Exception as e:
        print(f"Error translating text '{text}': {e}")
        return text  # Return the original text if there's an error

def translate_json_partial(input_file, output_file, start=1101, end=2300):
    # Load the JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Load existing translated data if the output file exists
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            translated_data = json.load(f)
    except FileNotFoundError:
        translated_data = []  # Initialize empty if no existing file

    # Extract the range of entries to translate
    if isinstance(data, list):
        data_to_translate = data[start-1:end]
    else:
        print("Error: JSON data is not a list. Ensure the input JSON file contains a list of entries.")
        return

    # Translate the specified range
    translated_chunk = []
    for item in data_to_translate:
        if isinstance(item, dict):
            translated_chunk.append({key: translate_text(value) if isinstance(value, str) else value for key, value in item.items()})
        elif isinstance(item, list):
            translated_chunk.append([translate_text(value) if isinstance(value, str) else value for value in item])
        elif isinstance(item, str):
            translated_chunk.append(translate_text(item))
        else:
            translated_chunk.append(item)

    # Append the translated chunk to the existing data
    translated_data.extend(translated_chunk)

    # Save the combined translated data back to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=4)

    print(f"Translation complete for entries {start} to {end}. Appended to {output_file}")

# Usage example
input_file = 'alpaca_data.json'    # Path to the original JSON file
output_file = 'alpaca_spanish.json'  # Path for the translated JSON file
translate_json_partial(input_file, output_file, start=1101, end=2300)