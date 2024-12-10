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

def translate_json(input_file, output_file, max_translations=None):
    # Load the JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Keep track of the number of translations
    translation_count = 0

    # Recursive function to translate strings up to the max_translations limit
    def translate_recursive(data):
        nonlocal translation_count
        print(translation_count)

        if isinstance(data, dict):
            return {key: translate_recursive(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [translate_recursive(element) for element in data]
        elif isinstance(data, str):
            # Only translate if translation limit has not been reached
            if max_translations is None or translation_count < max_translations:
                translation_count += 1
                return translate_text(data)
            else:
                return data
        else:
            return data

    # Translate the JSON data structure
    translated_data = translate_recursive(data)

    # Save the translated JSON data to a new file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=4)
    
    print(f"Translation complete. Translated {translation_count} items. Saved to {output_file}")

# Usage example
input_file = 'alpaca_data.json'    # Path to the original JSON file
output_file = 'alpaca_spanish.json'  # Path for the translated JSON file
translate_json(input_file, output_file, max_translations=1100)  # Set max_translations to 100