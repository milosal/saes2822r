import pandas as pd
from deep_translator import GoogleTranslator

def translate_text(text, dest_language="es"):
    """Translate a single text to the destination language (Spanish by default)."""
    try:
        translated = GoogleTranslator(source="en", target=dest_language).translate(text)
        return translated
    except Exception as e:
        print(f"Error translating text '{text}': {e}")
        return text  # Return the original text if there's an error

def translate_csv(input_file, output_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file)

    # Translate each cell in the DataFrame
    for col in df.columns:
        df[col] = df[col].apply(lambda x: translate_text(str(x)) if isinstance(x, str) else x)

    # Save the translated DataFrame to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Translation complete. Saved to {output_file}")

# Usage example
input_file = 'annotated_response_ChatGLM2.csv'    # Path to the original CSV file
output_file = 'spanish_benign.csv'  # Path for the translated CSV file
translate_csv(input_file, output_file)