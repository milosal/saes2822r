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

def translate_csv(input_file, output_file, max_lines=1100):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file)

    # Separate the rows to translate and the ones to leave untranslated
    df_to_translate = df.iloc[:max_lines]
    df_remaining = df.iloc[max_lines:]

    # Translate only the rows in `df_to_translate`
    for col in df_to_translate.columns:
        df_to_translate[col] = df_to_translate[col].apply(lambda x: translate_text(str(x)) if isinstance(x, str) else x)

    # Combine the translated and untranslated parts back into a single DataFrame
    translated_df = pd.concat([df_to_translate, df_remaining])

    # Save the translated DataFrame to a new CSV file
    translated_df.to_csv(output_file, index=False)
    print(f"Translation complete for the first {max_lines} lines. Saved to {output_file}")

# Usage example
input_file = 'harmful_wiki_cleaned.csv'    # Path to the original CSV file
output_file = 'harmful_wiki_cleaned_spanish.csv'  # Path for the translated CSV file
translate_csv(input_file, output_file, max_lines=1100)