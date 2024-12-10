import pandas as pd
from deep_translator import GoogleTranslator

def translate_text_batch(text_list, dest_language="es"):
    """
    Translate a batch of text to the destination language.
    Returns a list of translated strings.
    """
    try:
        translator = GoogleTranslator(source="en", target=dest_language)
        return [translator.translate(text) for text in text_list]
    except Exception as e:
        print(f"Error translating batch: {e}")
        return text_list  # Return the original list if there's an error

def translate_csv_append(input_file, output_file, start_line=1101, end_line=2300, batch_size=100):
    """
    Translate lines between start_line and end_line and append to the output CSV file.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file)

    # Extract rows to translate
    df_to_translate = df.iloc[start_line-1:end_line]

    # Translate in batches for speed optimization
    for col in df_to_translate.columns:
        if df_to_translate[col].dtype == "object":  # Translate only string columns
            col_text = df_to_translate[col].tolist()
            translated_text = []
            
            # Process in batches
            for i in range(0, len(col_text), batch_size):
                batch = col_text[i:i+batch_size]
                translated_text.extend(translate_text_batch(batch))
            
            df_to_translate[col] = translated_text

    # Append the translated rows to the output file
    with open(output_file, 'a', encoding='utf-8') as f:
        df_to_translate.to_csv(f, header=f.tell()==0, index=False)
    
    print(f"Translation complete for lines {start_line} to {end_line}. Appended to {output_file}")

# Usage example
input_file = 'harmful_wiki_cleaned.csv'    # Path to the original CSV file
output_file = 'harmful_wiki_cleaned_spanish.csv'  # Path for the translated CSV file
translate_csv_append(input_file, output_file, start_line=1101, end_line=2300, batch_size=100)