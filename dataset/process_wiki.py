import pandas as pd

def filter_and_save_comments(input_file, output_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file)

    # Filter rows where the label is 1 and select only the 'comment_text' column
    filtered_comments = df[df['label'] == 1]['comment_text']

    # Save the filtered comments to a new CSV file
    filtered_comments.to_csv(output_file, index=False, header=True)
    print(f"Filtered comments saved to {output_file}")

# Usage example
input_file = 'wiki_comments.csv'     # Path to the original CSV file
output_file = 'filtered_comments.csv'  # Path for the filtered comments CSV file
filter_and_save_comments(input_file, output_file)