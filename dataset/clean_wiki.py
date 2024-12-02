import pandas as pd
import re

# Read the CSV file
df = pd.read_csv('filtered_comments.csv')

# Define the cleaning function
def clean_text(text):
    if pd.isnull(text):
        return ''
    # Remove newlines and carriage returns
    text = re.sub(r'[\r\n]+', ' ', text)
    # Remove unwanted punctuation (keeping basic punctuation)
    text = re.sub(r'[^\w\s\.\,\!\?\']', '', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Clean the 'comment_text' column
df['comment_text'] = df['comment_text'].apply(clean_text)

# Define the word count function
def count_words(text):
    return len(text.split())

# Apply the word count function
df['word_count'] = df['comment_text'].apply(count_words)

# Filter the DataFrame to only include comments with 50 words or fewer
df = df[df['word_count'] <= 100]

# Optional: Drop the 'word_count' column if it's no longer needed
df = df.drop('word_count', axis=1)

# Save the cleaned and filtered data
df.to_csv('yourfile_cleaned.csv', index=False)