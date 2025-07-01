import os
import re
import logging
from nltk.tokenize import word_tokenize, sent_tokenize
from newspaper import Article

# Configure logging to display messages for debugging and status updates
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def extract_article(url, output_path):
    """
    Extracts the content of an article from a URL and saves it to a file.
    Skips processing if the file already exists and is non-empty.

    Parameters:
        url (str): The URL of the article.
        output_path (str): The file path to save the extracted content.

    Returns:
        str: The extracted text of the article, or None if extraction fails.
    """
    # Skip processing if the file already exists and contains content
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        logging.info(f"Skipping URL {url}: already processed successfully.")
        return None

    try:
        logging.info(f"Processing URL: {url}")
        article = Article(url, language="en")  # Initialize an Article object with the URL
        article.download()  # Download the article's content
        article.parse()  # Parse the article to extract text and metadata

        # Save the article's title and text if content was successfully extracted
        if article.text.strip():
            with open(output_path, "w", encoding="utf-8") as file:
                file.write("Title:\n")
                file.write(article.title)
                file.write("\n\nArticle Text:\n")
                file.write(article.text)

            logging.info(f"Successfully extracted article for URL: {url}")
            return article.text
        else:
            logging.warning(f"No content found for URL: {url}")
            return None

    except Exception as e:
        logging.error(f"Error processing URL {url}: {e}")
        return None




def load_words(file_path):
    """
    Loads words from a file and returns them as a set for efficient lookup.

    Parameters:
        file_path (str): The path to the file containing words.

    Returns:
        set: A set of words read from the file.
    """
    with open(file_path, 'r') as file:
        return set(file.read().splitlines())




def load_stopwords(folder_path):
    """
    Loads stopwords from multiple files in a folder and combines them into a single set.

    Parameters:
        folder_path (str): The directory containing stopword files.

    Returns:
        set: A set of combined stopwords from all files.
    """
    stopwords = set()
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):  # Process only files, not directories
            stopwords.update(load_words(file_path))
    return stopwords




def clean_and_tokenize(text, stopwords):
    """
    Cleans the text by removing non-alphanumeric characters and tokenizes it.
    Filters out stopwords from the tokenized words.

    Parameters:
        text (str): The input text to clean and tokenize.
        stopwords (set): A set of stopwords to exclude.

    Returns:
        list: A list of cleaned, tokenized words without stopwords.
    """
    text = re.sub(r'\W', ' ', text)  # Remove non-alphanumeric characters
    text = re.sub(r'\d', ' ', text)  # Remove digits
    text = text.lower()  # Convert text to lowercase
    words = word_tokenize(text)  # Tokenize the text into words
    return [word for word in words if word not in stopwords]




def count_syllables(word):
    """
    Counts the number of syllables in a given word.

    Parameters:
        word (str): The input word.

    Returns:
        int: The number of syllables in the word.
    """
    word = word.lower()
    vowels = "aeiouy"
    syllables = 0

    # Check if the first character is a vowel
    if word[0] in vowels:
        syllables += 1

    # Count vowel transitions in the word
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            syllables += 1

    # Adjust for specific ending patterns
    if word.endswith("e"):
        syllables -= 1
    if word.endswith("es") or word.endswith("ed"):
        if len(word) > 2 and word[-3] not in vowels:
            syllables -= 1

    return max(syllables, 1)  # Ensure at least one syllable is returned




def count_personal_pronouns(text):
    """
    Counts the number of personal pronouns in the given text.

    Parameters:
        text (str): The input text.

    Returns:
        int: The count of personal pronouns.
    """
    pronoun_pattern = r'\b(I|we|my|ours|us)\b'
    matches = re.findall(pronoun_pattern, text, re.IGNORECASE)
    return len([match for match in matches if match.lower() != 'us'])




def calculate_metrics(text, stopwords, positive_words, negative_words):
    """
    Calculates various readability and sentiment metrics for the input text.

    Parameters:
        text (str): The input text to analyze.
        stopwords (set): A set of stopwords.
        positive_words (set): A set of positive sentiment words.
        negative_words (set): A set of negative sentiment words.

    Returns:
        dict: A dictionary of calculated metrics.
    """
    # Tokenize sentences and clean/tokenize words
    sentences = sent_tokenize(text)
    words = clean_and_tokenize(text, stopwords)

    num_sentences = len(sentences)  # Total number of sentences
    num_words = len(words)  # Total number of words
    num_chars = sum(len(word) for word in words)  # Total number of characters

    # Sentiment analysis
    positive_score = sum(1 for word in words if word in positive_words)
    negative_score = sum(1 for word in words if word in negative_words)
    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 1e-6)
    subjectivity_score = (positive_score + negative_score) / (num_words + 1e-6)

    # Readability metrics
    avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0
    complex_words = [word for word in words if count_syllables(word) > 2]
    num_complex_words = len(complex_words)
    pct_complex_words = (num_complex_words / num_words) * 100 if num_words > 0 else 0
    fog_index = 0.4 * (avg_sentence_length + pct_complex_words)

    syllable_counts = [count_syllables(word) for word in words]
    avg_syllables_per_word = sum(syllable_counts) / num_words if num_words > 0 else 0

    # Additional metrics
    personal_pronoun_count = count_personal_pronouns(text)
    avg_word_length = num_chars / num_words if num_words > 0 else 0

    # Return all metrics as a dictionary
    return {
        'Positive Score': positive_score,
        'Negative Score': negative_score,
        'Polarity Score': polarity_score,
        'Subjectivity Score': subjectivity_score,
        'Avg Sentence Length': avg_sentence_length,
        'Percentage of Complex Words': pct_complex_words,
        'Fog Index': fog_index,
        'Complex Word Count': num_complex_words,
        'Word Count': num_words,
        'Syllables Per Word': avg_syllables_per_word,
        'Personal Pronouns': personal_pronoun_count,
        'Avg Word Length': avg_word_length
    }
