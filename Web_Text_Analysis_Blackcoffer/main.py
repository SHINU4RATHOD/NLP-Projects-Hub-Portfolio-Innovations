import os
import pandas as pd
# from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from helpers import load_words, load_stopwords, extract_article, calculate_metrics
from config import INPUT_FILE, STOPWORDS_FOLDER, POSITIVE_WORDS_FILE, NEGATIVE_WORDS_FILE, OUTPUT_DIR

def process_url(row, stopwords, positive_words, negative_words):
    """
    Process a single URL to extract article content, calculate text metrics, 
    and return the results.
    
    Args:
        row (dict): Row from the input DataFrame containing URL_ID and URL.
        stopwords (set): Set of stopwords to be excluded from text analysis.
        positive_words (set): Set of words with positive sentiment.
        negative_words (set): Set of words with negative sentiment.

    Returns:
        dict or None: A dictionary of calculated metrics if successful, otherwise None.
    """
    url_id = row['URL_ID']
    url = row['URL']
    output_path = os.path.join(OUTPUT_DIR, f"{url_id}.txt")
    article_text = None

    # Skip URLs that have already been processed
    if os.path.exists(output_path):
        print(f"Skipping URL {url}: already processed successfully.")
        with open(output_path, 'r', encoding='utf-8') as file:
            article_text = file.read()
    else:
        # Attempt to extract article content from the URL
        article_text = extract_article(url, output_path)

    if article_text:
        # Calculate text metrics for the extracted article
        metrics = calculate_metrics(article_text, stopwords, positive_words, negative_words)
        metrics["URL_ID"] = url_id
        metrics["URL"] = url
        return metrics
    else:
        # Return None if article extraction fails
        return None

def main():
    """
    Main function to process all URLs, calculate metrics, and save results.
    
    - Reads input data from an Excel file.
    - Loads stopwords and sentiment word lists.
    - Processes each URL concurrently to extract content and calculate metrics.
    - Saves the results to a CSV file and retries failed URLs.
    """
    # Load the input data containing URL_ID and URL
    df = pd.read_excel(INPUT_FILE)

    # Load stopwords and sentiment words
    stopwords = load_stopwords(STOPWORDS_FOLDER)
    positive_words = load_words(POSITIVE_WORDS_FILE)
    negative_words = load_words(NEGATIVE_WORDS_FILE)

    # Initialize lists for results and failed URLs
    results = []
    failed_urls = []

    # Use ThreadPoolExecutor for concurrent processing of URLs
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit each row of the DataFrame to the process_url function
        future_to_url = {executor.submit(process_url, row, stopwords, positive_words, negative_words): row for _, row in df.iterrows()}

        # Collect results or log failed URLs as tasks complete
        for future in concurrent.futures.as_completed(future_to_url):
            result = future.result()
            row = future_to_url[future]

            if result:
                results.append(result)
            else:
                failed_urls.append(row['URL'])

    # Save calculated metrics and merge with the input file
    if results:
        metrics_df = pd.DataFrame(results)

        # Merge calculated metrics with the input data
        merged_df = pd.merge(df, metrics_df, on=['URL_ID', 'URL'], how='left')

        # Save the merged DataFrame to a CSV file
        merged_df.to_csv("extraction_to_output.csv", index=False)
        print(f"Merged data saved to extraction_to_output.csv.")
    else:
        print("No metrics calculated. No data to save!")

    # Retry failed URLs
    if failed_urls:
        print(f"\nRetrying {len(failed_urls)} failed URLs...\n")
        retry_results = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Retry processing failed URLs
            future_to_url = {executor.submit(process_url, row, stopwords, positive_words, negative_words): row for row in df[df['URL'].isin(failed_urls)].to_dict('records')}

            for future in concurrent.futures.as_completed(future_to_url):
                result = future.result()
                if result:
                    retry_results.append(result)

        # Append retry results to the main results and save
        if retry_results:
            metrics_df = pd.DataFrame(results + retry_results)
            merged_df = pd.merge(df, metrics_df, on=['URL_ID', 'URL'], how='left')
            merged_df.to_csv("extraction_to_output.csv", index=False)
            print(f"Final merged data saved to extraction_to_output.csv.")
        else:
            print("No additional data could be processed from failed URLs.")

if __name__ == "__main__":
    main()
