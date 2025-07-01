import streamlit as st
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from helpers import load_words, load_stopwords, extract_article, calculate_metrics
from config import STOPWORDS_FOLDER, POSITIVE_WORDS_FILE, NEGATIVE_WORDS_FILE

# Load resources during app startup
stopwords = load_stopwords(STOPWORDS_FOLDER)
positive_words = load_words(POSITIVE_WORDS_FILE)
negative_words = load_words(NEGATIVE_WORDS_FILE)

st.title("Fast Text Analysis and Sentiment Metrics Tool  with Parallel Processing")

# Upload Input File
uploaded_file = st.file_uploader("Upload Excel File with URL Data", type=["xlsx"])

if uploaded_file:
    # Process input file
    df = pd.read_excel(uploaded_file)
    st.write("Preview of Uploaded File:", df.head())

    # Choose output directory
    output_dir = st.text_input("Enter Output Directory", "output")

    if st.button("Start Processing"):
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        results = []
        failed_urls = []


        def process_url(row):
            """
            Process a single URL: extract the article and compute metrics.
            """
            url_id = row['URL_ID']
            url = row['URL']
            output_path = os.path.join(output_dir, f"{url_id}.txt")

            # Extract article content
            article_text = extract_article(url, output_path)

            if article_text:
                # Calculate metrics
                metrics = calculate_metrics(article_text, stopwords, positive_words, negative_words)
                metrics["URL_ID"] = url_id
                metrics["URL"] = url
                return metrics
            else:
                return {"URL_ID": url_id, "URL": url, "Error": "Failed to extract article"}

        with st.spinner("Processing URLs with Thread Pooling..."):
            # Use ThreadPoolExecutor to process URLs in parallel
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(process_url, row): row for _, row in df.iterrows()}
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if "Error" in result:
                            failed_urls.append(result["URL"])
                        else:
                            results.append(result)
                    except Exception as e:
                        st.error(f"Error processing URL: {e}")

        if results:
            # Save results
            metrics_df = pd.DataFrame(results)
            merged_df = pd.merge(df, metrics_df, on=['URL_ID', 'URL'], how='left')
            merged_df.to_csv("APP_extraction_to_output.csv", index=False)

            st.success(f"Processing completed! Results saved to output_file.")
            st.write("Preview of Results:", metrics_df.head())
        else:
            st.error("No data processed successfully.")

        if failed_urls:
            st.warning(f"Failed to process {len(failed_urls)} URLs.")
            st.write(failed_urls)
