# Web Text Analysis and Sentiment Metrics Tool

## Overview
This project is a **Text Analysis and Sentiment Metrics Tool** designed to extract article content from a list of URLs, perform sentiment and readability analysis, and generate metrics such as positive/negative scores, polarity, subjectivity, and readability indices (e.g., Fog Index). The tool supports both batch processing via a command-line script (`main.py`) and interactive processing through a Streamlit web application (`app.py`). It uses parallel processing with `ThreadPoolExecutor` to efficiently handle multiple URLs.

The project processes URLs listed in an Excel file (`Input.xlsx`), extracts article text using the `newspaper3k` library, and calculates metrics based on positive/negative word lists and stopwords. Results are saved as CSV files, and extracted articles are stored as text files.

## Project Structure
The repository is organized as follows:
- **Main_Code/**: Contains the core scripts and data for the project.
  - `app.py`: Streamlit web application for interactive URL processing and metrics visualization.
  - `config.py`: Configuration file defining file paths and directories.
  - `helpers.py`: Utility functions for article extraction, text cleaning, tokenization, and metrics calculation.
  - `main.py`: Command-line script for batch processing URLs and generating metrics.
  - `requirements.txt`: List of Python dependencies required for the project.
  - `Input.xlsx`: Input Excel file containing `URL_ID` and `URL` columns for articles to process.
  - `03_Project Documentation Instructions/`: Directory containing project documentation (not tracked in Git).
  - `01_extracted_articles/`: Directory for storing extracted article text files (not tracked in Git).
  - `02_Output/`: Directory for output files, including `APP_extraction_to_output.csv` (not tracked in Git).
- **Ass_Material/**: Contains reference materials like stopwords and sentiment word lists (not tracked in Git).
- `.gitignore`: Specifies directories and files to exclude from Git (e.g., `Ass_Material`, `01_extracted_articles`, `02_Output`).

## Prerequisites
- **Python 3.8+**: Ensure Python is installed on your system.
- **Git**: For cloning the repository.
- **External Resources** (not included in the repository, must be provided):
  - Stopwords files in a folder (e.g., `StopWords-20240729T154945Z-001/StopWords/`).
  - Positive and negative word lists (e.g., `MasterDictionary-20240729T154927Z-001/MasterDictionary/positive-words.txt`, `negative-words.txt`).

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/SHINU4RATHOD/NLP-Projects-Hub-Portfolio-Innovations.git
   cd Web_Text_Analysis_Blackcoffer
   ```

2. **Install Dependencies**:
   Install the required Python packages listed in `requirements.txt`:
   ```bash
   pip install -r Main_Code/requirements.txt
   ```
   **Note**: The `nltk` library requires downloading additional data. Run the following in Python:
   ```python
   import nltk
   nltk.download('punkt')
   ```

3. **Prepare External Resources**:
   - Place the stopwords folder (e.g., `StopWords-20240729T154945Z-001/StopWords/`) and sentiment word files (e.g., `MasterDictionary-20240729T154927Z-001/MasterDictionary/positive-words.txt`, `negative-words.txt`) in a directory accessible to the project.
   - Update `config.py` to point to the correct paths for `STOPWORDS_FOLDER`, `POSITIVE_WORDS_FILE`, and `NEGATIVE_WORDS_FILE`. For example:
     ```python
     STOPWORDS_FOLDER = os.path.join(DATA_DIR, r'path/to/StopWords')
     POSITIVE_WORDS_FILE = os.path.join(DATA_DIR, r'path/to/positive-words.txt')
     NEGATIVE_WORDS_FILE = os.path.join(DATA_DIR, r'path/to/negative-words.txt')
     ```

4. **Prepare Input File**:
   Ensure `Input.xlsx` is in the `Main_Code` directory with columns `URL_ID` and `URL`. The provided `Input.xlsx` contains 147 URLs for processing.

## Usage
The project offers two ways to process URLs: a command-line script (`main.py`) for batch processing and a Streamlit web app (`app.py`) for interactive use.

### 1. Command-Line Script (`main.py`)
- **Purpose**: Batch process URLs from `Input.xlsx`, extract article content, calculate metrics, and save results to `extraction_to_output.csv`.
- **Run**:
  ```bash
  cd Main_Code
  python main.py
  ```
- **Output**:
  - Extracted articles are saved as `.txt` files in `Main_Code/01_extracted_articles/`.
  - Metrics are saved in `extraction_to_output.csv` in the `Main_Code` directory.
- **Notes**:
  - The script uses parallel processing with `ThreadPoolExecutor` for efficiency.
  - Failed URLs are retried once, and results are merged with the input data.

### 2. Streamlit Web App (`app.py`)
- **Purpose**: Interactive interface to upload an Excel file, process URLs, and view results.
- **Run**:
  ```bash
  cd Main_Code
  streamlit run app.py
  ```
- **Steps**:
  1. Open the Streamlit app in your browser (typically at `http://localhost:8501`).
  2. Upload an Excel file with `URL_ID` and `URL` columns.
  3. Specify an output directory for extracted articles (default: `output`).
  4. Click **Start Processing** to extract articles and calculate metrics.
- **Output**:
  - Extracted articles are saved as `.txt` files in the specified output directory.
  - Results are saved in `APP_extraction_to_output.csv`.
  - A preview of results is displayed in the Streamlit interface.
- **Notes**:
  - Requires the same external resources as `main.py`.
  - Failed URLs are listed in the interface.

## Metrics Calculated
The tool calculates the following metrics for each article:
- **Positive Score**: Count of positive words.
- **Negative Score**: Count of negative words.
- **Polarity Score**: `(Positive Score - Negative Score) / (Positive Score + Negative Score + 1e-6)`.
- **Subjectivity Score**: `(Positive Score + Negative Score) / (Word Count + 1e-6)`.
- **Average Sentence Length**: Words per sentence.
- **Percentage of Complex Words**: Percentage of words with more than two syllables.
- **Fog Index**: `0.4 * (Average Sentence Length + Percentage of Complex Words)`.
- **Complex Word Count**: Number of words with more than two syllables.
- **Word Count**: Total number of cleaned words (excluding stopwords).
- **Syllables Per Word**: Average syllables per word.
- **Personal Pronouns**: Count of pronouns like "I," "we," "my," "ours" (excluding "us").
- **Average Word Length**: Average characters per word.

## Dependencies
The project requires the following Python packages (listed in `requirements.txt`):
- `pandas`: Data manipulation and Excel file handling.
- `tqdm`: Progress bar for iterables.
- `openpyxl`: Reading and writing Excel files.
- `newspaper3k`: Web scraping and article extraction.
- `nltk`: Text processing (tokenization, sentence splitting).
- `streamlit`: Web application framework for `app.py`.

Install them using:
```bash
pip install -r Main_Code/requirements.txt
```

Additionally, `nltk` requires:
```python
import nltk
nltk.download('punkt')
```

Standard library modules (`os`, `re`, `logging`) are included with Python and do not require installation.

## Notes
- **External Resources**: The stopwords and sentiment word files are not included in the repository (ignored via `.gitignore`). Users must provide these files and update `config.py` with the correct paths.
- **Output Directories**: The `01_extracted_articles` and `02_Output` directories are ignored by `.gitignore` to avoid uploading large files. Ensure these directories exist or are created during execution.
- **Error Handling**: Both `main.py` and `app.py` handle failed URLs gracefully, with retries in `main.py` and error reporting in `app.py`.
- **Performance**: The use of `ThreadPoolExecutor` ensures efficient parallel processing of URLs.

## Troubleshooting
- **Missing Resources**: If stopwords or sentiment word files are missing, update `config.py` with the correct paths.
- **NLTK Errors**: Ensure `nltk.download('punkt')` is run before executing the scripts.
- **URL Extraction Failures**: Some URLs may fail due to network issues or unsupported formats. Check logs in `main.py` or the Streamlit interface for details.
- **Streamlit Issues**: Ensure `streamlit` is installed and run `streamlit run app.py` from the `Main_Code` directory.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue on the [GitHub repository](https://github.com/SHINU4RATHOD/NLP-Projects-Hub-Portfolio-Innovations).

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.