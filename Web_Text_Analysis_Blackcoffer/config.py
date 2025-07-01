import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR)

# # Define the output directory for extracted articles
OUTPUT_DIR = os.path.join(BASE_DIR, 'extracted_articles')
# Ensure the output directory exists; create it if it doesn't
os.makedirs(OUTPUT_DIR, exist_ok=True)


# File paths for input data and resources
INPUT_FILE = os.path.join(DATA_DIR, r'Input.xlsx')
STOPWORDS_FOLDER = os.path.join(DATA_DIR, r'D:\Desktop\internship assignment\02_Blackcoffer_ass\StopWords-20240729T154945Z-001')
POSITIVE_WORDS_FILE = os.path.join(DATA_DIR, r'D:\Desktop\internship assignment\02_Blackcoffer_ass\MasterDictionary-20240729T154927Z-001\MasterDictionary\positive-words.txt')
NEGATIVE_WORDS_FILE = os.path.join(DATA_DIR, r'D:\Desktop\internship assignment\02_Blackcoffer_ass\MasterDictionary-20240729T154927Z-001\MasterDictionary\negative-words.txt')

