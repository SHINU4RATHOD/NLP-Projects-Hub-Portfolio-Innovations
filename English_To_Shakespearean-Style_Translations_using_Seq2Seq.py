import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Function to load and parse the dataset from a text file
def load_dataset(file_path):
    """
    Loads parallel modern and Shakespearean English texts from a file.
    Each line in the file is expected to be in the format: modern_text||shakespearean_text
    Removes <sos> and <eos> tokens from Shakespearean text for consistency.
    
    Args:
        file_path (str): Path to the dataset file.
    
    Returns:
        tuple: Lists of modern_texts and shakespearean_texts.
    """
    modern_texts = []
    shakespearean_texts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                modern, shakespearean = line.strip().split('||') # Split line into modern and Shakespearean parts using '||' delimiter
                modern_texts.append(modern)
                shakespearean = shakespearean.replace('<sos> ', '').replace(' <eos>', '')  # Remove <sos> and <eos> tokens, as we'll add custom tokens later
                shakespearean_texts.append(shakespearean)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return [], []
    return modern_texts, shakespearean_texts

# Load the dataset from the specified file
file_path = 'shakespearean_dataset.txt'
modern_texts, shakespearean_texts = load_dataset(file_path)

# Verify that the dataset was loaded successfully
if not modern_texts or not shakespearean_texts:
    print("Failed to load dataset. Exiting.")
    exit()

# Add custom start and end tokens to Shakespearean texts for sequence generation
shakespearean_texts = ['<START> ' + text + ' <END>' for text in shakespearean_texts] 
'''The model needs to know where a sentence begins and ends when generating text.
<START> signals the decoder to start producing words, and <END> tells it to stop.'''

# Tokenize the texts to convert words to integer indices
modern_tokenizer = Tokenizer(filters='', oov_token='<OOV>')  # No filtering, use <OOV> for unknown words
shakespearean_tokenizer = Tokenizer(filters='', oov_token='<OOV>') # filters='' means we keep punctuation (e.g., "!" in "Hark!").       oov_token='<OOV>' assigns a number to unknown words (not in the training data).

# Build vocabulary for both modern and Shakespearean texts
modern_tokenizer.fit_on_texts(modern_texts)
shakespearean_tokenizer.fit_on_texts(shakespearean_texts)

# vocab_size is the number of unique words plus 1 (for padding).
modern_vocab_size = len(modern_tokenizer.word_index) + 1    # word_index is the dictionary (e.g., {"hello": 1, "how": 2, ...}).
shakespearean_vocab_size = len(shakespearean_tokenizer.word_index) + 1

# Convert text sequences to integer sequences
modern_sequences = modern_tokenizer.texts_to_sequences(modern_texts)
shakespearean_sequences = shakespearean_tokenizer.texts_to_sequences(shakespearean_texts)

# Determine maximum sequence lengths for padding  
max_modern_len = max(len(seq) for seq in modern_sequences) # finds the length of the longest modern sentence (in tokens).
max_shakespearean_len = max(len(seq) for seq in shakespearean_sequences)

# Pad sequences to ensure uniform length (post-padding with zeros)  Neural networks need inputs of the same size. ex: If max_modern_len=10, then [1, 2, 3] becomes [1, 2, 3, 0, 0, 0, 0, 0, 0, 0].
modern_padded = pad_sequences(modern_sequences, maxlen=max_modern_len, padding='post')
shakespearean_padded = pad_sequences(shakespearean_sequences, maxlen=max_shakespearean_len, padding='post')

# Prepare decoder input and target data for teacher forcing  (The decoder uses a technique called "teacher forcing," where it’s given the correct words to predict the next word.)
decoder_input_data = shakespearean_padded[:, :-1]  # Exclude last token for input
decoder_target_data = shakespearean_padded[:, 1:]  # Exclude first token for target


# Define the seq2seq model using Keras Functional API
# Encoder: Processes modern English input
encoder_inputs = Input(shape=(max_modern_len,), name='encoder_inputs')
encoder_embedding = Embedding(modern_vocab_size, 128, mask_zero=True, name='encoder_embedding')(encoder_inputs) #Turns each number into a 128-dimensional vector (like giving each word a unique description).    mask_zero=True: Ignores padded zeros so they don’t affect the learning.
encoder_lstm = LSTM(256, return_state=True, name='encoder_lstm')
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]  # Capture hidden and cell states for decoder

# Decoder: Generates Shakespearean English output
decoder_inputs = Input(shape=(max_shakespearean_len-1,), name='decoder_inputs')
decoder_embedding_layer = Embedding(shakespearean_vocab_size, 128, mask_zero=True, name='decoder_embedding')
decoder_embedding = decoder_embedding_layer(decoder_inputs)
decoder_lstm = LSTM(256, return_sequences=True, return_state=True, name='decoder_lstm')
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)  # Uses the encoder’s states to start generating words, outputting a sequence of predictions.
decoder_dense = Dense(shakespearean_vocab_size, activation='softmax', name='decoder_dense') # A layer that predicts the probability of each word in the Shakespearean
decoder_outputs = decoder_dense(decoder_outputs)

# Create the training model
'''Analogy:
Encoder: Like summarizing a book into a few key ideas.
Decoder: Like writing a new chapter based on that summary, using the teacher’s hints (teacher forcing).
The model is like a translator who learns by practicing with example translations.'''
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)  # Combines encoder and decoder, taking both inputs (encoder_inputs, decoder_inputs) and producing decoder_outputs.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) # Compile the model with loss function and optimizer

# Reshape target data to match expected shape for sparse categorical crossentropy
decoder_target_data = np.expand_dims(decoder_target_data, -1) # Adjusts the shape of decoder_target_data to match what the model expects.

# Train the model
model.fit([modern_padded, decoder_input_data], decoder_target_data, batch_size=16, epochs=100, validation_split=0.2, verbose=1)


####  Define inference models for translation
# Encoder inference model: Outputs states for a given input sequence
encoder_model = Model(encoder_inputs, encoder_states)  # Takes a modern sentence and outputs its states (summary). Same as the training encoder but only outputs state_h and state_c.


# Decoder inference model: Generates one token at a time
decoder_inputs_inf = Input(shape=(1,), name='decoder_inputs_inf')  # Input for single token
decoder_embedding_inf = decoder_embedding_layer(decoder_inputs_inf)
decoder_state_input_h = Input(shape=(256,), name='decoder_state_h')
decoder_state_input_c = Input(shape=(256,), name='decoder_state_c')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# Reuse the decoder LSTM and dense layers for consistency
decoder_outputs_inf, state_h_inf, state_c_inf = decoder_lstm(decoder_embedding_inf, initial_state=decoder_states_inputs)
decoder_states_inf = [state_h_inf, state_c_inf]
decoder_outputs_inf = decoder_dense(decoder_outputs_inf)
decoder_model = Model([decoder_inputs_inf] + decoder_states_inputs, [decoder_outputs_inf] + decoder_states_inf) # Create the decoder inference model

# Function to generate Shakespearean translation from modern English input
def generate_shakespearean(input_text):
    """
    Translates a modern English sentence to Shakespearean English using the trained seq2seq model.
    
    Args:
        input_text (str): Modern English sentence to translate.
    
    Returns:
        str: Shakespearean-style translation.
    """
    input_text = input_text.strip()
    # Convert input text to padded sequence
    input_seq = modern_tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_modern_len, padding='post')
    
    # Get encoder states for the input
    states_value = encoder_model.predict(input_seq, verbose=0)  # Runs the encoder to get the states (summary of the input).
    
    # Initialize decoder with <START> token
    target_seq = np.zeros((1, 1))
    try:
        target_seq[0, 0] = shakespearean_tokenizer.word_index['<start>']
    except KeyError:
        print("Error: <START> token not found in tokenizer.")
        return ""
    
    output_sentence = []
    stop_condition = False
    while not stop_condition:     # Generate tokens one by one until <END> or max length is reached
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)
        
        # Select the most likely token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = shakespearean_tokenizer.index_word.get(sampled_token_index, '')
        
        # Check for end condition
        if sampled_word == '<end>' or len(output_sentence) > max_shakespearean_len:
            stop_condition = True
        else:
            output_sentence.append(sampled_word)
        
        # Update target sequence and states for next iteration
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]
    
    return ' '.join(output_sentence)
'''Starts the decoder with the <START> token.
In a loop:
Feeds the current token and states to the decoder.
Picks the most likely next word (np.argmax).
Adds the word to the output unless it’s <end> or the sentence is too long.
Updates the token and states for the next iteration.'''




# Interactive function for user input and translation
def translate_to_shakespearean():
    """
    Prompts the user to input modern English sentences and prints their Shakespearean translations.
    Continues until the user enters 'quit'.
    """
    while True:
        user_input = input("Enter a modern English sentence (or 'quit' to exit): ").strip()
        if user_input.lower() == 'quit':
            print("Exiting program.")
            break
        try:
            translation = generate_shakespearean(user_input)
            print(f"Modern English: {user_input}")
            print(f"Shakespearean: {translation}\n")
        except Exception as e:
            print(f"Error processing input: {e}. Please try a simple sentence.")

# Start the translation interface
translate_to_shakespearean()





'''-------------------SUMMERY

How It All Fits Together
1. Data Preparation:
Load sentence pairs from the file.
Turn words into numbers and pad sequences.
Prepare decoder inputs and targets for training.

2. Training:
Build a seq2seq model with an encoder (reads modern English) and decoder (writes Shakespearean English).
Train it to predict the next word in a Shakespearean sentence, using modern sentences as context.

3. Inference:
Use separate encoder and decoder models to generate translations one word at a time.
The encoder summarizes the input, and the decoder builds the output.

4. User Interaction:
Let users input sentences and see the Shakespearean results.
'''