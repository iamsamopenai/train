import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Function to load and preprocess the tokenized data
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.readlines()
    return [line.strip() for line in data]

# File path for the tokenized data
file_path = '/Users/armenmerikyan/Desktop/wd/python/llm/data/tokenized.txt'

# Load and preprocess tokenized sentences
tokenized_sentences = load_data(file_path)

# Reduce vocabulary size using subword tokenization
tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(tokenized_sentences)
tokenizer.num_words = 10000 # Limit vocabulary size
sequences = tokenizer.texts_to_sequences(tokenized_sentences)

# Split sequences into train and test
train_sequences, test_sequences = train_test_split(sequences, test_size=0.1)

# Pad sequences to ensure uniform length
max_sequence_length = 100 # Reduce sequence length
sequences_padded_train = pad_sequences(train_sequences, maxlen=max_sequence_length, padding='pre')
sequences_padded_test = pad_sequences(test_sequences, maxlen=max_sequence_length, padding='pre')

# Define the model architecture
embedding_dim = 50 # Reduce embedding dimension
rnn_units = 64 # Reduce number of units

# Use mixed precision for better memory utilization
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.num_words, embedding_dim),
    tf.keras.layers.SimpleRNN(rnn_units, return_sequences=True),
    tf.keras.layers.Dense(tokenizer.num_words, activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Set batch size
batch_size = 256

# Function to generate data batches
def data_generator(sequences_padded, batch_size):
    num_samples = len(sequences_padded)
    num_batches = (num_samples + batch_size - 1) // batch_size
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        batch_sequences = sequences_padded[start_idx:end_idx]
        yield batch_sequences, batch_sequences

# Train the model using fit method with generator
num_epochs = 10
num_batches = (len(sequences_padded_train) + batch_size - 1) // batch_size
model.fit(data_generator(sequences_padded_train, batch_size), steps_per_epoch=num_batches, epochs=num_epochs)

# Evaluate the model
loss, accuracy = model.evaluate(sequences_padded_test, sequences_padded_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Save the model using model.save() method
save_dir = '/Users/armenmerikyan/Desktop/wd/python/llm/models/'
os.makedirs(save_dir, exist_ok=True)
model.save(os.path.join(save_dir, 'language_model'))
