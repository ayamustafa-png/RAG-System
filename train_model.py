import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# --- 1. Configuration & Data Ingestion ---
DATA_PATH = "data/arxiv-metadata-oai-snapshot.json"
MODEL_OUTPUT_PATH = "academic_classifier_model.h5"
TOKENIZER_OUTPUT_PATH = "tokenizer.pickle"
SAMPLE_SIZE = 40000
TOP_CATEGORIES = ['cs', 'math', 'physics', 'astro-ph']

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at execution path: {DATA_PATH}")

print(f"[*] Ingesting the first {SAMPLE_SIZE} records from arXiv dataset...")
raw_data = []
with open(DATA_PATH, "r", encoding="utf-8") as file:
    for idx, line in enumerate(file):
        if idx >= SAMPLE_SIZE:
            break
        raw_data.append(json.loads(line))

df = pd.DataFrame(raw_data)[['abstract', 'categories']]

# --- 2. Label Preprocessing & Encoding ---
# Extract the primary high-level taxonomic category
df['main_category'] = df['categories'].apply(lambda x: x.split()[0].split('.')[0])

# Filter dataset to balance across target classes
df = df[df['main_category'].isin(TOP_CATEGORIES)].reset_index(drop=True)

# Map text categories to numerical integer labels
category_mapping = {category: idx for idx, category in enumerate(TOP_CATEGORIES)}
df['label'] = df['main_category'].map(category_mapping)

print(f"[+] Dataset filtered successfully. Target instances: {len(df)}")

# --- 3. Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    df['abstract'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# --- 4. Text Tokenization & Sequence Padding ---
VOCAB_SIZE = 15000
MAX_SEQUENCE_LENGTH = 200

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

# --- 5. Model Architecture Specifications (LSTM) ---
EMBEDDING_DIM = 64

model = Sequential([
    Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(len(TOP_CATEGORIES), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- 6. Execution & Model Serialization ---
print("\n[!] Commencing Deep Learning Model Training Pipeline...")
model.fit(
    X_train_padded, y_train, 
    epochs=5, 
    validation_data=(X_test_padded, y_test), 
    batch_size=64
)

# Export structural configurations and learned weights
model.save(MODEL_OUTPUT_PATH)
with open(TOKENIZER_OUTPUT_PATH, "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(f"\n[+] Compilation successful. Deliverables saved: '{MODEL_OUTPUT_PATH}' and '{TOKENIZER_OUTPUT_PATH}'.")