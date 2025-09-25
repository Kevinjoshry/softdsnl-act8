import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models
import numpy as np
from datasets import load_from_disk
from sklearn.preprocessing import LabelEncoder
import pickle

# --- Configuration ---
NUM_WORDS = 10000
MAX_LEN = 50
EMBEDDING_DIM = 128  # Increased from 64
LSTM_UNITS = 128     # Increased from 64
NUM_EPOCHS = 15      # Increased from 5
# ---------------------

# Load dataset from disk
# Note: Ensure you have run the initial download of the GoEmotions dataset
dataset = load_from_disk("go_emotions_dataset") 

# Accessing the splits
train_data = dataset["train"]

# Prepare data
texts = train_data["text"]
# Handle multi-label: takes the first label if present, or index 27 (unannotated)
labels = [l[0] if len(l) > 0 else 27 for l in train_data["labels"]]

# Encode labels (maps 0-27 to 0-27)
encoder = LabelEncoder()
encoder.fit(list(range(28))) # Explicitly fit on all 28 expected classes (0-27)
labels_encoded = encoder.transform(labels)

# Tokenize text
tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

# Build model (Enhanced Architecture)
NUM_CLASSES = len(encoder.classes_) # Should be 28

model = models.Sequential([
    layers.Embedding(NUM_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),
    layers.Dropout(0.3), # Added Dropout for regularization
    # Stacked Bidirectional LSTMs for deeper feature extraction
    layers.Bidirectional(layers.LSTM(LSTM_UNITS, return_sequences=True)), 
    layers.Dropout(0.3),
    layers.Bidirectional(layers.LSTM(int(LSTM_UNITS/2))), # Second layer with fewer units
    layers.Dense(64, activation="relu"),
    layers.Dense(NUM_CLASSES, activation="softmax")
])

# Compile
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

print(model.summary())

# Train the model with increased epochs
print(f"\n--- Starting Training for {NUM_EPOCHS} Epochs ---")
model.fit(
    padded, 
    labels_encoded, 
    epochs=NUM_EPOCHS, 
    validation_split=0.1, # Use a validation split
    batch_size=64 # Increased batch size for faster training
)

# Save model + tokenizer + encoder
model.save("emotion_model.h5")
print("\nModel saved to emotion_model.h5")

# Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("Tokenizer saved to tokenizer.pkl")

# Save encoder
with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)
print("Encoder saved to encoder.pkl")