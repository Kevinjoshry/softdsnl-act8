import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models
import numpy as np
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from dataset import load_goemotions_data

train_data, test_data = load_goemotions_data()
# Continue with preprocessing and training


# Load dataset
dataset = load_dataset("go_emotions", split="train[:5000]")  # smaller subset for demo
texts = dataset["text"]
labels = [l[0] if len(l) > 0 else 27 for l in dataset["labels"]]  # handle multi-label

# Encode labels
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)

# Tokenize text
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=50)

# Build model
model = models.Sequential([
    layers.Embedding(10000, 64, input_length=50),
    layers.Bidirectional(layers.LSTM(64)),
    layers.Dense(64, activation="relu"),
    layers.Dense(len(set(labels_encoded)), activation="softmax")
])

# Compile
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Train
model.fit(padded, labels_encoded, epochs=5, validation_split=0.2)

# Save model + tokenizer + encoder
model.save("emotion_model.h5")
import pickle
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)