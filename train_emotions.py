import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models
import numpy as np
from datasets import load_from_disk
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset from disk
dataset = load_from_disk("go_emotions_dataset")  # Assuming the dataset is saved locally

# Accessing the splits
train_data = dataset["train"]
validation_data = dataset["validation"]
test_data = dataset["test"]

# Prepare data (You can choose to work with any split - here we use "train")
texts = train_data["text"]
labels = [l[0] if len(l) > 0 else 27 for l in train_data["labels"]]  # handle multi-label

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

# Train the model
model.fit(padded, labels_encoded, epochs=5, validation_split=0.2)

# Save model + tokenizer + encoder
model.save("emotion_model.h5")

# Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Save encoder
with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)