from datasets import load_dataset
import pandas as pd

# Load GoEmotions dataset
dataset = load_dataset("go_emotions")

# Use 'train' split for training, 'test' for evaluation
train_data = dataset["train"]
test_data = dataset["test"]

print(train_data[0])  # sample data