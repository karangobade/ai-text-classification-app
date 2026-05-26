import os
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_URL = "https://huggingface.co/distilbert-base-uncased/resolve/main/pytorch_model.bin"
MODEL_FILE = "pytorch_model.bin"


if not os.path.exists(MODEL_FILE):
    print("Downloading model file...")
    
    response = requests.get(MODEL_URL, stream=True)

    with open(MODEL_FILE, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)

    print("Download completed!")
else:
    print("Model file already exists.")

# Load tokenizer and model
print("Loading tokenizer and model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

print("Model loaded successfully!\n")

# Example sentences
sentences = [
    "I love this product!",
    "This movie was terrible.",
    "The service was amazing.",
    "I am very disappointed."
]


user_input = input("Enter your own sentence (or press Enter to skip): ")

if user_input.strip():
    sentences.append(user_input)

# Sentiment prediction
labels = ["NEGATIVE", "POSITIVE"]

for sentence in sentences:
    inputs = tokenizer(sentence, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()

    print(f"\nSentence: {sentence}")
    print(f"Predicted Sentiment: {labels[prediction]}")
