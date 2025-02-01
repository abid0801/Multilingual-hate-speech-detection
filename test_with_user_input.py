import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the trained model and tokenizer
model_path = "multilingual_hate_speech_model"  # Ensure this folder contains the trained model
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Function to predict hate speech
def predict_hate_speech(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits).item()
    return "Hate Speech" if prediction == 1 else "Non-Hate Speech"

# Continuous loop to ask user for input
print("\nHate Speech Detection System (Type 'exit' to stop)")
while True:
    user_input = input("Enter text to analyze: ")
    if user_input.lower() == "exit":
        print("Exiting program...")
        break
    result = predict_hate_speech(user_input)
    print(f"Prediction: {result}\n")
