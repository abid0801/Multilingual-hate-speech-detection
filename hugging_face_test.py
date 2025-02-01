import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Define your Hugging Face model name
MODEL_NAME = "abid0801/multilingual-hate-speech-model"

# Load the model and tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Function to make predictions
def predict_hate_speech(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Get softmax probabilities
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Get predicted class (0 = Non-Hate, 1 = Hate Speech)
    predicted_class = torch.argmax(probabilities).item()
    confidence = probabilities[0][predicted_class].item()

    # Class labels
    labels = ["Non-Hate Speech", "Hate Speech"]
    
    return labels[predicted_class], confidence

# Interactive testing
if __name__ == "__main__":
    while True:
        text = input("\nEnter a sentence to check for hate speech (or type 'exit' to quit): ")
        if text.lower() == "exit":
            break
        
        label, confidence = predict_hate_speech(text)
        print(f"\nPrediction: {label} (Confidence: {confidence:.2f})")
