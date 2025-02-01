import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Load trained model and tokenizer
model_path = "multilingual_hate_speech_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load test dataset instead of training dataset
test_df = pd.read_csv("sampled_test.csv")  # ✅ Now using the test dataset

# Ensure no missing values
test_df["text"].fillna("", inplace=True)

# Convert to Hugging Face dataset format
test_dataset = Dataset.from_pandas(test_df)

# Tokenization function
def tokenize_function(examples):
    texts = [str(text) if text is not None else "" for text in examples["text"]]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

# Tokenize the test dataset
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Define evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "precision": precision_score(labels, predictions),
        "recall": recall_score(labels, predictions),
    }

# Initialize Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Evaluate on test dataset
eval_results = trainer.evaluate(eval_dataset=test_dataset)

# Save evaluation results
eval_results_text = f"""
Test Dataset Evaluation:
Loss: {eval_results['eval_loss']}
Accuracy: {eval_results['eval_accuracy']}
F1 Score: {eval_results['eval_f1']}
Precision: {eval_results['eval_precision']}
Recall: {eval_results['eval_recall']}
"""

with open("test_evaluation.txt", "w", encoding="utf-8") as f:
    f.write(eval_results_text)

# Print results
print("\nTest Dataset Evaluation Complete!")
print(eval_results_text)
print("Results saved to 'test_evaluation.txt'.")
