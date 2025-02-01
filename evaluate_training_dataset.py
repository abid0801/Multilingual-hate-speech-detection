import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

model_path = "multilingual_hate_speech_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

train_df = pd.read_csv("sampled_train.csv")

train_df["text"].fillna("", inplace=True)

train_dataset = Dataset.from_pandas(train_df)

def tokenize_function(examples):
    texts = [str(text) if text is not None else "" for text in examples["text"]]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "precision": precision_score(labels, predictions),
        "recall": recall_score(labels, predictions),
    }

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

eval_results = trainer.evaluate(eval_dataset=train_dataset)

eval_results_text = f"""
Training Dataset Evaluation:
Loss: {eval_results['eval_loss']}
Accuracy: {eval_results['eval_accuracy']}
F1 Score: {eval_results['eval_f1']}
Precision: {eval_results['eval_precision']}
Recall: {eval_results['eval_recall']}
"""

with open("training_evaluation.txt", "w", encoding="utf-8") as f:
    f.write(eval_results_text)

print("\nTraining Dataset Evaluation Complete!")
print(eval_results_text)
print("Results saved to 'training_evaluation.txt'.")
