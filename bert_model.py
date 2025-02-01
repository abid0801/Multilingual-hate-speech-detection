import pandas as pd
import torch
import logging
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from tqdm import tqdm

# Enable logging
logging.basicConfig(filename="training_log.txt", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("Starting training script...")

# Use a Smaller Model for Speed
MODEL_NAME = "distilbert-base-multilingual-cased"  # Faster alternative to mBERT
logging.info(f"Using model: {MODEL_NAME}")

# Load cleaned datasets
eng_df = pd.read_csv("final_cleaned_english_hate_speech.csv")
spa_df = pd.read_csv("final_cleaned_spanish_hate_speech.csv")

# Ensure language labels are added
eng_df["language"] = "en"
spa_df["language"] = "es"

# Merge datasets and save
merged_df = pd.concat([eng_df, spa_df], ignore_index=True)
merged_df.to_csv("merged_train.csv", index=False)

# Load train & test datasets
train_df = pd.read_csv("merged_train.csv")
test_df = pd.read_csv("merged_test.csv")  # Ensure test dataset exists

# Ensure no missing values
train_df["text"].fillna("", inplace=True)
test_df["text"].fillna("", inplace=True)

# Check dataset columns before processing
print("Columns in train_df:", train_df.columns)
print("Columns in test_df:", test_df.columns)
logging.info(f"Train Dataset columns: {train_df.columns}")
logging.info(f"Test Dataset columns: {test_df.columns}")

# Ensure required columns exist
for df, name in [(train_df, "Train"), (test_df, "Test")]:
    if "language" not in df.columns or "label" not in df.columns:
        raise KeyError(f"{name} dataset must contain 'language' and 'label' columns.")

# Maintain Language & Label Distribution when Sampling
def stratified_sample(df, sample_size):
    """Ensure balanced sampling by maintaining label and language distribution."""
    stratified_df = df.groupby(['language', 'label'], group_keys=False).apply(
        lambda x: x.sample(frac=sample_size/len(df), random_state=42)
    )
    return stratified_df

# Sample dataset while maintaining language & label distribution
train_sample_size = 10000  # Adjust if needed
test_sample_size = 2000  # Adjust if needed

train_df = stratified_sample(train_df, train_sample_size)
test_df = stratified_sample(test_df, test_sample_size)

# Verify Data Distribution
print("Train Data Distribution:\n", train_df.groupby(['language', 'label']).size())
print("Test Data Distribution:\n", test_df.groupby(['language', 'label']).size())
logging.info(f"Train Data Distribution: {train_df.groupby(['language', 'label']).size()}")
logging.info(f"Test Data Distribution: {test_df.groupby(['language', 'label']).size()}")

# Save the sampled dataset
train_df.to_csv("sampled_train.csv", index=False)
test_df.to_csv("sampled_test.csv", index=False)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Reduce Sequence Length for Faster Processing
def tokenize_function(examples):
    texts = [str(text) if text is not None else "" for text in examples["text"]]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128)  # Reduced from 256 to 128

# Convert Pandas DataFrame to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)  # Ensure test dataset exists

# Apply tokenization with tqdm progress tracking
print("Tokenizing training dataset...")
train_dataset = train_dataset.map(tokenize_function, batched=True, desc="Tokenizing Train Data", batch_size=500)

print("Tokenizing test dataset...")
test_dataset = test_dataset.map(tokenize_function, batched=True, desc="Tokenizing Test Data", batch_size=500)
logging.info("Tokenization completed.")

# Define model (Binary classification)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

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

# Ensure Model is Saved Even If Training Stops
if not os.path.exists("multilingual_hate_speech_model"):
    os.makedirs("multilingual_hate_speech_model")

# Reduce Batch Size & Train for Fewer Epochs
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

# Fix: Include test_dataset as eval_dataset
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,  # Ensure test dataset is provided
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Confirm Model Directory Exists Before Training
print("Checking model directory before training...")
print("Existing Model Files:", os.listdir("./multilingual_hate_speech_model") if os.path.exists("./multilingual_hate_speech_model") else "No model saved yet.")

# Train model with progress tracking
print("Training model...")
trainer.train()
logging.info("Training completed.")

# Save trained model
model.save_pretrained("multilingual_hate_speech_model")
tokenizer.save_pretrained("multilingual_hate_speech_model")
logging.info("Model saved successfully.")

print("Training complete! Model saved as multilingual_hate_speech_model")

# Save training details
results_text = f"""
Model: {MODEL_NAME}
Training Parameters:
   - Batch Size: {training_args.per_device_train_batch_size}
   - Epochs: {training_args.num_train_epochs}
   - Weight Decay: {training_args.weight_decay}
   - Best Model Metric: {training_args.metric_for_best_model}
"""

# Save training details without special characters
with open("training_details.txt", "w", encoding="utf-8") as f:
    f.write(results_text)

print("\nTraining details saved to 'training_details.txt'")
logging.info("Training details saved successfully.")
