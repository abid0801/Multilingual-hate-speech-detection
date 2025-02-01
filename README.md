# Multilingual Hate Speech Detection (English & Spanish)

This project trains and evaluates a **multilingual hate speech detection model** using **DistilBERT** and **Hugging Face Transformers**.  
It is fine-tuned on **English and Spanish datasets** to classify text as either **Hate Speech or Non-Hate Speech**.

---

## Project Overview

- Supports English and Spanish using a multilingual model.
- Fine-tuned `distilbert-base-multilingual-cased` for efficient training.
- Trained on a subset of 10,000 samples due to resource limitations.
- Training was performed for only 1 epoch to prevent excessive runtime.
- Datasets and model are stored on Google Drive (Not in GitHub).
- Pretrained model is available on Hugging Face for easy use.

---

## Missing Folders (Stored in Google Drive)

The following large files are **not included** in this GitHub repository but can be downloaded from Google Drive:

### Dataset Files (CSV)
- `english_train.csv`
- `english_test.csv`
- `spanish_train.csv`
- `spanish_test.csv`
- `merged_train.csv`
- `merged_test.csv`

### Trained Model Folder
- `multilingual_hate_speech_model/` (Contains the fine-tuned model with tokenizer)

---

## Download Dataset & Model

Since GitHub does not support large files, the dataset and trained model are hosted on Google Drive.

### Download from Google Drive

- [Download Dataset & Model (ZIP)](https://drive.google.com/file/d/1UHKSFIn6QrMsZsbNeUj_U_mXAGRpnYNX/view?usp=sharing)

### Download Using Python

To programmatically download the dataset and model, use `gdown`:

```python
import gdown

# Download Dataset & Model
file_url = "https://drive.google.com/uc?id=1UHKSFIn6QrMsZsbNeUj_U_mXAGRpnYNX"
gdown.download(file_url, "model_&_datasets.zip", quiet=False)
```

### Extract the Files

After downloading, extract the ZIP file:

```sh
unzip model_&_datasets.zip
```

Now the dataset and trained model are ready to use.

---

## Pretrained Model on Hugging Face

The trained model is publicly available on Hugging Face Model Hub:

[Hugging Face Model: abid0801/multilingual-hate-speech-model](https://huggingface.co/abid0801/multilingual-hate-speech-model)

You can directly load and use it in Python:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load Model & Tokenizer from Hugging Face
MODEL_NAME = "abid0801/multilingual-hate-speech-model"

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
```

---

## Installation

This project does not contain a `requirements.txt`, so install dependencies manually.

### Clone the repository

```sh
git clone https://github.com/abid0801/Multilingual-hate-speech-detection.git
cd multilingual-hate-speech-detection
```

### Install dependencies manually

```sh
pip install pandas torch transformers datasets scikit-learn tqdm huggingface_hub gdown
```

---

## Training the Model

Due to **resource constraints**, the model was trained on **only 10,000 samples** and for **one epoch**.

To train the model:

```sh
python bert_model.py
```

- Uses **DistilBERT** instead of mBERT for faster training.
- Processes and tokenizes **10,000 samples**.
- Runs for **one epoch** due to hardware limitations.
- Saves the trained model in `multilingual_hate_speech_model/`.

---

## Evaluating the Model

### Evaluate on Training Data

To check how well the model memorized the training data:

```sh
python evaluate_training_dataset.py
```

- Computes accuracy, F1-score, precision, and recall on `sampled_train.csv`.
- Saves evaluation results in `training_evaluation.txt`.

### Evaluate on Test Data (Recommended)

To evaluate the model on unseen data:

```sh
python evaluate_testing_dataset.py
```

- Computes accuracy, F1-score, precision, and recall on `sampled_test.csv`.
- Saves evaluation results in `test_evaluation.txt`.

---

## Model Evaluation Metrics

The model achieved the following performance metrics on the test dataset:

| Metric       | Score  |
|-------------|--------|
| Loss        | 0.4608 |
| Accuracy    | 77.31% |
| F1 Score    | 77.76% |
| Precision   | 76.49% |
| Recall      | 79.08% |

Future improvements include training for **more epochs** and using a **larger dataset**.

---

## Testing the Model from Hugging Face

You can directly use the trained model hosted on Hugging Face in Python.

### Load and Test the Model

Use this script to test any text input:

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Define your Hugging Face model name
MODEL_NAME = "abid0801/multilingual-hate-speech-model"

# Load the model and tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Function to make predictions
def predict_hate_speech(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities).item()
    confidence = probabilities[0][predicted_class].item()

    labels = ["Non-Hate Speech", "Hate Speech"]
    
    return labels[predicted_class], confidence

# Interactive testing
if __name__ == "__main__":
    while True:
        text = input("
Enter a sentence to check for hate speech (or type 'exit' to quit): ")
        if text.lower() == "exit":
            break
        
        label, confidence = predict_hate_speech(text)
        print(f"
Prediction: {label} (Confidence: {confidence:.2f})")
```

### Run the Script

Save it as `test_huggingface_model.py` and run:

```sh
python test_huggingface_model.py
```

---

## Project Folder Structure

```
/multilingual-hate-speech-detection
│── logs/                        # Logs for training and evaluation
│── results/                      # Training results
│── bert_model.py                 # Model training script
│── data_cleaning.py              # Script for cleaning datasets
│── evaluate_testing_dataset.py    # Evaluates model on the test dataset
│── evaluate_training_dataset.py   # Evaluates model on the training dataset
│── test_with_user_input.py        # Interactive script for user input testing
│── test_huggingface_model.py      # Calls the model from Hugging Face
│── training_details.txt           # Training parameters and details
│── test_evaluation.txt            # Final test evaluation results
│── training_evaluation.txt        # Final training evaluation results
│── training_log.txt               # Logs of training process
```

---
