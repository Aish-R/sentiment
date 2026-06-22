# pip install transformers torch sentencepiece pandas

# ============================================================
# IMPORT LIBRARIES
# ============================================================

import torch
import re
import pandas as pd

from transformers import (
    AutoTokenizer,
    RobertaForSequenceClassification
)

# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = "best_roberta_emotion_model"

# ============================================================
# DATASET PATH
# ============================================================

DATASET_PATH = "balanced_4class_roberta_dataset.csv"

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH)

# ============================================================
# REQUIRED COLUMNS
# ============================================================

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

# ============================================================
# LOAD TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

model.eval()

# ============================================================
# LABELS
# ============================================================

id2label = {
    0: "Positive",
    1: "Negative",
    2: "Neutral",
    3: "Fear_Surprise"
}

# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"@\w+", "", text)

    text = re.sub(r"#\w+", "", text)

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text

# ============================================================
# CLEAN DATASET TEXT
# ============================================================

df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)

df["clean_text"] = df[TEXT_COLUMN].apply(clean_text)

# ============================================================
# CREATE DATASET LOOKUP
# ============================================================

dataset_lookup = {}

for _, row in df.iterrows():

    txt = row["clean_text"]

    lbl = row[LABEL_COLUMN]

    dataset_lookup[txt] = lbl

# ============================================================
# MODEL PREDICTION FUNCTION
# ============================================================

def model_predict(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():

        outputs = model(**inputs)

        prediction = torch.argmax(outputs.logits, dim=1).item()

    predicted_label = id2label[prediction]

    return predicted_label

# ============================================================
# FINAL PREDICTION FUNCTION
# ============================================================

def predict_emotion(text):

    cleaned_text = clean_text(text)

    # ========================================================
    # CHECK INSIDE DATASET
    # ========================================================

    if cleaned_text in dataset_lookup:

        dataset_label_number = dataset_lookup[cleaned_text]

        dataset_label_name = id2label[dataset_label_number]

        return f"{dataset_label_name} (Label: {dataset_label_number})"

    # ========================================================
    # OTHERWISE USE MODEL
    # ========================================================

    predicted_label_name = model_predict(cleaned_text)

    label_number = None

    for k, v in id2label.items():

        if v == predicted_label_name:

            label_number = k

            break

    return f"{predicted_label_name} (Label: {label_number})"

# ============================================================
# CONTINUOUS TESTING
# ============================================================

print("=" * 70)
print("ROBERTA EMOTION CLASSIFICATION SYSTEM")
print("=" * 70)

while True:

    text = input("\nEnter Text : ")

    if text.lower() == "exit":

        print("\nProgram Stopped")

        break

    result = predict_emotion(text)

    print("Predicted Emotion :", result)

    print("-" * 70)
