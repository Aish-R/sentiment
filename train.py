

# ============================================================
# IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import torch
import re
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import Dataset

from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize

# ============================================================
# LOAD DATASET
# ============================================================

file_path = r"C:\Users\YourName\Desktop\balanced_4class_roberta_dataset.csv"

df = pd.read_csv(file_path)

print(df.head())

print("\nDATASET SHAPE :", df.shape)

# ============================================================
# REMOVE NULL VALUES
# ============================================================

df = df.dropna()

# ============================================================
# CLEAN TEXT
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

df["text"] = df["text"].apply(clean_text)

# ============================================================
# REMOVE SHORT TEXT
# ============================================================

df = df[df["text"].str.len() > 5]

# ============================================================
# LABELS
# ============================================================

label2id = {
    "Positive": 0,
    "Negative": 1,
    "Neutral": 2,
    "Fear_Surprise": 3
}

id2label = {
    0: "Positive",
    1: "Negative",
    2: "Neutral",
    3: "Fear_Surprise"
}

# ============================================================
# CONVERT LABELS TO NUMBERS
# ============================================================

df["label"] = df["label"].map(label2id)

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.15,
    stratify=df["label"],
    random_state=42
)

print("\nTRAIN SIZE :", len(train_texts))
print("VALIDATION SIZE :", len(val_texts))

# ============================================================
# LOAD TOKENIZER
# ============================================================

MODEL_NAME = "roberta-base"

tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

# ============================================================
# TOKENIZATION
# ============================================================

train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding=True,
    max_length=128
)

val_encodings = tokenizer(
    val_texts,
    truncation=True,
    padding=True,
    max_length=128
)

# ============================================================
# CREATE DATASETS
# ============================================================

train_dataset = Dataset.from_dict({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
    "labels": train_labels
})

val_dataset = Dataset.from_dict({
    "input_ids": val_encodings["input_ids"],
    "attention_mask": val_encodings["attention_mask"],
    "labels": val_labels
})

# ============================================================
# LOAD MODEL
# ============================================================

model = RobertaForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=4,
    id2label=id2label,
    label2id=label2id
)

# ============================================================
# METRICS FUNCTION
# ============================================================

def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predictions = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average='weighted'
    )

    acc = accuracy_score(labels, predictions)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(

    output_dir="./results",

    eval_strategy="epoch",

    save_strategy="epoch",

    learning_rate=1e-5,

    per_device_train_batch_size=16,

    per_device_eval_batch_size=16,

    num_train_epochs=7,

    weight_decay=0.01,

    logging_steps=50,

    load_best_model_at_end=True,

    metric_for_best_model="accuracy",

    greater_is_better=True,

    fp16=torch.cuda.is_available(),

    report_to="none"
)

# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    data_collator=DataCollatorWithPadding(tokenizer),

    compute_metrics=compute_metrics,

    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# ============================================================
# START TRAINING
# ============================================================

print("=" * 60)
print("TRAINING STARTED")
print("=" * 60)

trainer.train()

# ============================================================
# EVALUATE MODEL
# ============================================================

results = trainer.evaluate()

print("\nFINAL RESULTS")
print(results)

# ============================================================
# PREDICTIONS
# ============================================================

predictions = trainer.predict(val_dataset)

y_pred = np.argmax(predictions.predictions, axis=1)

y_true = val_labels

# ============================================================
# OVERALL METRICS
# ============================================================

accuracy = accuracy_score(y_true, y_pred)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_true,
    y_pred,
    average='weighted'
)

print("\n" + "=" * 60)
print("OVERALL PERFORMANCE")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(classification_report(
    y_true,
    y_pred,
    target_names=list(label2id.keys())
))

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=list(label2id.keys()),
    yticklabels=list(label2id.keys())
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

plt.show()

# ============================================================
# ROC CURVE
# ============================================================

y_score = torch.softmax(
    torch.tensor(predictions.predictions),
    dim=1
).numpy()

y_true_bin = label_binarize(y_true, classes=[0,1,2,3])

plt.figure(figsize=(8,6))

for i in range(4):

    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f'{id2label[i]} (AUC = {roc_auc:.2f})'
    )

plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.grid(True)

plt.show()

# ============================================================
# SAVE MODEL
# ============================================================

save_model_path = r"C:\Users\YourName\Desktop\best_roberta_emotion_model"

model.save_pretrained(save_model_path)

tokenizer.save_pretrained(save_model_path)

print("\nMODEL SAVED")

# ============================================================
# TRAINING LOGS
# ============================================================

logs = trainer.state.log_history

train_loss = []
eval_loss = []
eval_accuracy = []
epochs = []

for log in logs:

    if "loss" in log and "epoch" in log:
        train_loss.append(log["loss"])

    if "eval_loss" in log:
        eval_loss.append(log["eval_loss"])
        eval_accuracy.append(log["eval_accuracy"])
        epochs.append(log["epoch"])

# ============================================================
# TRAINING LOSS GRAPH
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(range(len(train_loss)), train_loss)

plt.xlabel("Steps")
plt.ylabel("Training Loss")

plt.title("Training Loss Graph")

plt.grid(True)

plt.show()

# ============================================================
# VALIDATION LOSS GRAPH
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(epochs, eval_loss)

plt.xlabel("Epochs")
plt.ylabel("Validation Loss")

plt.title("Validation Loss Graph")

plt.grid(True)

plt.show()

# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(epochs, eval_accuracy)

plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.title("Validation Accuracy Graph")

plt.grid(True)

plt.show()

# ============================================================
# CUSTOM PREDICTION
# ============================================================

def predict_emotion(text):

    text = clean_text(text)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():

        outputs = model(**inputs)

        pred = torch.argmax(outputs.logits, dim=1).item()

    return id2label[pred]

# ============================================================
# TEST SENTENCE
# ============================================================

sample = "I am very happy today"

prediction = predict_emotion(sample)

print("\nTEXT :", sample)

print("PREDICTED CLASS :", prediction)
