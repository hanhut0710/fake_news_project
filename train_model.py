from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import numpy as np

# Load data
df_train = pd.read_csv('data/split/train.csv')
df_val = pd.read_csv('data/split/val.csv')
df_test = pd.read_csv('data/split/test.csv')

train_dataset = Dataset.from_pandas(df_train[["claim", "evidence", "label"]])
val_dataset = Dataset.from_pandas(df_val[["claim", "evidence", "label"]])
test_dataset = Dataset.from_pandas(df_test[["claim", "evidence", "label"]])


tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def tokenize(batch):
    # Batched tokenization for performance and consistent output shapes
    enc = tokenizer(
        batch["claim"],
        batch['evidence'],
        truncation=True,
        padding="max_length",
        max_length=256,
    )
    enc["labels"] = batch["label"]
    return enc

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.remove_columns(["claim", "evidence", "label"])
val_dataset = val_dataset.remove_columns(["claim", "evidence", "label"])
test_dataset = test_dataset.remove_columns(["claim", "evidence", "label"])

print("Sample tokenized input:", train_dataset[0])

# Let datasets return PyTorch tensors for Trainer
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=2
)

training_args = TrainingArguments(
    output_dir="model/model_bert/",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    fp16=True,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

metric = trainer.evaluate(test_dataset)

df_metric = pd.DataFrame([metric])
df_metric.to_csv("model_evidence_claim/test_metrics.csv", index=False)

# Save final model/tokenizer to model_bert
model.save_pretrained("model/model_bert/")
tokenizer.save_pretrained("model/model_bert/")