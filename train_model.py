from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
df = pd.read_csv("data/processed/evidence_retrieved_live.csv")

df_train, temp_df = train_test_split(df, test_size=0.2, random_state=42)
df_test, df_val = train_test_split(temp_df, test_size=0.5, random_state=42)

label_map = {"real": 0, "fake": 1}
df_train["label"] = df_train["label"].map(label_map)
df_test["label"] = df_test["label"].map(label_map)
df_val["label"] = df_val["label"].map(label_map)

train_dataset = Dataset.from_pandas(df_train[["claim", "evidence", "label"]])
val_dataset = Dataset.from_pandas(df_val[["claim", "evidence", "label"]])
test_dataset = Dataset.from_pandas(df_test[["claim", "evidence", "label"]])


tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def tokenize(example):
    # Tokenize as a pair (claim, evidence) so model sees both inputs
    return tokenizer(
        example["claim"],
        example.get("evidence", ""),
        truncation=True,
        padding="max_length",
        max_length=256
    )

train_dataset = train_dataset.map(tokenize)
val_dataset = val_dataset.map(tokenize)
test_dataset = test_dataset.map(tokenize)

model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=2
)

training_args = TrainingArguments(
    output_dir="model/",
    
    eval_strategy = "steps",
    eval_steps = 500,

    save_strategy = "steps",
    save_steps = 500,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,

)

trainer.train()

metric = trainer.evaluate(test_dataset)

df_metric = pd.DataFrame([metric])
df_metric.to_csv("model/test_metrics.csv", index=False)

model.save_pretrained("model_evidence_claim/")
tokenizer.save_pretrained("model_evidence_claim/")