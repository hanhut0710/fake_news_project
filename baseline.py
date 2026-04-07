import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os


DATA_TRAIN_PATH = "data/split/train.csv"
DATA_TEST_PATH = "data/split/test.csv"

# Load data
df_train = pd.read_csv(DATA_TRAIN_PATH)
df_test = pd.read_csv(DATA_TEST_PATH)

df_train = df_train[["claim", "evidence", "label"]]
df_test = df_test[["claim", "evidence", "label"]]

# 🔥 Dùng claim ONLY (để baseline fair)
X_train = df_train["claim"] + " " + df_train["evidence"]
y_train = df_train["label"].astype(str)

X_test = df_test["claim"] + " " + df_test["evidence"]
y_test = df_test["label"].astype(str)

def train_baseline():
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1,2),
        stop_words="english"
    )
    X_train_vec = vectorizer.fit_transform(X_train)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train_vec, y_train)

    return model, vectorizer

def evaluate_baseline(model, vectorizer):
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)

    baseline_metrics = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist()
    }

    pd.DataFrame(baseline_metrics).to_csv("metrics/baseline_metrics.csv", index=False)

    return baseline_metrics


def save_baseline(model, vectorizer, out_dir="model/baseline"):
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "baseline_model.pkl")
    vec_path = os.path.join(out_dir, "vectorizer.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)
    return model_path, vec_path


def load_baseline(in_dir="model/baseline"):
    model_path = os.path.join(in_dir, "baseline_model.pkl")
    vec_path = os.path.join(in_dir, "vectorizer.pkl")
    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        raise FileNotFoundError(f"Baseline model or vectorizer not found in {in_dir}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


if __name__ == '__main__':
    print("Training baseline model and saving to model/baseline...")
    model_obj, vec = train_baseline()
    save_baseline(model_obj, vec)
    print("Saved baseline model.")





