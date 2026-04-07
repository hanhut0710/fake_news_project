import ast
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_cm_entry(s):
    try:
        val = ast.literal_eval(s)
        return list(val)
    except Exception:
        # fallback: try to parse numbers separated by non-digits
        nums = [int(x) for x in ''.join(ch if (ch.isdigit() or ch == '-') else ' ' for ch in s).split()]
        return nums


def plot_metrics(df, out_dir: Path):
    metrics = [c for c in ["accuracy", "precision", "recall", "f1"] if c in df.columns]
    if not metrics:
        return None

    plt.figure(figsize=(8, 4))
    df[metrics].plot(kind="bar")
    plt.title("Metrics per run")
    plt.xlabel("run")
    plt.tight_layout()
    out_path = out_dir / "metrics_over_runs.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_confusion_matrices(df, out_dir: Path):
    cms = [parse_cm_entry(x) for x in df['confusion_matrix'].astype(str).tolist()]

    saved = []
    # Case 1: any entry with 4 values -> treat each as 2x2
    for i, cm in enumerate(cms):
        if len(cm) == 4:
            arr = np.array(cm).reshape((2, 2))
            plt.figure(figsize=(4, 3))
            sns.heatmap(arr, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Confusion matrix (row {i})")
            p = out_dir / f"confusion_matrix_row{i}.png"
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            saved.append(p)

    if saved:
        return saved

    # Case 2: entries have length 2 and there are at least 2 rows -> stack rows into 2x2
    if all(len(cm) == 2 for cm in cms) and len(cms) >= 2:
        arr = np.vstack([cms[0], cms[1]])
        plt.figure(figsize=(4, 3))
        sns.heatmap(arr, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion matrix")
        p = out_dir / "confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(p)
        plt.close()
        return [p]

    # Case 3: can't interpret -> return empty
    return []


def main():
    base = Path(__file__).resolve().parents[0]
    csv_path = base / "baseline_metrics.csv"
    out_dir = base / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    metrics_img = plot_metrics(df, out_dir)
    cms = []
    if 'confusion_matrix' in df.columns:
        cms = plot_confusion_matrices(df, out_dir)

    print("Saved:")
    if metrics_img:
        print(f"- {metrics_img}")
    for c in cms:
        print(f"- {c}")


if __name__ == '__main__':
    main()
