import argparse
import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import numpy as np

from reasoning import apply_reasoning


LABEL_MAP = {0: "real", 1: "fake"}
ML_LABEL_NAME = {0: "SUPPORTS", 1: "REFUTES"}


def evaluate(model_dir: str, test_csv: str, save_csv: str, batch_size: int = 32, use_reasoning: bool = True, abstain_as_incorrect: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model/tokenizer from {model_dir} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    df = pd.read_csv(test_csv)
    # Expect columns: claim, evidence, label (0/1)
    if 'label' not in df.columns:
        raise RuntimeError("Test CSV must contain a 'label' column with 0/1 values")

    claims = df['claim'].fillna("").astype(str).tolist()
    evidences = df['evidence'].fillna("").astype(str).tolist()
    gold = df['label'].astype(int).tolist()

    preds = []
    probs = []
    final_labels = []
    reasons = []

    for i in range(0, len(claims), batch_size):
        batch_claims = claims[i:i+batch_size]
        batch_evidences = evidences[i:i+batch_size]

        enc = tokenizer(batch_claims, batch_evidences, truncation=True, padding=True, max_length=256, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            logits = out.logits
            probs_batch = torch.softmax(logits, dim=1).cpu().numpy()
            pred_idx = np.argmax(probs_batch, axis=1).tolist()

        preds.extend(pred_idx)
        probs.extend([float(p.max()) for p in probs_batch])

        if use_reasoning:
            for c, e, p_idx, p_conf in zip(batch_claims, batch_evidences, pred_idx, probs_batch.max(axis=1)):
                ml_label_name = ML_LABEL_NAME.get(p_idx, "SUPPORTS")
                final_label, reason = apply_reasoning(c, e, ml_label_name, float(p_conf))
                # map final_label back to 0/1/-1
                if final_label == "SUPPORTS":
                    final_labels.append(0)
                elif final_label == "REFUTES":
                    final_labels.append(1)
                else:
                    final_labels.append(-1)
                reasons.append(reason)
        else:
            final_labels.extend([-1] * len(pred_idx))
            reasons.extend([""] * len(pred_idx))

    # ML metrics
    acc = accuracy_score(gold, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(gold, preds, average='binary', zero_division=0)
    cm = confusion_matrix(gold, preds)

    ml_metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm.tolist()
    }

    # Reasoning/pipeline metrics: handle abstains
    pipeline_metrics = None
    if use_reasoning:
        # treat -1 as abstain; if abstain_as_incorrect -> map to wrong label
        eval_preds = []
        for fl, mlp in zip(final_labels, preds):
            if fl == -1:
                if abstain_as_incorrect:
                    # count as wrong: pick a label different from gold later via comparison
                    eval_preds.append(1 - mlp)  # intentionally mark opposite to ML prediction
                else:
                    eval_preds.append(mlp)
            else:
                eval_preds.append(fl)

        acc_p = accuracy_score(gold, eval_preds)
        prec_p, rec_p, f1_p, _ = precision_recall_fscore_support(gold, eval_preds, average='binary', zero_division=0)
        cm_p = confusion_matrix(gold, eval_preds)
        pipeline_metrics = {
            'accuracy': acc_p,
            'precision': prec_p,
            'recall': rec_p,
            'f1': f1_p,
            'confusion_matrix': cm_p.tolist()
        }

    # Save per-sample outputs
    out_df = pd.DataFrame({
        'claim': claims,
        'evidence': evidences,
        'gold': gold,
        'pred': preds,
        'pred_conf': probs,
        'final_label': final_labels,
        'reason': reasons,
    })
    out_predictions_csv = os.path.splitext(save_csv)[0] + '_predictions.csv'
    out_df.to_csv(out_predictions_csv, index=False)

    # Save summary metrics
    results = {'ml_metrics': ml_metrics, 'pipeline_metrics': pipeline_metrics}
    # flatten for CSV
    rows = []
    ml_row = {'mode': 'ml', **{k: v for k, v in ml_metrics.items() if k != 'confusion_matrix'}}
    rows.append(ml_row)
    if pipeline_metrics is not None:
        p_row = {'mode': 'pipeline', **{k: v for k, v in pipeline_metrics.items() if k != 'confusion_matrix'}}
        rows.append(p_row)

    pd.DataFrame(rows).to_csv(save_csv, index=False)

    print("Evaluation finished.")
    print("ML metrics:")
    print(ml_metrics)
    if pipeline_metrics is not None:
        print("Pipeline metrics:")
        print(pipeline_metrics)
    print(f"Per-sample predictions saved to: {out_predictions_csv}")
    print(f"Summary metrics saved to: {save_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='model/model_bert', help='Model directory')
    parser.add_argument('--test_csv', default='data/split/test.csv', help='Test CSV file (claim,evidence,label)')
    parser.add_argument('--save_csv', default='model/test_metrics.csv', help='Where to save metrics summary')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--no_reasoning', action='store_true', help='Disable reasoning pipeline evaluation')
    parser.add_argument('--count_abstain_as_incorrect', action='store_true', help='Treat NOT ENOUGH INFO as incorrect')

    args = parser.parse_args()

    evaluate(
        model_dir=args.model_dir,
        test_csv=args.test_csv,
        save_csv=args.save_csv,
        batch_size=args.batch_size,
        use_reasoning=not args.no_reasoning,
        abstain_as_incorrect=args.count_abstain_as_incorrect,
    )


if __name__ == '__main__':
    main()
