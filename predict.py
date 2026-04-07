import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from reasoning import apply_reasoning

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

labels_map = {
    1: "REFUTES",
    0: "SUPPORTS",
}
def predict_bert(claim, evidence, model, tokenizer, use_reasoning: bool = True):
    # Tokenize as a pair to match training (claim, evidence)
    inputs = tokenizer(
        claim,
        evidence,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    model_label = labels_map[pred.item()]
    confidence = confidence.item()

    if use_reasoning:
        final_label, reason = apply_reasoning(claim, evidence, model_label, confidence)
    else:
        final_label = model_label
        reason = ""

    return {
        "claim": claim,
        "evidence": evidence,
        "model_label": model_label,
        "confidence": round(confidence, 4),
        "final_label": final_label,
        "reason": reason,
    }


def predict_baseline(claim, evidence, model, vectorizer):
    # baseline model expects combined text like in training
    text = (claim or "") + " " + (evidence or "")
    X_vec = vectorizer.transform([text])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_vec)[0]
        pred = int(model.predict(X_vec)[0])
        confidence = float(max(probs))
    else:
        pred = int(model.predict(X_vec)[0])
        confidence = 0.0

    model_label = labels_map.get(pred, "SUPPORTS")

    return {
        "claim": claim,
        "evidence": evidence,
        "model_label": model_label,
        "confidence": round(confidence, 4),
        "final_label": model_label,
        "reason": "",
    }


# Backwards compatible default: BERT with reasoning
def predict(claim, evidence, model, tokenizer):
    return predict_bert(claim, evidence, model, tokenizer, use_reasoning=True)