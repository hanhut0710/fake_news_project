import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from reasoning import apply_reasoning

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

labels_map = {
    1: "REFUTES",
    0: "SUPPORTS",
}

def predict(claim, evidence, model, tokenizer):
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

    # 👉 APPLY REASONING
    final_label, reason = apply_reasoning(claim, evidence, model_label, confidence)

    return {
        "claim": claim,
        "evidence": evidence,
        "model_label": model_label,
        "confidence": round(confidence, 4),
        "final_label": final_label,
        "reason": reason,
    }