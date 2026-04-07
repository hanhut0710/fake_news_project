
import os
import sys
from xml.parsers.expat import model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, request, render_template
from predict import predict, predict_baseline, predict_bert
from query_evidence import query_evidence, get_timing_summary
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import torch
import spacy

nlp = spacy.load("en_core_web_sm")

predict_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("model/model_bert/")
model_predict = AutoModelForSequenceClassification.from_pretrained("model/model_bert/")
model_predict.to(predict_device)
model_predict.eval()

retriever_device = "cpu"
model_retriever = SentenceTransformer("all-MiniLM-L6-v2", device=retriever_device)

# Try load baseline model/vectorizer if available
baseline_model = None
baseline_vectorizer = None
try:
    from baseline import load_baseline
    baseline_model, baseline_vectorizer = load_baseline("model/baseline")
    print("Loaded baseline model from model/baseline")
except Exception as e:
    print("Baseline model not loaded:", e)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        print("Received claim:", request.form["claim"])
        claim = request.form["claim"]
        mode = request.form.get("mode", "bert_with_reasoning")

        # Step 1: Retrieve evidence (use shared retriever instance)
        evidence = query_evidence(claim, model_retriever, nlp)

        # Step 2: Predict (pass classifier + tokenizer)
        if mode == "baseline":
            if baseline_model is None or baseline_vectorizer is None:
                result = {
                    "claim": claim,
                    "evidence": evidence,
                    "model_label": "N/A",
                    "confidence": 0.0,
                    "final_label": "N/A",
                    "reason": "Baseline model not available on server. Run baseline.py to create it."
                }
            else:
                result = predict_baseline(claim, evidence, baseline_model, baseline_vectorizer)
        elif mode == "bert_no_reasoning":
            result = predict_bert(claim, evidence, model_predict, tokenizer, use_reasoning=False)
        else:
            # default: bert with reasoning
            result = predict_bert(claim, evidence, model_predict, tokenizer, use_reasoning=True)
        # collect timing summary from retriever (if available)
        timings = get_timing_summary()
    else:
        timings = {}

    # pass mode so the UI can reflect the selected option
    if request.method == "POST":
        return render_template("index.html", result=result, timings=timings, mode=mode)
    else:
        return render_template("index.html", result=result, timings=timings, mode="bert_with_reasoning")

if __name__ == "__main__":
    app.run(debug=True)