
import os
import sys
from xml.parsers.expat import model

from sentence_transformers import SentenceTransformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



from flask import Flask, request, render_template
from predict import predict
from query_evidence import query_evidence
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("model_evidence_claim/")
model_predict = AutoModelForSequenceClassification.from_pretrained("model_evidence_claim/")
model_predict.to(device)
model_predict.eval()

model_query = SentenceTransformer("all-MiniLM-L6-v2").to(device)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        print("Received claim:", request.form["claim"])
        claim = request.form["claim"]

        # Step 1: Retrieve evidence
        evidence = query_evidence(claim, model_query)

        # Step 2: Predict
        result = predict(claim, evidence, model_predict, tokenizer)

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)