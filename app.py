from flask import Flask, render_template, request
from query_core import query_rag

app = Flask(__name__)

@app.get("/")
def index():
    return render_template("index.html", answer=None, sources=None)

@app.post("/ask")
def ask():
    question = (request.form.get("question") or "").strip()
    model = (request.form.get("model") or "qwen2.5:1.5b").strip()
    source = (request.form.get("source") or "").strip()
    k_raw = (request.form.get("k") or "4").strip()

    try:
        k = max(1, min(20, int(k_raw)))
    except ValueError:
        k = 5

    if not question:
        return render_template("index.html", answer="Pose une question 🙂", sources=[])

    answer, sources = query_rag(question, k=k, model_name=model, source_filter=source)
    return render_template("index.html", answer=answer, sources=sources, question=question, model=model, source=source, k=k)

if __name__ == "__main__":
    # Local only (sécurité)
    app.run(host="127.0.0.1", port=8085, debug=True, use_reloader=False)
