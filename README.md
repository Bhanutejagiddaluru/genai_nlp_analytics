# Generative AI & NLP Analytics Systems — Working Prototype

This repo is a **production-style starter** for building RAG, LLM fine‑tuning, real‑time streaming via WebSockets, and **financial analytics**—with optional distributed compute (Spark, Ray, Dask).

## Features
- **RAG pipeline** (TF‑IDF retriever by default; easily swap in embeddings/FAISS/Chroma).
- **LLM inference** via Hugging Face `transformers`, plus **stubs for PyTorch & TensorFlow fine‑tuning**.
- **FastAPI service** exposing `/rag/query`, `/finance/signal`, and `/health`.
- **WebSocket streaming** endpoint for token‑by‑token responses.
- **Streamlit UI** with two dropdowns: **Model** (RAG/LLM) and **Series** (financial symbol) + chat.
- **Financial analytics** (RSI, MACD) and basic time‑series forecasting baseline.
- **Distributed compute stubs**: Spark batch job, Ray actor, Dask pipeline.

> ✅ You can run everything locally without GPUs; fine‑tuning scripts are stubs to extend as needed.

## Quickstart
```bash
python -m venv .venv && . .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Build a tiny doc index for RAG
python scripts/bootstrap_index.py --docs data/docs --out configs/tfidf_index.joblib

# 2) Run the API (HTTP + WebSocket)
uvicorn genai_analytics.api.main:app --reload --port 8000

# 3) Launch the Streamlit UI (in a separate terminal)
streamlit run genai_analytics/ui/streamlit_app.py
```

### Try it
- Open Streamlit: http://localhost:8501
- Query: “Explain RAG for financial tickers”
- Finance symbol: `AAPL` or `MSFT` (toy synthetic data by default)

## Project Layout
```
genai_nlp_analytics/
├─ genai_analytics/
│  ├─ api/
│  │  └─ main.py
│  ├─ analytics/
│  │  ├─ financial_signals.py
│  │  ├─ timeseries_models.py
│  │  ├─ spark_job.py
│  │  ├─ ray_workers.py
│  │  └─ dask_pipeline.py
│  ├─ llm/
│  │  ├─ finetune_torch.py
│  │  ├─ finetune_tensorflow.py
│  │  └─ inference.py
│  ├─ rag/
│  │  ├─ retriever.py
│  │  ├─ rag_pipeline.py
│  │  └─ prompts.py
│  ├─ streaming/
│  │  └─ websocket_server.py
│  └─ ui/
│     ├─ streamlit_app.py
│     └─ images/
│        ├─ architecture.png
│        └─ ui_mock.png
├─ configs/
│  ├─ config.yaml
│  └─ tfidf_index.joblib  # created after bootstrap
├─ data/
│  └─ docs/               # sample text docs
├─ scripts/
│  └─ bootstrap_index.py
├─ tests/
│  └─ test_retriever.py
├─ requirements.txt
├─ docker-compose.yml
├─ Dockerfile
└─ README.md
```

## Notes
- Swap `TFIDFRetriever` with an embeddings store later (FAISS, ChromaDB). The code is layered for easy replacement.
- `transformers` pipeline uses a small model for demo; replace with your preferred LLM.
- Spark/Ray/Dask examples are **runnable stubs**—adapt to your infra.
