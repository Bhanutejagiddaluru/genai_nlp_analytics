from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import joblib
import os

from genai_analytics.rag.rag_pipeline import RagPipeline
from genai_analytics.analytics.financial_signals import compute_signals
from genai_analytics.llm.inference import small_generate
from genai_analytics.rag.retriever import TFIDFRetriever

CFG_INDEX = os.getenv("RAG_INDEX_PATH", "configs/tfidf_index.joblib")

app = FastAPI(title="GenAI & NLP Analytics Systems")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = None
rag = None

class RagQuery(BaseModel):
    query: str
    k: int = 3

class FinanceQuery(BaseModel):
    symbol: str = "AAPL"
    n: int = 200

@app.on_event("startup")
def on_startup():
    global retriever, rag
    if os.path.exists(CFG_INDEX):
        retriever = joblib.load(CFG_INDEX)
    else:
        retriever = TFIDFRetriever.fit_from_folder("data/docs")
    rag = RagPipeline(retriever=retriever)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/rag/query")
def rag_query(payload: RagQuery):
    answer, ctx = rag.answer(payload.query, k=payload.k)
    return {"answer": answer, "context": ctx}

@app.post("/finance/signal")
def finance_signal(payload: FinanceQuery):
    df, feats = compute_signals(symbol=payload.symbol, n=payload.n)
    # Return small preview to keep payload light
    head = df.tail(10).reset_index().to_dict(orient="records")
    return {"symbol": payload.symbol, "preview": head, "features": feats}

@app.websocket("/ws/generate")
async def ws_generate(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            # Stream token-by-token demo from a tiny generator
            async for token in small_generate(data):
                await ws.send_text(token)
                await asyncio.sleep(0.02)
            await ws.send_text("\n[END]")
    except WebSocketDisconnect:
        pass
