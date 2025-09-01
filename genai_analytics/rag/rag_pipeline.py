from typing import Tuple, List
from genai_analytics.rag.retriever import TFIDFRetriever, DocChunk
from genai_analytics.rag.prompts import build_prompt
from genai_analytics.llm.inference import generate_text

class RagPipeline:
    def __init__(self, retriever: TFIDFRetriever):
        self.retriever = retriever

    def answer(self, query: str, k: int = 3) -> Tuple[str, List[DocChunk]]:
        ctx = self.retriever.retrieve(query, k=k)
        prompt = build_prompt(query, ctx)
        out = generate_text(prompt, max_new_tokens=128)
        return out, ctx
