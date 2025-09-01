from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import os, glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

@dataclass
class DocChunk:
    doc_id: str
    text: str

class TFIDFRetriever:
    def __init__(self, vectorizer: TfidfVectorizer, doc_ids: List[str], matrix):
        self.vectorizer = vectorizer
        self.doc_ids = doc_ids
        self.matrix = matrix

    @classmethod
    def fit_from_folder(cls, folder: str) -> 'TFIDFRetriever':
        paths = sorted(glob.glob(os.path.join(folder, "*.txt")))
        texts = [open(p, "r", encoding="utf-8").read() for p in paths]
        vec = TfidfVectorizer(stop_words="english")
        X = vec.fit_transform(texts)
        return cls(vec, paths, X)

    def save(self, path: str):
        joblib.dump(self, path)

    def retrieve(self, query: str, k: int = 3) -> List[DocChunk]:
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix)[0]
        top_idx = sims.argsort()[::-1][:k]
        chunks = []
        for i in top_idx:
            with open(self.doc_ids[i], "r", encoding="utf-8") as f:
                chunks.append(DocChunk(doc_id=os.path.basename(self.doc_ids[i]), text=f.read().strip()))
        return chunks
