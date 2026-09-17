from __future__ import annotations

from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag.schemas import KnowledgeChunk, RetrievedChunk


class LocalTfidfRetriever:
    """
    Lightweight local-first retriever using TF-IDF and cosine similarity.
    """

    def __init__(self, chunks: List[KnowledgeChunk]) -> None:
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")

        documents = [
            f"{chunk.title} {chunk.category} {' '.join(chunk.tags)} {chunk.text}"
            for chunk in self.chunks
        ]
        self.chunk_vectors = self.vectorizer.fit_transform(documents)

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedChunk]:
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.chunk_vectors)[0]

        ranked_indices = scores.argsort()[::-1][:top_k]

        results: List[RetrievedChunk] = []
        for idx in ranked_indices:
            chunk = self.chunks[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    title=chunk.title,
                    category=chunk.category,
                    audience=chunk.audience,
                    text=chunk.text,
                    tags=chunk.tags,
                    similarity_score=round(float(scores[idx]), 6),
                )
            )

        return results