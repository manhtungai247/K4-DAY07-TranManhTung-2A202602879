from __future__ import annotations

from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3, metadata_filter: dict | None = None) -> str:
        if self.store.get_collection_size() == 0:
            return "Không tìm thấy tài liệu phù hợp trong cơ sở tri thức."

        if metadata_filter:
            results = self.store.search_with_filter(question, top_k=top_k, metadata_filter=metadata_filter)
        else:
            results = self.store.search(question, top_k=top_k)

        if not results:
            return "Không tìm thấy thông tin phù hợp để trả lời câu hỏi."

        context_parts: list[str] = []
        for idx, r in enumerate(results, start=1):
            source = r.get("metadata", {}).get("title", r.get("id", "Doc"))
            context_parts.append(f"[{idx}] (Nguồn: {source}):\n{r['content']}")

        context_str = "\n\n".join(context_parts)
        prompt = (
            f"Dưới đây là ngữ cảnh trích xuất từ cơ sở tri thức:\n\n"
            f"{context_str}\n\n"
            f"Dựa trên các thông tin trên, hãy trả lời câu hỏi sau (trích dẫn số thứ tự nguồn [1], [2] nếu có):\n"
            f"Câu hỏi: {question}\n"
            f"Trả lời:"
        )
        return self.llm_fn(prompt)
