"""
Benchmark script for K4-L3B: E-commerce Policy Retrieval.
Strategy: HeadingChunker (Section-aware Markdown chunking with header inheritance).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
from dotenv import load_dotenv

from src.chunking import RecursiveChunker
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    GEMINI_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    GeminiEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore


class HeadingChunker:
    """
    Chiến lược chia nhỏ theo Tiêu đề / Mục (Heading/Section-based chunking).
    
    Lý do thiết kế:
    - Văn bản chính sách sàn TMĐT được biên soạn theo từng mục và điều khoản rõ ràng.
    - Cắt theo heading giúp bảo tồn trọn vẹn đơn vị ngữ nghĩa của từng quy định.
    - Với các section quá dài, chia nhỏ bằng RecursiveChunker và gắn lại tiêu đề mục
      vào đầu mỗi chunk con để không bị mất ngữ cảnh (Header Inheritance).
    """

    def __init__(self, max_chunk_size: int = 600) -> None:
        self.max_chunk_size = max_chunk_size
        self._fallback_chunker = RecursiveChunker(chunk_size=max_chunk_size)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        # Tách theo Markdown headings (#, ##, ###) hoặc đề mục lớn (A., B., C.)
        sections = re.split(r"(?=(?:^|\n)(?:#{1,4}\s+|[A-Z]\.\s+))", text.strip())
        sections = [s.strip() for s in sections if s.strip()]

        chunks: list[str] = []
        for sec in sections:
            if len(sec) <= self.max_chunk_size:
                chunks.append(sec)
            else:
                # Trích xuất dòng tiêu đề của section nếu có
                first_line = sec.splitlines()[0] if sec.splitlines() else ""
                header_prefix = f"{first_line}\n" if first_line.startswith(("#", "A.", "B.", "C.", "D.")) else ""

                sub_chunks = self._fallback_chunker.chunk(sec)
                for idx, sub in enumerate(sub_chunks):
                    if idx > 0 and header_prefix and not sub.startswith(header_prefix.strip()):
                        chunks.append(f"{header_prefix.strip()}\n{sub}")
                    else:
                        chunks.append(sub)

        return chunks if chunks else [text]


# Định nghĩa 5 Benchmark Queries chuẩn cho biến thể K4-L3B
BENCHMARK_QUERIES = [
    {
        "id": 1,
        "query": "Quy định về kích thước và trọng lượng tối đa của kiện hàng đối với dịch vụ SPX Instant là bao nhiêu?",
        "gold_answer": "Dài <= 60 cm, Rộng <= 60 cm, Cao <= 60 cm; Trọng lượng tối đa <= 30kg sau khi đóng gói.",
        "expected_doc": "ecom-shopee-co-check",
        "keywords": ["60 cm", "30kg", "SPX Instant"],
        "filter": None,
    },
    {
        "id": 2,
        "query": "Người mua cần lưu ý gì khi đóng gói hàng hóa hoàn trả có chứa chất lỏng hoặc dễ vỡ?",
        "gold_answer": "Đóng chặt nắp chai, cho vào thùng vừa với kích cỡ, sử dụng vật liệu đệm (bong bóng, màng co, xốp) để giảm va đập và luôn ghi hình lại quá trình đóng gói.",
        "expected_doc": "shopee-dong-goi-don-hoan-tra",
        "keywords": ["chất lỏng", "đóng chặt nắp", "bong bóng", "xốp"],
        "filter": None,
    },
    {
        "id": 3,
        "query": "Người bán cần làm gì khi đơn vị vận chuyển hoàn trả hàng về kho thành công và hàng còn nguyên vẹn?",
        "gold_answer": "Tại mục Trả hàng thành công, chọn Xác nhận Nhận hàng -> chọn 'Nhập lại hàng vào kho' -> nhập số lượng thực tế và nhấn 'Nhập tồn kho nhanh'.",
        "expected_doc": "shopee-seller-quan-ly-don-tra-hang",
        "keywords": ["Nhập lại hàng vào kho", "Nhập tồn kho nhanh"],
        "filter": None,
    },
    {
        "id": 4,
        "query": "Giá trị đơn hàng tối đa áp dụng cho phương thức thanh toán COD khi sử dụng dịch vụ SPX Instant là bao nhiêu?",
        "gold_answer": "Giá trị đơn hàng tối đa đối với phương thức thanh toán COD là 5.000.000Đ (các phương thức khác là 10.000.000Đ).",
        "expected_doc": "ecom-shopee-co-check",
        "keywords": ["5.000.000Đ", "COD", "SPX Instant"],
        "filter": None,
    },
    {
        # Câu hỏi bắt buộc của L3B cần metadata_filter={"audience": "seller"}
        "id": 5,
        "query": "Quy trình xử lý đơn trả hàng và quản lý hàng hoàn trả tại Kênh Quản Lý Shop được thực hiện như thế nào?",
        "gold_answer": "Người bán theo dõi trạng thái tại Kênh Quản Lý Shop. Nếu hàng nguyên vẹn thì chọn Nhập lại hàng vào kho; nếu hàng thất lạc/hư hỏng thì chọn Thêm chi phí để được Shopee đền bù.",
        "expected_doc": "shopee-seller-quan-ly-don-tra-hang",
        "keywords": ["Kênh Quản Lý Shop", "Nhập lại hàng vào kho", "Thêm chi phí"],
        "filter": {"audience": "seller"},
    },
]


def load_and_chunk_corpus(data_dir: Path, chunker: HeadingChunker) -> list[Document]:
    md_files = sorted(data_dir.glob("*.md"))
    all_docs: list[Document] = []

    for file_path in md_files:
        content = file_path.read_text(encoding="utf-8")
        if "---" in content:
            parts = content.split("---", 2)
            frontmatter_raw = parts[1]
            body_content = parts[2] if len(parts) > 2 else ""
        else:
            frontmatter_raw = ""
            body_content = content

        # Parse YAML frontmatter đơn giản
        metadata = {}
        for line in frontmatter_raw.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                metadata[k.strip()] = v.strip().strip('"').strip("'").split("#")[0].strip()

        doc_id = metadata.get("doc_id", file_path.stem)
        metadata["doc_id"] = doc_id

        # Chunk nội dung phần thân
        chunks = chunker.chunk(body_content)
        for idx, ch in enumerate(chunks):
            chunk_doc = Document(
                id=f"{doc_id}#{idx}",
                content=ch,
                metadata={**metadata, "chunk_index": idx, "doc_id": doc_id},
            )
            all_docs.append(chunk_doc)

    return all_docs


def run_benchmark(output_file: Path | None = None) -> None:
    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "gemini":
        try:
            embedder = GeminiEmbedder(model_name=os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    data_dir = Path("data/ecommerce")
    chunker = HeadingChunker(max_chunk_size=600)
    docs = load_and_chunk_corpus(data_dir, chunker)

    store = EmbeddingStore("ecommerce_benchmark", embedding_fn=embedder)
    store.add_documents(docs)

    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("K4-L3B BENCHMARK RETRIEVAL REPORT")
    lines.append("Chiến lược: HeadingChunker (Chia theo Tiêu đề / Section + Header Inheritance)")
    lines.append(f"Mô hình nhúng (Embedder): {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")
    lines.append(f"Số lượng chunk đã nạp vào Vector Store: {store.get_collection_size()} chunks")
    lines.append("=" * 80)
    lines.append("")

    total_score = 0
    max_score = len(BENCHMARK_QUERIES) * 2

    for item in BENCHMARK_QUERIES:
        q_id = item["id"]
        q_text = item["query"]
        expected = item["expected_doc"]
        keywords = item["keywords"]
        m_filter = item["filter"]

        lines.append(f"Query {q_id}: {q_text}")
        lines.append(f"Câu trả lời chuẩn (Gold): {item['gold_answer']}")
        if m_filter:
            lines.append(f"Metadata filter áp dụng: {m_filter}")
            results = store.search_with_filter(q_text, top_k=3, metadata_filter=m_filter)
        else:
            results = store.search(q_text, top_k=3)

        # Đánh giá 2 mức (doc_id và nội dung chứa từ khóa)
        top1_doc = results[0]["metadata"].get("doc_id") if results else None
        top_docs = [r["metadata"].get("doc_id") for r in results]
        
        has_relevant_content = False
        gold_at_top1 = (top1_doc == expected)
        gold_in_top3 = (expected in top_docs)

        for r in results:
            if any(kw.lower() in r["content"].lower() for kw in keywords):
                has_relevant_content = True
                break

        if gold_at_top1 and has_relevant_content:
            score = 2
        elif gold_in_top3 or has_relevant_content:
            score = 1
        else:
            score = 0

        total_score += score
        lines.append(f"Điểm đánh giá câu này: {score}/2 đ")
        lines.append("Top-3 Chunks tìm thấy:")
        for rank, r in enumerate(results, start=1):
            chunk_doc_id = r["metadata"].get("doc_id")
            score_val = r["score"]
            preview = r["content"].replace("\n", " ")[:110]
            lines.append(f"  [{rank}] Score: {score_val:.4f} | Doc: {chunk_doc_id} | Preview: {preview}...")
        lines.append("-" * 80)

    # Thử nghiệm A/B trên Query 5 (So sánh Không filter vs Có filter)
    lines.append("")
    lines.append("=== THỬ NGHIỆM A/B: HIỆU QUẢ CỦA METADATA FILTERING TRÊN QUERY 5 ===")
    q5 = BENCHMARK_QUERIES[4]["query"]
    res_no_filter = store.search(q5, top_k=3)
    res_filtered = store.search_with_filter(q5, top_k=3, metadata_filter={"audience": "seller"})

    lines.append(f"Query: {q5}")
    lines.append("1. Khi KHÔNG dùng filter:")
    for rank, r in enumerate(res_no_filter, start=1):
        lines.append(f"   [{rank}] Doc: {r['metadata'].get('doc_id')} (audience: {r['metadata'].get('audience')})")

    lines.append("2. Khi CÓ filter {'audience': 'seller'}:")
    for rank, r in enumerate(res_filtered, start=1):
        lines.append(f"   [{rank}] Doc: {r['metadata'].get('doc_id')} (audience: {r['metadata'].get('audience')})")

    lines.append("")
    lines.append("=" * 80)
    lines.append(f"TỔNG KẾT ĐIỂM TRUY XUẤT (RETRIEVAL QUALITY): {total_score} / {max_score}")
    lines.append("=" * 80)

    output_content = "\n".join(lines)
    print(output_content)

    if output_file:
        output_file.write_text(output_content, encoding="utf-8")
        print(f"\nĐã xuất toàn bộ kết quả vào: {output_file}")


if __name__ == "__main__":
    out_path = Path("ket_qua_benchmark.txt")
    run_benchmark(out_path)
