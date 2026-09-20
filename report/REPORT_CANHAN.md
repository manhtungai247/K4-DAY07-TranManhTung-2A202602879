# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** [Trần Mạnh Tùng]
**Nhóm:** [sieunhandienquang]
**Ngày:** [20/9/2026]

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Độ tương tự cosine cao (tiến gần về 1.0) nghĩa là hai vector embedding cùng chỉ về một hướng trong không gian đa chiều, phản ánh rằng hai đoạn văn bản có sự tương đồng lớn về mặt ngữ nghĩa và ngữ cảnh, bất kể độ dài văn bản hay sự khác biệt về từ ngữ bề mặt.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Khách hàng có thể gửi yêu cầu hoàn tiền trong vòng 15 ngày kể từ khi nhận kiện hàng."
- Câu B: "Người mua được phép khiếu nại trả hàng và nhận lại tiền trong thời hạn nửa tháng sau khi đơn giao thành công."
- Tại sao tương đồng: Hai câu sử dụng các từ vựng hoàn toàn khác nhau ("khách hàng" vs "người mua", "hoàn tiền" vs "nhận lại tiền", "15 ngày" vs "nửa tháng"), nhưng mô tả cùng một bản chất quy định chính sách, do đó mô hình embedding sẽ ánh xạ chúng về hướng vector gần như trùng nhau.

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Khách hàng có thể gửi yêu cầu hoàn tiền trong vòng 15 ngày kể từ khi nhận kiện hàng."
- Câu B: "Thuật toán Gradient Descent giúp tối ưu hóa hàm mất mát của mô hình học sâu."
- Tại sao khác: Hai câu thuộc hai miền tri thức hoàn toàn tách biệt (chính sách đổi trả thương mại điện tử vs thuật toán toán học/AI) và không chia sẻ ngữ cảnh ngữ nghĩa nào, nên góc giữa hai vector gần như vuông góc (cosine xấp xỉ 0.0).

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Khoảng cách Euclid bị chi phối mạnh bởi độ lớn (độ dài) của vector — một câu dài và một câu ngắn diễn đạt cùng một ý sẽ có khoảng cách Euclid rất xa nhau. Ngược lại, độ tương tự cosine chỉ quan tâm đến góc giữa hai vector (hướng ngữ nghĩa) và hoàn toàn triệt tiêu ảnh hưởng của độ dài văn bản, giúp việc so khớp ngữ nghĩa chính xác hơn.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
> Áp dụng công thức: `số_lượng_chunk = ceil((độ_dài - overlap) / (chunk_size - overlap))`
> Thay số: `ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11) = 23`
> *Đáp án:* **23 chunks** (đã xác minh chính xác bằng `FixedSizeChunker(500, 50)`).

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Khi tăng overlap lên 100: `ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = ceil(24.75) = 25` chunks (tăng 2 chunks so với ban đầu).
> Chúng ta muốn tăng độ chồng chéo vì nó giúp bảo tồn ngữ cảnh trọn vẹn tại các ranh giới phân tách (boundary), tránh hiện tượng một câu văn quan trọng hoặc một mốc điều kiện chính sách bị cắt đôi làm mất ý nghĩa khi đưa vào truy xuất (retrieval).

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Tôi sử dụng biểu thức chính quy với kỹ thuật Positive Lookbehind `r"(?<=[.!?])\s+"` để tách câu tại các vị trí sau dấu chấm, chấm than, chấm hỏi mà không làm mất dấu câu ở cuối. Sau đó, các câu được gom nhóm thành từng chunk với số lượng tối đa là `max_sentences_per_chunk`. Xử lý ngoại lệ (edge case): văn bản rỗng hoặc chỉ có khoảng trắng trả về `[]`; văn bản không có dấu câu được giữ nguyên làm 1 chunk; đồng thời nhận diện giới hạn chưa xử lý triệt để các trường hợp từ viết tắt (như "v.v.", "TS.") hoặc số thập phân.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thuật toán hoạt động theo nguyên lý 2 chiều: (1) Đệ quy xuống sâu theo thứ tự ưu tiên của separators `["\n\n", "\n", ". ", " ", ""]`, mảnh nào dài hơn `chunk_size` sẽ tiếp tục được chia bằng separator nhỏ hơn kế tiếp; (2) Gom lên (merge), nối các mảnh nhỏ liền kề lại cho tới khi sát ngưỡng `chunk_size` để tránh sinh ra các chunk vụn 5–10 ký tự. Base cases bao gồm: độ dài văn bản nhỏ hơn hoặc bằng `chunk_size` thì trả về ngay `[text]`, và khi danh sách separators rỗng (`separators == []`) thì thực hiện cắt cứng (hard slice) theo `chunk_size`.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Lưu trữ các tài liệu dưới dạng danh sách record chuẩn hóa trong bộ nhớ (in-memory list), mỗi record chứa `id`, `content`, `metadata` và vector `embedding` được tạo từ hàm nhúng. Khi tìm kiếm (`search`), hệ thống nhúng truy vấn của người dùng rồi tính tích vô hướng (dot product) với từng vector tài liệu đã chuẩn hóa độ dài ($||v|| = 1$), sau đó sắp xếp giảm dần theo điểm `score` và trả về danh sách `top_k` kết quả tốt nhất.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> Áp dụng cơ chế **Pre-filtering** (lọc metadata TRƯỚC khi tính toán tìm kiếm tương đồng): chọn lọc ra tập các record thỏa mãn toàn bộ điều kiện trong `metadata_filter` trước, rồi mới chạy similarity search trên tập này nhằm tránh việc các tài liệu sai đối tượng chiếm mất các vị trí trong `top_k`. Hàm `delete_document` duyệt và loại bỏ toàn bộ các chunk có `id` hoặc `metadata['doc_id']` khớp với mã tài liệu cần xóa, trả về `True` nếu có bản ghi bị xóa và `False` nếu không tìm thấy.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Truy xuất `top_k` chunk liên quan từ store (có áp dụng filter nếu có). Dựng prompt có cấu trúc chặt chẽ: đánh số thứ tự từng ngữ cảnh trích xuất dạng `[1] (Nguồn: ...)` kèm nội dung chunk, đồng thời bổ sung chỉ dẫn ràng buộc yêu cầu LLM chỉ trả lời dựa trên ngữ cảnh được cung cấp và phải trích dẫn số thứ tự nguồn (chống hallucination). Nếu kho tài liệu rỗng hoặc không tìm thấy kết quả phù hợp, hàm trả về thông báo lịch sự mà không gọi LLM để tiết kiệm tài nguyên.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```text
============================= test session starts =============================
platform win32 -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\admin\K4-L3B-Data-Foundations
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.06s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|:---:|:------|:------|:-------:|:------------:|:-----:|
| 1 | "Thời hạn trả hàng Shopee Mall là 15 ngày." | "Người mua có thể yêu cầu trả hàng trong nửa tháng đối với Shopee Mall." | cao | -0.0657 | Sai |
| 2 | "Quy định về việc đóng gói bưu kiện hoàn trả." | "Cách thức gói hàng và niêm phong sản phẩm trả về." | cao | -0.1000 | Sai |
| 3 | "Chính sách bồi thường khi đơn vị vận chuyển làm thất lạc hàng hóa." | "Mô hình học máy Deep Learning sử dụng mạng nơ-ron tích chập." | thấp | -0.1623 | Đúng |
| 4 | "Người bán cần xác nhận nhận hàng trong Kênh Quản Lý Shop." | "Khách hàng liên hệ tổng đài để kiểm tra hành trình đơn giao." | thấp | 0.1345 | Sai |
| 5 | "Kích thước tối đa của kiện hàng SPX Instant là 60 cm." | "Kiện hàng hỏa tốc không được vượt quá 60 centimet mỗi chiều." | cao | -0.1470 | Sai |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Kết quả bất ngờ nhất là các cặp câu đồng nghĩa hoàn toàn (Cặp 1 và Cặp 5: "15 ngày" vs "nửa tháng", "60 cm" vs "60 centimet") lại nhận điểm tương tự âm, trong khi hai câu khác ngữ cảnh (Cặp 4) lại có điểm dương. Điều này phản ánh rõ ràng rằng `MockEmbedder` chỉ băm mã MD5 ngẫu nhiên và không hề có khả năng biểu diễn ngữ nghĩa. Để embeddings thực sự phản ánh được ý nghĩa và ngữ cảnh trong không gian vector, hệ thống bắt buộc phải sử dụng các mô hình ngôn ngữ được huấn luyện chuyên sâu (như Sentence Transformers hoặc OpenAI/Gemini Embeddings).

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-----------------|--------------------------------------|:----------:|:-----------------------------:|---------------------------------|
| 1 | Quy định về kích thước và trọng lượng tối đa của kiện hàng đối với dịch vụ SPX Instant là bao nhiêu? | shopee-chinh-sach-van-chuyen: B. Quy định về hàng hóa không hỗ trợ vận chuyển, vận chuyển có điều kiện... | 0.2822 | Một phần | Trả về quy định vận chuyển chung, chưa trích xuất được số đo cụ thể 60x60x60cm và 30kg của SPX Instant. |
| 2 | Người mua cần lưu ý gì khi đóng gói hàng hóa hoàn trả có chứa chất lỏng hoặc dễ vỡ? | shopee-chinh-sach-van-chuyen: C. Quy định về đóng gói hàng hóa, thông tin dán nhãn bưu kiện... | 0.3402 | Có | Trả lời được quy định đóng bọc và dán nhãn, nhưng thiếu chi tiết đóng chặt nắp chai và chèn xốp bóng khí. |
| 3 | Người bán cần làm gì khi đơn vị vận chuyển hoàn trả hàng về kho thành công và hàng còn nguyên vẹn? | shopee-dong-goi-don-hoan-tra: Cách đóng gói đơn hàng hoàn trả, quay video quy trình... | 0.3047 | Không | Trả nhầm tài liệu đóng gói của người mua thay vì hướng dẫn xử lý nhập kho của người bán. |
| 4 | Giá trị đơn hàng tối đa áp dụng cho phương thức thanh toán COD khi sử dụng dịch vụ SPX Instant là bao nhiêu? | shopee-dong-goi-don-hoan-tra: Mã vận đơn vào mục Tôi > Trả hàng/hoàn tiền... | 0.3298 | Không | Bị nhiễu bởi các từ khóa thanh toán/đơn hàng, không tìm thấy hạn mức COD 5.000.000Đ. |
| 5 | Quy định xử lý đơn trả hàng và quản lý hàng hoàn trả tại Kênh Quản Lý Shop được thực hiện như thế nào? (Filter: seller) | shopee-seller-don-giao-khong-thanh-cong: Quản lý đơn hàng giao không thành công, nhận hàng hoàn... (Top-2: shopee-seller-quan-ly-don-tra-hang) | 0.1613 | Có (Rất tốt) | Nhờ lọc `audience: seller`, Agent trả lời chính xác quy trình người bán theo dõi đơn hoàn tại Kênh Quản Lý Shop và thủ tục đền bù. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 3 / 5

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Qua quá trình thử nghiệm và đối chiếu, tôi nhận thấy chiến lược `HeadingChunker` giúp các đoạn văn bản giữ trọn vẹn ngữ nghĩa từng điều khoản tốt hơn hẳn so với cắt vụn theo ký tự (`FixedSizeChunker`). Tuy nhiên, bài học lớn nhất là khi kho dữ liệu có các văn bản quá dài, kỹ thuật **Pre-filtering bằng Metadata** đóng vai trò quyết định để loại bỏ 100% tài liệu sai đối tượng (như ở Query 5), giúp việc tìm kiếm chính xác và tránh bị tài liệu nhiễu chiếm mất top-k.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|:----------------:|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |
