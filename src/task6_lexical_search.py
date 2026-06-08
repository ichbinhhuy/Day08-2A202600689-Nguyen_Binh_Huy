"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

from pathlib import Path
import chromadb
from rank_bm25 import BM25Okapi

# Load corpus từ ChromaDB
CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}
_BM25_INDEX = None

def load_corpus_from_chroma():
    global CORPUS, _BM25_INDEX
    if CORPUS:
        return
        
    current_dir = Path(__file__).parent
    db_path = str(current_dir.parent / "chroma_data")
    try:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection("DrugLawDocs")
        
        # Get all documents
        results = collection.get(include=["metadatas", "documents"])
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        
        for doc, meta in zip(docs, metas):
            CORPUS.append({"content": doc, "metadata": meta or {}})
            
        if CORPUS:
            _BM25_INDEX = build_bm25_index(CORPUS)
            print(f"Loaded {len(CORPUS)} chunks into BM25 index.")
    except Exception as e:
        print("Lỗi load corpus từ ChromaDB:", e)

def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    if not CORPUS:
        load_corpus_from_chroma()
        
    if not _BM25_INDEX:
        return []

    tokenized_query = query.lower().split()
    scores = _BM25_INDEX.get_scores(tokenized_query)

    import numpy as np
    # Get top_k indices
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(scores[idx]),
                "metadata": CORPUS[idx]["metadata"]
            })
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
