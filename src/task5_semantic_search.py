"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Tải model 1 lần (Singleton pattern) để tránh load lại model nhiều lần khi gọi hàm nhiều lần
_MODEL = None

def get_embedding_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _MODEL

def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    model = get_embedding_model()
    query_embedding = model.encode(query).tolist()

    # Kết nối ChromaDB
    current_dir = Path(__file__).parent
    db_path = str(current_dir.parent / "chroma_data")
    
    try:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection("DrugLawDocs")
    except Exception as e:
        print("Không tìm thấy collection. Bạn đã chạy task 4 để tạo index chưa?")
        return []

    # Thực hiện search theo vector
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    out_results = []
    if results["ids"] and len(results["ids"][0]) > 0:
        docs = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        
        for i in range(len(docs)):
            # ChromaDB mặc định trả về L2 distance. Để có score giống similarity (càng lớn càng tốt), ta chuyển đổi:
            dist = distances[i]
            score = 1.0 / (1.0 + dist)
            
            meta = metadatas[i] or {}
            out_results.append({
                "content": docs[i],
                "score": score,
                "metadata": {
                    "source": meta.get("source", ""),
                    "type": meta.get("type", ""),
                    "chunk_index": meta.get("chunk_index", 0)
                }
            })

    # Đảm bảo kết quả được sắp xếp giảm dần theo điểm
    out_results.sort(key=lambda x: x["score"], reverse=True)
    
    return out_results


if __name__ == "__main__":
    # Test thử trực tiếp
    print("Testing Semantic Search with ChromaDB...")
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=3)
    if not results:
        print("Không có kết quả. Hãy chắc chắn bạn đã chạy Task 4 để đưa dữ liệu vào ChromaDB.")
    else:
        for i, r in enumerate(results):
            # To avoid Windows console UnicodeEncodeError, using ascii-safe strings
            print(f"\n[{i+1}] Score: {r['score']:.4f} | Source: {r['metadata']['source']}")
            print(f"{r['content'][:200]}...")
