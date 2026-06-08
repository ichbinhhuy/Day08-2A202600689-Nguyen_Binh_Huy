"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# Chọn chunking strategy: RecursiveCharacterTextSplitter
# Vì nó đơn giản, an toàn và giữ được các đoạn văn không bị cắt đứt giữa chừng nếu setup separator chuẩn.
CHUNK_SIZE = 1000        # Độ dài khoảng 150-200 từ, đủ chứa 1 điều luật hoặc 1 đoạn tin tức
CHUNK_OVERLAP = 200      # Đủ dài (khoảng 1-2 câu) để nối ý giữa các chunk liền kề, tránh mất ngữ cảnh
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# Chọn embedding model: all-MiniLM-L6-v2
# Vì đây là model nhẹ, nhanh, lý tưởng để test nhanh pipeline trên máy không có GPU mạnh.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Chọn vector store: ChromaDB (Local, không cần Docker)
VECTOR_STORE = "chromadb"  # "weaviate" | "chromadb" | "faiss"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        if not content.strip():
            continue # Bỏ qua file rỗng (như 105.signed_02.md đang bị lỗi)
            
        doc_type = "legal" if "legal" in md_file.parts else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type}
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i}
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    import chromadb
    import os
    
    # Lưu chroma data vào thư mục gốc của project
    db_path = str(STANDARDIZED_DIR.parent.parent / "chroma_data")
    client = chromadb.PersistentClient(path=db_path)
    
    collection_name = "DrugLawDocs"
    
    # Xoá collection cũ nếu tồn tại
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
        
    # Tạo collection mới
    collection = client.create_collection(name=collection_name)
    
    # Insert chunks
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    documents = [chunk["content"] for chunk in chunks]
    embeddings = [chunk["embedding"] for chunk in chunks]
    metadatas = []
    for chunk in chunks:
        # ChromaDB metadata values must be str, int, float or bool
        meta = {
            "source": str(chunk["metadata"]["source"]),
            "type": str(chunk["metadata"]["type"]),
            "chunk_index": int(chunk["metadata"]["chunk_index"])
        }
        metadatas.append(meta)
        
    # Add theo batch để tối ưu (ChromaDB tự động xử lý dưới lớp client)
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents
    )


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n- Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"- Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"- Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("- Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
