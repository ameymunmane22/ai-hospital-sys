# hybrid search workflow using vector search and keyword search using BM25, performing reranking using cross encoder
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

# Fix Windows console charmap encoding errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

# Prevent network warning logs from LangSmith when offline
if os.getenv("LANGCHAIN_TRACING_V2") == "true":
    try:
        import socket
        socket.create_connection(("api.smith.langchain.com", 443), timeout=0.5)
    except OSError:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"


def is_online() -> bool:
    """Check whether HuggingFace hub is reachable."""
    try:
        import socket
        socket.create_connection(("huggingface.co", 443), timeout=0.5)
        return True
    except OSError:
        return False


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Initialize HuggingFace embedding model with offline tolerance."""
    emb_kwargs: dict = {}
    if not is_online():
        emb_kwargs["local_files_only"] = True

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs=emb_kwargs,
        encode_kwargs={"normalize_embeddings": True},
    )


def get_cross_encoder_model(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> CrossEncoder:
    """Initialize CrossEncoder model for reranking, with local cache fallback."""
    if is_online():
        try:
            return CrossEncoder(model_name)
        except Exception:
            pass

    # Fast offline fallback using locally cached sentence-transformer model in cross-encoder mode
    return CrossEncoder(
        "sentence-transformers/all-MiniLM-L6-v2",
        local_files_only=True,
    )


class HybridSearchEngine:
    """
    Hybrid Search Engine combining:
    1. Dense semantic search via ChromaDB vector store
    2. Sparse lexical search via BM25
    3. Cross-Encoder reranking for precision scoring
    """

    def __init__(
        self,
        docs: Optional[list[Document]] = None,
        chroma_dir: Path = CHROMA_DIR,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.chroma_dir = chroma_dir
        self.embeddings = get_embedding_model()

        # Load chunks if not provided directly
        if docs is None:
            try:
                from main import all_chunks
                self.documents = all_chunks
            except Exception:
                self.documents = []
        else:
            self.documents = docs

        # Initialize ChromaDB Vector Store
        if self.chroma_dir.exists():
            self.vector_store = Chroma(
                persist_directory=str(self.chroma_dir),
                embedding_function=self.embeddings,
            )
        elif self.documents:
            self.vector_store = Chroma.from_documents(
                documents=self.documents,
                embedding=self.embeddings,
                persist_directory=str(self.chroma_dir),
            )
        else:
            raise ValueError("No indexed vector store found and no documents provided.")

        # Initialize BM25 Sparse Retriever
        if self.documents:
            self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        else:
            # Reconstruct documents from Chroma if not directly in memory
            chroma_data = self.vector_store.get()
            reconstructed_docs = [
                Document(
                    page_content=text,
                    metadata=chroma_data["metadatas"][i] if chroma_data.get("metadatas") else {},
                )
                for i, text in enumerate(chroma_data.get("documents", []))
            ]
            self.documents = reconstructed_docs
            self.bm25_retriever = BM25Retriever.from_documents(self.documents)

        # Initialize Cross-Encoder Reranker
        self.cross_encoder = get_cross_encoder_model(cross_encoder_model)

    def dense_search(self, query: str, k: int = 10) -> list[Document]:
        """Perform dense semantic vector retrieval using ChromaDB."""
        return self.vector_store.similarity_search(query, k=k)

    def sparse_search(self, query: str, k: int = 10) -> list[Document]:
        """Perform sparse keyword retrieval using BM25."""
        self.bm25_retriever.k = k
        return self.bm25_retriever.invoke(query)

    def fuse_and_deduplicate(
        self,
        dense_results: list[Document],
        sparse_results: list[Document],
    ) -> list[Document]:
        """Merge dense and sparse document pools and remove duplicate chunks."""
        seen_contents: set[str] = set()
        unique_docs: list[Document] = []

        for doc in dense_results + sparse_results:
            key = doc.page_content.strip()
            if key not in seen_contents:
                seen_contents.add(key)
                unique_docs.append(doc)

        return unique_docs

    def rerank(
        self,
        query: str,
        candidates: list[Document],
        top_n: int = 5,
    ) -> list[tuple[Document, float]]:
        """Rerank candidate documents against the query using CrossEncoder."""
        if not candidates:
            return []

        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.cross_encoder.predict(pairs)

        # Pair candidates with their rerank scores
        scored_candidates = [
            (doc, float(score)) for doc, score in zip(candidates, scores)
        ]

        # Sort descending by rerank score
        ranked_docs = sorted(
            scored_candidates, key=lambda item: item[1], reverse=True
        )

        return ranked_docs[:top_n]

    def search(
        self,
        query: str,
        top_k_candidates: int = 10,
        top_n_reranked: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Execute the complete hybrid search workflow:
        Dense Search + BM25 Sparse Search -> Deduplication -> Cross-Encoder Rerank
        """
        dense_docs = self.dense_search(query, k=top_k_candidates)
        sparse_docs = self.sparse_search(query, k=top_k_candidates)

        fused_docs = self.fuse_and_deduplicate(dense_docs, sparse_docs)
        reranked_results = self.rerank(query, fused_docs, top_n=top_n_reranked)

        dept_lookup = {
            "HSP-SOP": "Hospital Operations & Governance",
            "HSP-CLN": "Clinical Operations & Medical Affairs",
            "HSP-NUR": "Nursing Department",
            "HSP-ADM": "Patient Admissions & Registration",
            "HSP-DSC": "Discharge Planning & Care Coordination",
        }

        formatted_results: list[dict[str, Any]] = []
        for rank, (doc, score) in enumerate(reranked_results, start=1):
            source = doc.metadata.get("source", "Hospital Knowledge Base")
            doc_id = doc.metadata.get("document_id")

            # Extract doc_id from text if not in metadata
            if not doc_id or doc_id == "N/A":
                id_match = re.search(r"\b(HSP-[A-Z0-9]+)\b", doc.page_content)
                doc_id = id_match.group(1) if id_match else "HSP-DOC"

            department = doc.metadata.get("department")
            if not department or department == "N/A":
                prefix = "-".join(doc_id.split("-")[:2])
                department = dept_lookup.get(prefix, "Hospital Administration")

            formatted_results.append(
                {
                    "rank": rank,
                    "score": round(score, 4),
                    "source": Path(source).name if str(source) else "Hospital Document",
                    "document_id": doc_id,
                    "department": department,
                    "type": doc.metadata.get("type", "content"),
                    "content": doc.page_content,
                }
            )

        return formatted_results


# Global engine instance for tool usage
_engine_instance: Optional[HybridSearchEngine] = None


def get_hybrid_engine() -> HybridSearchEngine:
    """Singleton getter for HybridSearchEngine."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = HybridSearchEngine()
    return _engine_instance


@tool
def hybrid_search_tool(query: str) -> str:
    """
    Execute hybrid search (Dense vector + BM25 keyword + Cross-Encoder reranking)
    across the hospital knowledge base policies, SOPs, and clinical protocols.
    """
    engine = get_hybrid_engine()
    results = engine.search(query, top_k_candidates=10, top_n_reranked=4)

    if not results:
        return "No relevant hospital knowledge base documents found."

    output_lines: list[str] = [
        f"Hybrid Search Results for query: '{query}'\n"
    ]
    for r in results:
        header = (
            f"[Rank {r['rank']} | Score: {r['score']} | Source: {r['source']} | "
            f"Doc ID: {r['document_id']} | Dept: {r['department']}]"
        )
        output_lines.append(f"{header}\n{r['content']}\n{'-' * 40}")

    return "\n".join(output_lines)


hybrid_search = hybrid_search_tool


if __name__ == "__main__":
    test_query = "What is the procedure and documentation required for patient admission and bed allocation?"
    print("=" * 70)
    print("Testing Hybrid Search (Dense Chroma + BM25 + Cross-Encoder Reranker)")
    print("=" * 70)
    print(f"Query: {test_query}\n")

    engine = get_hybrid_engine()
    ranked_records = engine.search(test_query, top_k_candidates=8, top_n_reranked=3)

    for item in ranked_records:
        print(f"Rank {item['rank']} | Score: {item['score']} | Source: {item['source']}")
        print(f"Content Snippet: {item['content'][:200]}...\n")
