#import necessary libraries
import sys
import os
import csv
import json
import re

# Fix Windows console charmap encoding errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path
from typing import Literal, Optional, TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

# LangChain, ChromaDB & Groq
from exa_py import Exa
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    CSVLoader,
    DirectoryLoader,
    PyPDFDirectoryLoader,
)
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

# Load environment variables from .env
load_dotenv()


# Ingest all pdf from Knowldege_base file folder
base_dir: Path = Path(__file__).resolve().parent
kb_path: Path = base_dir / "Knowledge_base"
if not kb_path.exists():
    kb_path = base_dir / "Knowldege_base"


def ingest_pdfs_from_folder(folder_path: Path) -> list[Document]:
    """Ingest and return all PDF documents from the specified knowledge base folder."""
    if not folder_path.exists() or not folder_path.is_dir():
        return []
    loader = PyPDFDirectoryLoader(str(folder_path))
    return loader.load()


documents: list[Document] = ingest_pdfs_from_folder(kb_path)


#genrate document level metadata like sectional headings , keywords , entities , department etc in json format and store it in new file 
def generate_document_metadata(
    docs: list[Document],
    output_path: Path = base_dir / "document_metadata.json",
) -> list[dict]:
    """Generate document-level metadata (headings, keywords, entities, department) and save to JSON."""
    if not docs:
        return []

    dept_mapping = {
        "HSP-SOP": "Hospital Operations & Governance",
        "HSP-CLN": "Clinical Operations & Medical Affairs",
        "HSP-NUR": "Nursing Department",
        "HSP-ADM": "Patient Admissions & Registration",
        "HSP-DSC": "Discharge Planning & Care Coordination",
    }

    grouped_docs: dict[str, list[Document]] = {}
    for doc in docs:
        source_key = str(doc.metadata.get("source", ""))
        grouped_docs.setdefault(source_key, []).append(doc)

    metadata_list: list[dict] = []
    for source_path, pages in grouped_docs.items():
        if not pages:
            continue
        first_page_text = pages[0].page_content
        full_text = "\n".join(p.page_content for p in pages)
        file_name = Path(source_path).name

        doc_id_match = re.search(r"Document ID\s*\n\s*([A-Z0-9-]+)", first_page_text)
        doc_id = doc_id_match.group(1).strip() if doc_id_match else "UNKNOWN"

        prefix = "-".join(doc_id.split("-")[:2])
        department = dept_mapping.get(prefix, "Hospital Administration")

        title_match = re.search(r"Page 1\s*\n\s*([^\n]+(?:\n[^\n]+)?)", first_page_text)
        title = " ".join(title_match.group(1).split()) if title_match else file_name

        headings = [h.strip() for h in re.findall(r"(\d{2}\.\s+[^\n]+)", first_page_text)]

        kw_candidates = re.findall(r"\b[A-Za-z]{4,}\b", full_text)
        stop_words = {
            "this", "that", "with", "from", "have", "were", "when", "page",
            "synthetic", "demonstration", "content", "approved", "hospital",
            "policy", "fictional", "document", "sample", "review", "scope"
        }
        word_counts: dict[str, int] = {}
        for word in kw_candidates:
            w_lower = word.lower()
            if w_lower not in stop_words:
                word_counts[w_lower] = word_counts.get(w_lower, 0) + 1
        keywords = sorted(word_counts, key=word_counts.get, reverse=True)[:10]

        entities = list(
            dict.fromkeys(
                re.findall(r"\b(?:HSP-[A-Z0-9-]+|ICU|ED|EMR|EHR|MAR|SOP|SBAR|HIPAA|WHO)\b", full_text)
            )
        )

        doc_meta = {
            "file_name": file_name,
            "document_id": doc_id,
            "title": title,
            "department": department,
            "sectional_headings": headings,
            "keywords": keywords,
            "entities": entities,
        }
        metadata_list.append(doc_meta)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)

    return metadata_list


document_metadata: list[dict] = generate_document_metadata(documents)

# using langchin Recursive Charecter chunking perform on both dataset and genrated metadata
def chunk_dataset_and_metadata(
    dataset_docs: list[Document],
    meta_records: list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> tuple[list[Document], list[Document], list[Document]]:
    """Split dataset documents and generated metadata into chunks using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    metadata_docs: list[Document] = []
    for meta in meta_records:
        headings_str = ", ".join(meta.get("sectional_headings", []))
        keywords_str = ", ".join(meta.get("keywords", []))
        entities_str = ", ".join(meta.get("entities", []))

        content = (
            f"Document Title: {meta.get('title', '')}\n"
            f"Document ID: {meta.get('document_id', '')}\n"
            f"Department: {meta.get('department', '')}\n"
            f"Sectional Headings: {headings_str}\n"
            f"Keywords: {keywords_str}\n"
            f"Entities: {entities_str}"
        )
        metadata_docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": meta.get("file_name", ""),
                    "document_id": meta.get("document_id", ""),
                    "department": meta.get("department", ""),
                    "type": "metadata",
                },
            )
        )

    dataset_chunked: list[Document] = text_splitter.split_documents(dataset_docs)
    metadata_chunked: list[Document] = text_splitter.split_documents(metadata_docs)
    combined_chunks: list[Document] = dataset_chunked + metadata_chunked

    return dataset_chunked, metadata_chunked, combined_chunks


dataset_chunks, metadata_chunks, all_chunks = chunk_dataset_and_metadata(
    documents, document_metadata
)

#using huggingface embedding , make embeddings of chunks of both documents and metadata
emb_kwargs: dict = {}
try:
    import socket
    socket.create_connection(("huggingface.co", 443), timeout=1.0)
except OSError:
    emb_kwargs["local_files_only"] = True

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs=emb_kwargs,
    encode_kwargs={"normalize_embeddings": True},
)


def generate_chunk_embeddings(
    dataset_docs: list[Document],
    meta_docs: list[Document],
    embedding_model: HuggingFaceEmbeddings,
) -> tuple[list[list[float]], list[list[float]]]:
    """Generate vector embeddings for dataset document chunks and metadata chunks."""
    dataset_texts = [doc.page_content for doc in dataset_docs]
    metadata_texts = [doc.page_content for doc in meta_docs]

    dataset_vectors = embedding_model.embed_documents(dataset_texts) if dataset_texts else []
    metadata_vectors = embedding_model.embed_documents(metadata_texts) if metadata_texts else []

    return dataset_vectors, metadata_vectors


dataset_embeddings, metadata_embeddings = generate_chunk_embeddings(
    dataset_chunks, metadata_chunks, embeddings
)

# use chromaDB for indexing both documents embeddings and metadata embedding and store it
def initialize_chroma_db(
    chunks: list[Document],
    embedding_model: HuggingFaceEmbeddings,
    persist_path: Path = base_dir / "chroma_db",
) -> Chroma:
    """Index document and metadata chunks in ChromaDB and persist to disk."""
    persist_path.mkdir(parents=True, exist_ok=True)
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(persist_path),
    )
    return vector_db


vector_store: Chroma = initialize_chroma_db(all_chunks, embeddings)

def initialize_chroma_db(
    chunks: list[Document],
    embedding_model: HuggingFaceEmbeddings,
    persist_path: Path = base_dir / "chroma_db",
) -> Chroma:
    """Index document and metadata chunks in ChromaDB and persist to disk."""
    persist_path.mkdir(parents=True, exist_ok=True)
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(persist_path),
    )
    return vector_db


vector_store: Chroma = initialize_chroma_db(all_chunks, embeddings)

