from __future__ import annotations

import hashlib
import os
import re
from typing import Iterable

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# QDRANT CONFIGURATION
# ============================================================

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "aws_knowledge",
)

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

VECTOR_SIZE = 384


# ============================================================
# RAG CONFIGURATION
# ============================================================

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180

DEFAULT_RETRIEVAL_K = 8


# ============================================================
# GLOBAL OBJECTS
# ============================================================

_embeddings = None
_qdrant = None
_splitter = None


# ============================================================
# VALIDATE QDRANT CONFIG
# ============================================================

def validate_qdrant_config():
    if not QDRANT_URL:
        raise RuntimeError(
            "QDRANT_URL environment variable is not configured."
        )

    if not QDRANT_API_KEY:
        raise RuntimeError(
            "QDRANT_API_KEY environment variable is not configured."
        )


# ============================================================
# EMBEDDINGS
# ============================================================

def get_embeddings():
    """
    Load the embedding model only once.
    """

    global _embeddings

    if _embeddings is None:

        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={
                "device": "cpu",
            },
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 32,
            },
        )

    return _embeddings


# ============================================================
# QDRANT CLIENT
# ============================================================

def get_qdrant():

    global _qdrant

    if _qdrant is None:

        validate_qdrant_config()

        _qdrant = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )

    return _qdrant


# ============================================================
# CREATE / VERIFY COLLECTION
# ============================================================

def ensure_collection():

    client = get_qdrant()

    collections = (
        client.get_collections()
        .collections
    )

    collection_names = [
        collection.name
        for collection in collections
    ]

    if COLLECTION_NAME not in collection_names:

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )


# ============================================================
# TEXT SPLITTER
# ============================================================

def get_splitter():

    global _splitter

    if _splitter is None:

        _splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "; ",
                ", ",
                " ",
                "",
            ],
        )

    return _splitter


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:

    text = text.replace(
        "\x00",
        " ",
    )

    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# ============================================================
# DOCUMENT ID
# ============================================================

def make_document_id(
    source: str,
    page: int | None,
    chunk_index: int,
    text: str,
) -> str:

    raw = (
        f"{source}|"
        f"{page}|"
        f"{chunk_index}|"
        f"{text}"
    )

    digest = hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()

    # Qdrant accepts UUID strings as point IDs.
    # Convert deterministic SHA256 into UUID format.
    return (
        f"{digest[:8]}-"
        f"{digest[8:12]}-"
        f"{digest[12:16]}-"
        f"{digest[16:20]}-"
        f"{digest[20:32]}"
    )


# ============================================================
# SPLIT DOCUMENTS
# ============================================================

def split_documents(
    documents: Iterable[Document],
) -> list[Document]:

    splitter = get_splitter()

    chunks: list[Document] = []

    for document in documents:

        text = normalize_text(
            document.page_content
        )

        if not text:
            continue

        source = document.metadata.get(
            "source",
            "unknown",
        )

        page = document.metadata.get(
            "page",
            None,
        )

        split_texts = splitter.split_text(
            text
        )

        for chunk_index, chunk_text in enumerate(
            split_texts
        ):

            chunk_text = chunk_text.strip()

            if not chunk_text:
                continue

            metadata = {
                **document.metadata,
                "source": source,
                "page": page,
                "chunk_index": chunk_index,
                "chunk_size": len(chunk_text),
            }

            metadata["chunk_id"] = make_document_id(
                source,
                page,
                chunk_index,
                chunk_text,
            )

            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata=metadata,
                )
            )

    return chunks


# ============================================================
# ADD DOCUMENTS
# ============================================================

def add_documents(
    documents: list[Document],
    batch_size: int = 32,
) -> int:

    if not documents:
        return 0

    ensure_collection()

    chunks = split_documents(
        documents
    )

    if not chunks:
        return 0

    embeddings = get_embeddings()
    client = get_qdrant()

    total = len(chunks)

    for start in range(
        0,
        total,
        batch_size,
    ):

        batch = chunks[
            start:start + batch_size
        ]

        texts = [
            document.page_content
            for document in batch
        ]

        vectors = embeddings.embed_documents(
            texts
        )

        points = []

        for document, vector in zip(
            batch,
            vectors,
        ):

            points.append(
                PointStruct(
                    id=document.metadata[
                        "chunk_id"
                    ],
                    vector=vector,
                    payload={
                        "text": document.page_content,
                        **document.metadata,
                    },
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

        print(
            f"Indexed "
            f"{min(start + batch_size, total)}"
            f"/{total} chunks"
        )

    return total


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(
    query: str,
    k: int = DEFAULT_RETRIEVAL_K,
) -> list[Document]:

    ensure_collection()

    embeddings = get_embeddings()
    client = get_qdrant()

    query_vector = embeddings.embed_query(
        query
    )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=k,
        with_payload=True,
    )

    documents = []

    for result in results.points:

        payload = result.payload or {}

        text = payload.get(
            "text",
            "",
        )

        metadata = {
            key: value
            for key, value in payload.items()
            if key != "text"
        }

        documents.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )

    return documents


# ============================================================
# RETRIEVE CONTEXT
# ============================================================

def retrieve_context(
    query: str,
    k: int = DEFAULT_RETRIEVAL_K,
) -> str:

    documents = retrieve_documents(
        query,
        k=k,
    )

    if not documents:

        return (
            "No relevant AWS documentation "
            "was found."
        )

    parts = []

    for index, document in enumerate(
        documents,
        1,
    ):

        source = document.metadata.get(
            "source",
            "Unknown source",
        )

        page = document.metadata.get(
            "page",
            "Unknown page",
        )

        parts.append(
            f"SOURCE {index}\n"
            f"Document: {source}\n"
            f"Page: {page}\n"
            f"Content:\n"
            f"{document.page_content}"
        )

    return "\n\n---\n\n".join(parts)


# ============================================================
# RETRIEVED SOURCES
# ============================================================

def get_retrieved_sources(
    query: str,
    k: int = DEFAULT_RETRIEVAL_K,
) -> list[dict]:

    documents = retrieve_documents(
        query,
        k=k,
    )

    sources = []

    for document in documents:

        sources.append(
            {
                "source": document.metadata.get(
                    "source"
                ),
                "page": document.metadata.get(
                    "page"
                ),
                "chunk_id": document.metadata.get(
                    "chunk_id"
                ),
            }
        )

    return sources