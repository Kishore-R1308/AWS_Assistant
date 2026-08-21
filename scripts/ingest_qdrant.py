from __future__ import annotations

import os
import re
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAG_DIRECTORY = PROJECT_ROOT / "rag_data"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "aws_knowledge",
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

UPLOAD_BATCH_SIZE = 64


# ============================================================
# VALIDATE ENVIRONMENT
# ============================================================

if not QDRANT_URL:
    raise ValueError(
        "QDRANT_URL environment variable is missing."
    )

if not QDRANT_API_KEY:
    raise ValueError(
        "QDRANT_API_KEY environment variable is missing."
    )


# ============================================================
# CONNECT TO QDRANT
# ============================================================

print("=" * 70)
print("AWS ASSISTANT - QDRANT INGESTION")
print("=" * 70)

print(f"Collection: {COLLECTION_NAME}")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print()
print("Loading embedding model...")

model = SentenceTransformer(
    EMBEDDING_MODEL
)

VECTOR_SIZE = model.get_sentence_embedding_dimension()

print(
    f"Embedding dimension: {VECTOR_SIZE}"
)


# ============================================================
# CREATE COLLECTION
# ============================================================

existing_collections = [
    collection.name
    for collection in client.get_collections().collections
]

if COLLECTION_NAME not in existing_collections:

    print()
    print(
        f"Creating collection: "
        f"{COLLECTION_NAME}"
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )

else:

    print()
    print(
        f"Collection already exists: "
        f"{COLLECTION_NAME}"
    )


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:

    if not text:
        return ""

    text = text.replace(
        "\x00",
        " ",
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
# CHUNK TEXT
# ============================================================

def split_text(
    text: str,
) -> list[str]:

    chunks = []

    start = 0

    text_length = len(text)

    while start < text_length:

        end = min(
            start + CHUNK_SIZE,
            text_length,
        )

        chunk = text[
            start:end
        ].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - CHUNK_OVERLAP

    return chunks


# ============================================================
# FIND PDFs
# ============================================================

pdf_files = sorted(
    RAG_DIRECTORY.glob("*.pdf")
)

if not pdf_files:

    raise FileNotFoundError(
        f"No PDF files found in {RAG_DIRECTORY}"
    )


print()
print("=" * 70)
print("PDF FILES")
print("=" * 70)

for pdf in pdf_files:

    print(
        f"- {pdf.name}"
    )


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

all_chunks = []

global_id = 1


for pdf_path in pdf_files:

    print()
    print("=" * 70)
    print(
        f"Processing: "
        f"{pdf_path.name}"
    )
    print("=" * 70)

    reader = PdfReader(
        str(pdf_path)
    )

    total_pages = len(
        reader.pages
    )

    print(
        f"Pages: {total_pages}"
    )

    for page_index, page in enumerate(
        reader.pages
    ):

        text = page.extract_text()

        text = clean_text(
            text or ""
        )

        if not text:
            continue

        chunks = split_text(
            text
        )

        for chunk in chunks:

            all_chunks.append(
                {
                    "id": global_id,
                    "text": chunk,
                    "metadata": {
                        "source": pdf_path.name,
                        "document_name": pdf_path.stem,
                        "page": page_index + 1,
                        "file_type": "pdf",
                    },
                }
            )

            global_id += 1

        if (
            (page_index + 1) % 100 == 0
        ):

            print(
                f"Processed "
                f"{page_index + 1}/"
                f"{total_pages} pages"
            )


# ============================================================
# EMBED + UPLOAD
# ============================================================

print()
print("=" * 70)
print("CREATING EMBEDDINGS")
print("=" * 70)

print(
    f"Total chunks: "
    f"{len(all_chunks)}"
)


for start in range(
    0,
    len(all_chunks),
    UPLOAD_BATCH_SIZE,
):

    batch = all_chunks[
        start:
        start + UPLOAD_BATCH_SIZE
    ]

    texts = [
        item["text"]
        for item in batch
    ]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    points = []

    for item, vector in zip(
        batch,
        embeddings,
    ):

        points.append(
            PointStruct(
                id=item["id"],
                vector=vector.tolist(),
                payload={
                    "text": item["text"],
                    **item["metadata"],
                },
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    uploaded = min(
        start + UPLOAD_BATCH_SIZE,
        len(all_chunks),
    )

    print(
        f"Uploaded "
        f"{uploaded}/"
        f"{len(all_chunks)} chunks"
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("QDRANT INGESTION COMPLETE")
print("=" * 70)

print(
    f"Collection: "
    f"{COLLECTION_NAME}"
)

print(
    f"Total chunks: "
    f"{len(all_chunks)}"
)

print()
print(
    "Your AWS documents are now "
    "stored in Qdrant Cloud."
)