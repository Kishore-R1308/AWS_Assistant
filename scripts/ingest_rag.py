from __future__ import annotations

import re
import sys
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAG_DIRECTORY = PROJECT_ROOT / "rag_data"

sys.path.insert(0, str(PROJECT_ROOT))

from backend.rag import add_documents


# ============================================================
# CONFIGURATION
# ============================================================

SUPPORTED_EXTENSIONS = {".pdf"}

# Number of PDF pages kept in memory at once
PAGE_BATCH_SIZE = 25

# Embedding batch size
EMBEDDING_BATCH_SIZE = 32


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# INGEST ONE PDF
# ============================================================

def ingest_pdf(pdf_path: Path) -> int:

    print()
    print("=" * 70)
    print(f"Loading PDF: {pdf_path.name}")
    print("=" * 70)

    try:
        loader = PyMuPDFLoader(str(pdf_path))
    except Exception as exc:
        print(f"ERROR creating loader: {exc}")
        return 0

    batch: list[Document] = []
    total_pages = 0
    total_chunks = 0

    try:
        # IMPORTANT:
        # lazy_load() processes pages one at a time.
        for index, page_document in enumerate(loader.lazy_load()):

            total_pages += 1

            text = clean_text(page_document.page_content)

            if not text:
                continue

            original_page = page_document.metadata.get("page")

            if original_page is not None:
                page_number = int(original_page) + 1
            else:
                page_number = total_pages

            metadata = {
                "source": pdf_path.name,
                "file_path": str(pdf_path),
                "document_name": pdf_path.stem,
                "page": page_number,
                "file_type": "pdf",
            }

            batch.append(
                Document(
                    page_content=text,
                    metadata=metadata,
                )
            )

            if total_pages % 100 == 0:
                print(
                    f"  Processed pages: "
                    f"{total_pages}"
                )

            # Send a small batch to Chroma
            if len(batch) >= PAGE_BATCH_SIZE:

                chunks = add_documents(
                    batch,
                    batch_size=EMBEDDING_BATCH_SIZE,
                )

                total_chunks += chunks

                print(
                    f"  Indexed pages through "
                    f"{total_pages} | "
                    f"Total chunks: {total_chunks}"
                )

                # Release memory
                batch.clear()

        # Process remaining pages
        if batch:

            chunks = add_documents(
                batch,
                batch_size=EMBEDDING_BATCH_SIZE,
            )

            total_chunks += chunks
            batch.clear()

    except Exception as exc:

        print()
        print("=" * 70)
        print("ERROR DURING PDF INGESTION")
        print("=" * 70)
        print(f"PDF: {pdf_path.name}")
        print(f"Error: {exc}")
        raise

    print()
    print(
        f"Finished {pdf_path.name}: "
        f"{total_pages} pages, "
        f"{total_chunks} chunks"
    )

    return total_chunks


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 70)
    print("AWS MONITORING AGENT - RAG INGESTION")
    print("=" * 70)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Knowledge directory: {RAG_DIRECTORY}")

    if not RAG_DIRECTORY.exists():
        raise FileNotFoundError(
            f"RAG directory does not exist: {RAG_DIRECTORY}"
        )

    pdf_files = sorted(
        [
            path
            for path in RAG_DIRECTORY.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        ]
    )

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {RAG_DIRECTORY}"
        )

    print()
    print("=" * 70)
    print("PDF FILES FOUND")
    print("=" * 70)

    for index, pdf in enumerate(pdf_files, start=1):
        print(f"{index}. {pdf.name}")

    total_chunks = 0

    # Process ONE PDF at a time
    for pdf_path in pdf_files:

        chunks = ingest_pdf(pdf_path)

        total_chunks += chunks

    print()
    print("=" * 70)
    print("RAG INDEXING COMPLETE")
    print("=" * 70)

    print(f"PDFs processed: {len(pdf_files)}")
    print(f"Total chunks indexed: {total_chunks}")

    print()
    print("All documents are now available in ChromaDB.")


if __name__ == "__main__":
    main()