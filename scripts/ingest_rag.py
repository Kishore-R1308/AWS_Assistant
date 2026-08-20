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

# Make backend package available
sys.path.insert(0, str(PROJECT_ROOT))


from backend.rag import add_documents


# ============================================================
# CONFIGURATION
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".pdf",
}

# Number of pages kept in memory before sending them to Chroma
PAGE_BATCH_SIZE = 25

# Number of chunks embedded at once
EMBEDDING_BATCH_SIZE = 32

# Maximum number of pages processed from each PDF.
# This keeps Railway resource usage manageable.
MAX_PAGES_PER_PDF = 300


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    """
    Clean extracted PDF text while preserving useful
    paragraph and line boundaries.
    """

    if not text:
        return ""

    # Remove null characters
    text = text.replace("\x00", " ")

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive spaces/tabs
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    # Remove excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# ============================================================
# INGEST ONE PDF
# ============================================================

def ingest_pdf(pdf_path: Path) -> int:
    """
    Process one PDF incrementally.

    Important:
    We use lazy_load() instead of load() so that the entire
    PDF is NOT loaded into memory.

    Only PAGE_BATCH_SIZE pages are kept in memory at once.
    """

    print()
    print("=" * 70)
    print(f"Loading PDF: {pdf_path.name}")
    print("=" * 70)

    try:
        loader = PyMuPDFLoader(
            str(pdf_path)
        )

    except Exception as exc:
        print(
            f"ERROR creating PDF loader: {exc}"
        )
        return 0

    page_batch: list[Document] = []

    total_pages = 0
    total_chunks = 0

    try:

        # Process PDF page-by-page
        for index, page_document in enumerate(
            loader.lazy_load()
        ):

            # Stop after MAX_PAGES_PER_PDF
            if index >= MAX_PAGES_PER_PDF:

                print()
                print(
                    f"Reached page limit of "
                    f"{MAX_PAGES_PER_PDF} "
                    f"for {pdf_path.name}"
                )

                break

            total_pages += 1

            # ------------------------------------------------
            # Clean text
            # ------------------------------------------------

            text = clean_text(
                page_document.page_content
            )

            # Skip empty pages
            if not text:
                continue

            # ------------------------------------------------
            # Page number
            # ------------------------------------------------

            original_page = (
                page_document.metadata.get(
                    "page"
                )
            )

            if original_page is not None:

                page_number = (
                    int(original_page) + 1
                )

            else:

                page_number = total_pages

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            metadata = {
                "source": pdf_path.name,
                "file_path": str(pdf_path),
                "document_name": pdf_path.stem,
                "page": page_number,
                "file_type": "pdf",
            }

            # ------------------------------------------------
            # Create document
            # ------------------------------------------------

            page_batch.append(
                Document(
                    page_content=text,
                    metadata=metadata,
                )
            )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if total_pages % 100 == 0:

                print(
                    f"Processed pages: "
                    f"{total_pages}/"
                    f"{MAX_PAGES_PER_PDF}"
                )

            # ------------------------------------------------
            # Send batch to Chroma
            # ------------------------------------------------

            if len(page_batch) >= PAGE_BATCH_SIZE:

                chunks_indexed = add_documents(
                    page_batch,
                    batch_size=EMBEDDING_BATCH_SIZE,
                )

                total_chunks += chunks_indexed

                print(
                    f"Indexed pages through "
                    f"{total_pages} | "
                    f"Total chunks: "
                    f"{total_chunks}"
                )

                # Release memory
                page_batch.clear()

        # ----------------------------------------------------
        # Process remaining pages
        # ----------------------------------------------------

        if page_batch:

            chunks_indexed = add_documents(
                page_batch,
                batch_size=EMBEDDING_BATCH_SIZE,
            )

            total_chunks += chunks_indexed

            page_batch.clear()

        print()
        print("=" * 70)
        print(
            f"Finished: {pdf_path.name}"
        )
        print(
            f"Pages processed: {total_pages}"
        )
        print(
            f"Chunks indexed: {total_chunks}"
        )
        print("=" * 70)

    except Exception as exc:

        print()
        print("=" * 70)
        print(
            "ERROR DURING PDF INGESTION"
        )
        print("=" * 70)

        print(
            f"PDF: {pdf_path.name}"
        )

        print(
            f"Error: {exc}"
        )

        raise

    return total_chunks


# ============================================================
# FIND PDFs
# ============================================================

def find_pdf_files() -> list[Path]:
    """
    Find all PDF files in rag_data.
    """

    if not RAG_DIRECTORY.exists():

        print()
        print(
            "ERROR: RAG directory does not exist:"
        )

        print(
            RAG_DIRECTORY
        )

        return []

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

    return pdf_files


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 70)
    print(
        "AWS MONITORING AGENT - RAG INGESTION"
    )
    print("=" * 70)

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Knowledge directory: "
        f"{RAG_DIRECTORY}"
    )

    print(
        f"Maximum pages per PDF: "
        f"{MAX_PAGES_PER_PDF}"
    )

    print(
        f"Page batch size: "
        f"{PAGE_BATCH_SIZE}"
    )

    print(
        f"Embedding batch size: "
        f"{EMBEDDING_BATCH_SIZE}"
    )

    # --------------------------------------------------------
    # Find PDFs
    # --------------------------------------------------------

    pdf_files = find_pdf_files()

    if not pdf_files:

        print()
        print(
            "ERROR: No PDF files found."
        )

        print(
            "Place PDFs inside:"
        )

        print(
            RAG_DIRECTORY
        )

        return

    # --------------------------------------------------------
    # Display PDFs
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PDF FILES FOUND")
    print("=" * 70)

    for index, pdf in enumerate(
        pdf_files,
        start=1,
    ):

        print(
            f"{index}. {pdf.name}"
        )

    # --------------------------------------------------------
    # Process PDFs ONE AT A TIME
    # --------------------------------------------------------

    total_chunks = 0

    for pdf_path in pdf_files:

        chunks = ingest_pdf(
            pdf_path
        )

        total_chunks += chunks

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "RAG INDEXING COMPLETE"
    )
    print("=" * 70)

    print(
        f"PDFs processed: "
        f"{len(pdf_files)}"
    )

    print(
        f"Total chunks indexed: "
        f"{total_chunks}"
    )

    print()
    print(
        "Documents are now available "
        "in ChromaDB."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()