from pathlib import Path

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from vectordb import VectorDB


# ============================================================
# CONFIG
# ============================================================

PDF_FOLDER = Path(".")

DB_FILE = "vector_db.json"

MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

EMBED_BATCH_SIZE = 32


# ============================================================
# TEXT CHUNKING
# ============================================================

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):

    text = text.strip()

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf(pdf_path):

    reader = PdfReader(str(pdf_path))

    records = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        if not text:
            continue

        chunks = chunk_text(text)

        for chunk_number, chunk in enumerate(chunks):

            records.append({
                "text": chunk,
                "metadata": {
                    "source": pdf_path.name,
                    "page": page_number,
                    "chunk": chunk_number
                }
            })

    return records


# ============================================================
# MAIN INGESTION
# ============================================================

def main():

    print("=" * 30)
    print("VECTOR DB INGESTION")
    print("=" * 30)

    # --------------------------------------------------------
    # 1. Load embedding model
    # --------------------------------------------------------

    print("\nLoading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Model loaded.")

    # --------------------------------------------------------
    # 2. Create fresh VectorDB
    # --------------------------------------------------------

    db = VectorDB()

    print("\nVectorDB initialized.")
    print("Dimension:", db.dimension)

    # --------------------------------------------------------
    # 3. Find PDFs
    # --------------------------------------------------------

    pdf_files = sorted(
        PDF_FOLDER.glob("*.pdf")
    )

    if not pdf_files:

        print("\nNo PDF files found.")

        return

    print("\nPDFs found:")

    for pdf in pdf_files:
        print(" -", pdf.name)

    # --------------------------------------------------------
    # 4. Extract and chunk PDFs
    # --------------------------------------------------------

    records = []

    for pdf_path in pdf_files:

        print(
            f"\nProcessing: {pdf_path.name}"
        )

        pdf_records = extract_pdf(pdf_path)

        print(
            "Chunks extracted:",
            len(pdf_records)
        )

        records.extend(pdf_records)

    if not records:

        print("\nNo text was extracted.")

        return

    print(
        "\nTotal chunks:",
        len(records)
    )

    # --------------------------------------------------------
    # 5. Generate embeddings
    # --------------------------------------------------------

    texts = [
        record["text"]
        for record in records
    ]

    print("\nGenerating embeddings...")

    vectors = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True
    )

    vectors = np.asarray(
        vectors,
        dtype=np.float32
    )

    print(
        "Embedding matrix:",
        vectors.shape
    )

    # --------------------------------------------------------
    # 6. Prepare metadata
    # --------------------------------------------------------

    metadata = [
        record["metadata"]
        for record in records
    ]

    documents = texts

    # --------------------------------------------------------
    # 7. Bulk insertion
    # --------------------------------------------------------

    print("\nAdding batch to VectorDB...")

    result = db.add_batch(
        documents,
        vectors,
        metadata
    )

    # --------------------------------------------------------
    # 8. Print insertion result
    # --------------------------------------------------------

    print("\n" + "=" * 30)
    print("INGESTION RESULT")
    print("=" * 30)

    print(
        "Added:",
        result["added"]
    )

    print(
        "Failed:",
        result["failed"]
    )

    if result["errors"]:

        print("\nErrors:")

        for error in result["errors"]:
            print(error)

    # --------------------------------------------------------
    # 9. Save database
    # --------------------------------------------------------

    db.save(DB_FILE)

    print("\nDatabase saved to:")
    print(DB_FILE)

    # --------------------------------------------------------
    # 10. Final state
    # --------------------------------------------------------

    print("\n" + "=" * 30)
    print("INGESTION COMPLETE")
    print("=" * 30)

    print(
        "Documents:",
        len(db.document)
    )

    print(
        "HNSW count:",
        db.index.get_current_count()
    )

    print(
        "Next HNSW ID:",
        db.next_index_id
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()