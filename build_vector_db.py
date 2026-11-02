"""
RAG Pipeline - Build FAISS Vector Database from Textbooks
=========================================================
Reads all PDFs from textbooks/<COURSE>/<file>.pdf
Embeds chunks using all-MiniLM-L6-v2
Stores FAISS index + metadata to vector_db/
"""

import os
import json
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import PyPDF2

# ── Config ────────────────────────────────────────────────────────────────────
TEXTBOOKS_DIR  = Path("textbooks")
VECTOR_DB_DIR  = Path("vector_db")
EMBED_MODEL    = "all-MiniLM-L6-v2"
CHUNK_SIZE     = 500          # characters per chunk
CHUNK_OVERLAP  = 100          # overlap between consecutive chunks
BATCH_SIZE     = 64           # embeddings per batch
# ──────────────────────────────────────────────────────────────────────────────


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF file."""
    text = []
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
    except Exception as e:
        print(f"  ⚠  Could not read {pdf_path.name}: {e}")
    return "\n".join(text)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_vector_db():
    VECTOR_DB_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("  RAG PIPELINE — Building FAISS Vector Database")
    print("=" * 60)
    print(f"  Embedding model : {EMBED_MODEL}")
    print(f"  Chunk size      : {CHUNK_SIZE} chars  (overlap {CHUNK_OVERLAP})")
    print(f"  Source          : {TEXTBOOKS_DIR.resolve()}")
    print(f"  Output          : {VECTOR_DB_DIR.resolve()}")
    print("=" * 60)

    # ── 1. Load embedding model ───────────────────────────────────────────────
    print("\n[1/4] Loading sentence-transformer model...")
    embedder = SentenceTransformer(EMBED_MODEL)
    dim = embedder.get_sentence_embedding_dimension()
    print(f"  ✓ Model loaded  (embedding dim = {dim})")

    # ── 2. Extract & chunk all PDFs ───────────────────────────────────────────
    print("\n[2/4] Extracting text from PDFs...")
    all_chunks   = []   # list of str
    all_metadata = []   # list of dict

    course_dirs = sorted([d for d in TEXTBOOKS_DIR.iterdir() if d.is_dir()])
    if not course_dirs:
        print("  ✗  No course folders found inside textbooks/")
        return

    for course_dir in course_dirs:
        course_code = course_dir.name.upper()
        pdf_files   = sorted(course_dir.glob("*.pdf"))

        if not pdf_files:
            print(f"  ⚠  {course_code}: no PDFs found, skipping")
            continue

        for pdf_path in pdf_files:
            print(f"  → {course_code}/{pdf_path.name}")
            raw_text = extract_text_from_pdf(pdf_path)

            if not raw_text.strip():
                print(f"     ⚠  Empty text extracted, skipping")
                continue

            chunks = chunk_text(raw_text)
            print(f"     ✓  {len(chunks)} chunks")

            for idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    "course_code" : course_code,
                    "source_file" : pdf_path.name,
                    "chunk_index" : idx,
                    "text"        : chunk,
                })

    total_chunks = len(all_chunks)
    print(f"\n  Total chunks to embed: {total_chunks}")

    if total_chunks == 0:
        print("  ✗  No text extracted from any PDF. Aborting.")
        return

    # ── 3. Embed in batches ───────────────────────────────────────────────────
    print("\n[3/4] Generating embeddings (this may take a few minutes)...")
    t0 = time.time()
    all_embeddings = []

    for i in range(0, total_chunks, BATCH_SIZE):
        batch = all_chunks[i : i + BATCH_SIZE]
        vecs  = embedder.encode(batch, show_progress_bar=False,
                                convert_to_numpy=True)
        all_embeddings.append(vecs)
        done = min(i + BATCH_SIZE, total_chunks)
        print(f"  Embedded {done}/{total_chunks} chunks...", end="\r")

    embeddings = np.vstack(all_embeddings).astype("float32")
    print(f"\n  ✓ Embeddings done in {time.time()-t0:.1f}s  "
          f"shape={embeddings.shape}")

    # ── 4. Build & save FAISS index ───────────────────────────────────────────
    print("\n[4/4] Building FAISS index and saving...")

    # Normalise for cosine similarity via inner-product search
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path    = VECTOR_DB_DIR / "faiss_index.bin"
    metadata_path = VECTOR_DB_DIR / "metadata.pkl"
    info_path     = VECTOR_DB_DIR / "db_info.json"

    faiss.write_index(index, str(index_path))
    with open(metadata_path, "wb") as f:
        pickle.dump(all_metadata, f)

    db_info = {
        "total_chunks"  : total_chunks,
        "embedding_dim" : dim,
        "embed_model"   : EMBED_MODEL,
        "chunk_size"    : CHUNK_SIZE,
        "chunk_overlap" : CHUNK_OVERLAP,
        "courses"       : sorted({m["course_code"] for m in all_metadata}),
        "built_at"      : time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(info_path, "w") as f:
        json.dump(db_info, f, indent=2)

    print(f"  ✓ FAISS index   → {index_path}")
    print(f"  ✓ Metadata      → {metadata_path}")
    print(f"  ✓ DB info       → {info_path}")
    print("\n" + "=" * 60)
    print(f"  Vector DB built successfully!")
    print(f"  Courses indexed : {', '.join(db_info['courses'])}")
    print(f"  Total vectors   : {total_chunks}")
    print("=" * 60)


if __name__ == "__main__":
    build_vector_db()
