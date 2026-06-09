import sqlite3
import requests
import tempfile
import os
import pandas as pd

from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker import HybridChunker
from transformers import AutoTokenizer
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode

EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "telecom_hybrid"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/tnahddisttud/sample-doc/main/data"
PDF_URL         = f"{GITHUB_RAW_BASE}/telecom_guide.pdf"
CSV_URL         = f"{GITHUB_RAW_BASE}/faq.csv"


# Parse tickets.db (SQLite)
# 20 support tickets → LangChain Documents

def load_tickets(db_path: str) -> list[Document]:
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("""
        SELECT ticket_id, category, issue_type, description, resolution
        FROM tickets
    """)
    docs = []
    for ticket_id, category, issue_type, description, resolution in cur.fetchall():
        content = (
            f"Issue type: {issue_type}\n"
            f"Description: {description}\n"
            f"Resolution: {resolution}"
        )
        docs.append(Document(
            page_content=content,
            metadata={
                "source":    "tickets_db",
                "ticket_id": ticket_id,
                "category":  category,
            }
        ))
    conn.close()
    return docs


# Parse PDF via Docling
# CSV is already in tabular format so use pandas to load it
# telecom_guide.pdf + faq.csv → Documents
# chunking pdf using hybridchunker but no need chunking for csv and db
# because faq.csv -> Each row is already one complete Q&A unit. tickets.db -> Each ticket is one complete issue+resolution unit

def load_pdf(url: str = PDF_URL) -> list[Document]:
    # Step 1: download PDF to a temp file (Docling needs local path)
    response = requests.get(url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(response.content)
        pdf_path = tmp.name

    # Step 2: parse with Docling
    converter = DocumentConverter()
    dl_doc    = converter.convert(pdf_path).document
    os.unlink(pdf_path)  # clean up temp file

    # Step 3: chunk with HybridChunker — token limit aligned to embed model
    tokenizer  = AutoTokenizer.from_pretrained(EMBED_MODEL)
    chunker    = HybridChunker(
        tokenizer=tokenizer,
        max_tokens=128,
        merge_peers=True,   # merge small sibling chunks under same heading
    )
    chunk_iter = chunker.chunk(dl_doc)

    # Step 4: convert to LangChain Documents
    docs = []
    for chunk in chunk_iter:
        docs.append(Document(
            page_content=chunker.contextualize(chunk=chunk),  # updated
            metadata={"source": "telecom_guide"},
        ))
    return docs


def load_csv(url: str = CSV_URL) -> list[Document]:
    df = pd.read_csv(url)
    docs = []
    for _, row in df.iterrows():
        content = f"Question: {row['question']}\nAnswer: {row['answer']}"
        docs.append(Document(
            page_content=content,
            metadata={
                "source":   "faq_csv",
                "category": row["category"],
                "id":       row["id"],
            }
        ))
    return docs

#

def build_vectorstore(all_docs: list[Document]) -> QdrantVectorStore:
    dense_embeddings  = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
    )
    sparse_embeddings = FastEmbedSparse(model_name="Prithivida/Splade_PP_en_v1")

    vectorstore = QdrantVectorStore.from_documents(
        documents=all_docs,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        path="/tmp/telecom_qdrant",
        collection_name=COLLECTION_NAME,
        retrieval_mode=RetrievalMode.HYBRID,
    )
    return vectorstore


if __name__ == "__main__":
    print("Loading sources...")
    ticket_docs = load_tickets("data/tickets.db")
    pdf_docs    = load_pdf()
    csv_docs    = load_csv()
    all_docs    = ticket_docs + pdf_docs + csv_docs
    print(f"  Tickets  : {len(ticket_docs)}")
    print(f"  PDF chunks: {len(pdf_docs)}")
    print(f"  FAQ docs : {len(csv_docs)}")
    print(f"  Total    : {len(all_docs)}")

    print("\nBuilding hybrid vectorstore...")
    vectorstore = build_vectorstore(all_docs)
    print("Vectorstore built ✓")

    print("\nTesting hybrid retrieval...")
    results = vectorstore.similarity_search("roaming charges Spain", k=2)
    for r in results:
        print(f"\n  [{r.metadata.get('source')} | {r.metadata.get('ticket_id','')}]")
        print(f"  {r.page_content[:150]}")

    print("\nPhase 2 complete ✓")