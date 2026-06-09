from src.ingest import load_tickets, load_pdf, load_csv, build_vectorstore
from src.chain import build_rag_chain, build_reranking_chain
from src.sql_chain import build_db, build_sql_chain, TICKETS_URL
import requests, tempfile, os

def ask(chain, question: str, label: str = ""):
    if label:
        print(f"\n{'='*10} {label} {'='*10}")
    print(f"Q: {question}")
    result = chain.invoke({"input": question})
    print(f"A: {result['answer']}")
    print("Sources:")
    for doc in result["context"]:
        src = doc.metadata.get("source")
        tid = doc.metadata.get("ticket_id", "")
        print(f"  - {src} {tid}".strip())

def ask_sql(chain, question: str):
    print(f"\nQ: {question}")
    answer = chain(question)
    print(f"A: {answer}")
    print("-" * 60)

if __name__ == "__main__":
    # --- Vector RAG (Phase 3 + 4) ---
    print("Loading sources...")
    ticket_docs = load_tickets("data/tickets.db")
    pdf_docs    = load_pdf()
    csv_docs    = load_csv()
    all_docs    = ticket_docs + pdf_docs + csv_docs
    vectorstore = build_vectorstore(all_docs)

    chain           = build_rag_chain(vectorstore)
    reranking_chain = build_reranking_chain(vectorstore)

    question = "What should I do if I get unexpected roaming charges?"
    ask(chain,           question, label="Without reranking")
    ask(reranking_chain, question, label="With reranking")

    # --- SQL RAG (Phase 5) ---
    print("\n" + "="*40)
    print("SQL RAG")
    print("="*40)

    # --- SQL RAG (Phase 5) ---
    print("\n" + "="*40)
    print("SQL RAG")
    print("="*40)

    db        = build_db("data/tickets.db")
    sql_chain = build_sql_chain(db)

    ask_sql(sql_chain, "How many tickets are in each category?")
    ask_sql(sql_chain, "What is the most common issue type in connectivity?")
    ask_sql(sql_chain, "How many tickets have been resolved vs escalated?")

    print("\nPhase 5 complete ✓")