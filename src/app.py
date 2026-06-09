import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from src.ingest import load_tickets, load_pdf, load_csv, build_vectorstore
from src.chain import build_rag_chain, build_reranking_chain
from src.sql_chain import build_db, build_sql_chain, is_sql_question

# ── Global state — built once at startup ─────────────────────────
print("Starting up — loading sources...")
ticket_docs = load_tickets("data/tickets.db")
pdf_docs    = load_pdf()
csv_docs    = load_csv()
all_docs    = ticket_docs + pdf_docs + csv_docs
print(f"Loaded {len(all_docs)} documents")

print("Building vectorstore...")
vectorstore = build_vectorstore(all_docs)

print("Building chains...")
rag_chain       = build_rag_chain(vectorstore)
reranking_chain = build_reranking_chain(vectorstore)
sql_chain       = build_sql_chain(build_db("data/tickets.db"))
print("Ready ✓")

# ── Response function ─────────────────────────────────────────────
def respond(message: str, history: list, use_reranking: bool):
    if not message.strip():
        return "", history

    if is_sql_question(message):
        chain_label = "SQL RAG"
        answer      = sql_chain(message)
        sources_md  = "_Source: tickets.db (SQL query)_"
    else:
        chain       = reranking_chain if use_reranking else rag_chain
        chain_label = "Vector RAG + Reranking" if use_reranking else "Vector RAG"
        result      = chain.invoke({"input": message})
        answer      = result["answer"]

        seen    = set()
        sources = []
        for doc in result["context"]:
            src = doc.metadata.get("source", "unknown")
            tid = doc.metadata.get("ticket_id", "")
            cat = doc.metadata.get("category", "")
            key = f"{src}_{tid}"
            if key not in seen:
                seen.add(key)
                label = src
                if tid: label += f" · {tid}"
                if cat: label += f" · {cat}"
                sources.append(label)
        sources_md = "_Sources: " + ", ".join(sources) + "_"

    full_response = f"{answer}\n\n{sources_md}\n`[{chain_label}]`"

    history.append({"role": "user",      "content": message})
    history.append({"role": "assistant", "content": full_response})
    return "", history

# ── Gradio UI ─────────────────────────────────────────────────────
with gr.Blocks(title="TelecomCo Support Assistant") as demo:
    gr.Markdown("## TelecomCo Support Assistant")
    gr.Markdown(
        "Ask anything about telecom support. "
        "Analytical questions (counts, stats) use **SQL RAG**. "
        "Everything else uses **Hybrid Vector RAG**."
    )

    chatbot = gr.Chatbot(
        label="Chat",
        height=480,
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="e.g. How do I fix my APN settings? / How many billing tickets?",
            label="Your question",
            scale=8,
            autofocus=True,
        )
        submit_btn = gr.Button("Send", scale=1, variant="primary")

    with gr.Row():
        use_reranking = gr.Checkbox(
            label="Enable reranking (slower but more precise)",
            value=True,
        )
        clear_btn = gr.Button("Clear chat", scale=1)

    with gr.Accordion("Example questions", open=False):
        gr.Examples(
            examples=[
                ["How do I fix my APN settings?"],
                ["My phone shows full bars but can't load websites"],
                ["What should I do about unexpected roaming charges?"],
                ["How many tickets are in each category?"],
                ["How many tickets have been resolved vs escalated?"],
                ["What is the most common connectivity issue?"],
            ],
            inputs=msg,
        )

    # Wire up events
    submit_btn.click(
        respond,
        inputs=[msg, chatbot, use_reranking],
        outputs=[msg, chatbot],
    )
    msg.submit(
        respond,
        inputs=[msg, chatbot, use_reranking],
        outputs=[msg, chatbot],
    )
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])

if __name__ == "__main__":
    demo.launch(share=False)