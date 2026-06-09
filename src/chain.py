import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

load_dotenv()

GROQ_MODEL = "openai/gpt-oss-20b"

def build_llm() -> ChatGroq:
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        max_tokens=1024,
        api_key=os.getenv("GROQ_API_KEY"),
    )


SYSTEM_PROMPT = """You are a TelecomCo support assistant.
Answer the user's question using ONLY the context provided below.
If the context does not contain enough information, say so clearly.
When referencing a support ticket, always mention the ticket ID.
Be concise and factual.

Context:
{context}"""

def build_rag_chain(vectorstore: QdrantVectorStore):
    llm = build_llm()

    # Hybrid retriever — uses both dense + sparse vectors in Qdrant
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    # Chain: retriever → stuff docs into prompt → LLM
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain          = create_retrieval_chain(retriever, combine_docs_chain)
    return rag_chain

def build_reranking_chain(vectorstore: QdrantVectorStore):
    llm = build_llm()

    # Step 1: broad retriever — fetch 10 candidates
    broad_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10},
    )

    # Step 2: cross-encoder reranker — scores each candidate
    # against the query, keeps only top 3
    cross_encoder = HuggingFaceCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=3)

    # Step 3: wrap retriever with reranker
    reranking_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=broad_retriever,
    )

    # Step 4: same RAG chain structure, different retriever
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])
    combine_docs_chain  = create_stuff_documents_chain(llm, prompt)
    reranking_rag_chain = create_retrieval_chain(
        reranking_retriever, combine_docs_chain
    )
    return reranking_rag_chain