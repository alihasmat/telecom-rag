import os
import re
import requests
import tempfile
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_classic.chains import create_sql_query_chain
from langchain_core.prompts import ChatPromptTemplate
from src.chain import build_llm

load_dotenv()

TICKETS_URL = "https://raw.githubusercontent.com/tnahddisttud/sample-doc/main/data/tickets.db"

SYSTEM_PROMPT = """You are a TelecomCo support analytics assistant.
Given a user question and the SQL query result from our tickets database,
provide a clear, concise natural language answer.
Be specific with numbers and facts from the data."""


def clean_sql(raw: str) -> str:
    """Strip markdown fences and preamble — keep only the SQL statement."""
    raw = re.sub(r"```(?:sql)?", "", raw).strip("`").strip()
    if "SQLQuery:" in raw:
        raw = raw.split("SQLQuery:")[-1].strip()
    return raw


def build_db(db_path: str) -> SQLDatabase:
    return SQLDatabase.from_uri(f"sqlite:///{db_path}")


def build_sql_chain(db: SQLDatabase):
    llm           = build_llm()
    sql_query_chain = create_sql_query_chain(llm, db)

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Question: {question}\nSQL Result: {result}\n\nAnswer:"),
    ])

    def sql_rag_chain(question: str) -> str:
        # Step 1: LLM generates SQL
        raw_sql = sql_query_chain.invoke({"question": question})
        sql     = clean_sql(raw_sql)
        print(f"  [SQL] {sql}")

        # Step 2: execute SQL against tickets.db
        result = db.run(sql)

        # Step 3: LLM turns raw result into natural language
        response = answer_prompt | llm
        return response.invoke({
            "question": question,
            "result":   result,
        }).content

    return sql_rag_chain

import re

SQL_KEYWORDS = [
    "how many", "count", "total", "number of",
    "most common", "least common", "breakdown",
    "how much", "percentage", "average", "avg",
    "which category", "list all", "show all",
    "resolved vs", "escalated", "statistics", "stats",
]

def is_sql_question(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in SQL_KEYWORDS)