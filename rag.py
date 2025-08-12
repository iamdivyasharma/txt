# rag_sql_langchain_bedrock.py
# Minimal Vanna-style RAG using LangChain + AWS Bedrock + Chroma + MSSQL

# pip install:
#   langchain langchain-community langchain-aws chromadb pandas pyodbc boto3

import os
import json
import pandas as pd
from typing import List, Optional

# ---- LangChain / Bedrock / Chroma ----
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma

# NOTE: Depending on your LangChain version, Bedrock integrations can live in either:
# - langchain_aws (new)  OR
# - langchain_community (older)
try:
    from langchain_aws.chat_models import ChatBedrock
    from langchain_aws.embeddings import BedrockEmbeddings
except Exception:
    from langchain_community.chat_models import BedrockChat as ChatBedrock
    from langchain_community.embeddings import BedrockEmbeddings

from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

# ------------- CONFIG -------------
BEDROCK_REGION = os.getenv("AWS_REGION", "us-east-1")
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
LLM_MODEL_ID   = "anthropic.claude-3-haiku-20240307-v1:0"
CHROMA_PATH    = "./chroma_langchain"
SAMPLE_ROWS_PER_TABLE = 40
ROW_CHUNK_SIZE = 5
EXCLUDE_SCHEMAS = {"sys", "INFORMATION_SCHEMA"}  # tune as needed
# -----------------------------------
from datetime import date, datetime, time
from decimal import Decimal
import numpy as np
import pandas as pd

def _to_jsonable(v):
    # Pandas NA
    if pd.isna(v):
        return None
    # Numpy scalars -> python
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        return v.item()
    # Datetime-like -> ISO8601
    if isinstance(v, (datetime, pd.Timestamp)):
        return v.isoformat()
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, time):
        return v.isoformat()
    # Decimals
    if isinstance(v, Decimal):
        # choose float or str; float is convenient, str preserves precision
        try:
            return float(v)
        except Exception:
            return str(v)
    # Bytes -> hex (or str(v) if you prefer)
    if isinstance(v, (bytes, bytearray)):
        return v.hex()
    # Fallback: keep primitives or stringify
    if isinstance(v, (int, float, bool, str)) or v is None:
        return v
    # Last resort
    return str(v)

def qual(schema: str, table: str) -> str:
    return f"[{schema}].[{table}]"

def get_info_schema(conn) -> pd.DataFrame:
    # Safe aliases (avoid reserved keyword 'schema')
    q = """
    SELECT
        TABLE_SCHEMA  AS table_schema,
        TABLE_NAME    AS table_name,
        COLUMN_NAME   AS column_name,
        DATA_TYPE     AS data_type,
        IS_NULLABLE   AS is_nullable,
        ORDINAL_POSITION
    FROM INFORMATION_SCHEMA.COLUMNS
    ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION;
    """
    return pd.read_sql(q, conn)

def format_table_ddl(schema: str, table: str, df_tbl: pd.DataFrame) -> str:
    lines = [f"TABLE {qual(schema, table)} ("]
    for _, r in df_tbl.iterrows():
        nulls = "NULL" if str(r["is_nullable"]).upper() == "YES" else "NOT NULL"
        lines.append(f"  [{r['column_name']}] {r['data_type']} {nulls},")
    lines[-1] = lines[-1].rstrip(",")
    lines.append(")")
    return "\n".join(lines)

def index_into_chroma_langchain(
    conn,
    chroma_path: str = CHROMA_PATH,
    bedrock_region: str = BEDROCK_REGION,
    embed_model_id: str = EMBED_MODEL_ID,
    sample_rows_per_table: int = SAMPLE_ROWS_PER_TABLE,
    row_chunk_size: int = ROW_CHUNK_SIZE,
    only_schemas: Optional[List[str]] = None,
):
    """Build two sets of Documents (schema + row samples) and index into Chroma."""
    df = get_info_schema(conn)
    if only_schemas:
        df = df[df["table_schema"].isin(only_schemas)]
    df = df[~df["table_schema"].isin(EXCLUDE_SCHEMAS)]

    tables = (
        df.groupby(["table_schema","table_name"])
          .size()
          .reset_index()[["table_schema","table_name"]]
    )

    # 1) Build LangChain Documents
    docs: List[Document] = []

    # Schema docs (one per table)
    for _, row in tables.iterrows():
        sch, tbl = row["table_schema"], row["table_name"]
        df_tbl = df[(df["table_schema"] == sch) & (df["table_name"] == tbl)]
        ddl_text = format_table_ddl(sch, tbl, df_tbl)
        docs.append(
            Document(
                page_content=ddl_text,
                metadata={"type": "schema", "schema": sch, "table": tbl}
            )
        )

    # Row sample docs (small batches in JSON-lines)
    for _, row in tables.iterrows():
        sch, tbl = row["table_schema"], row["table_name"]
        qt = qual(sch, tbl)
        try:
            df_rows = pd.read_sql(f"SELECT TOP {sample_rows_per_table} * FROM {qt} ORDER BY NEWID()", conn)
        except Exception:
            continue

        for i in range(0, len(df_rows), row_chunk_size):
            part = df_rows.iloc[i:i+row_chunk_size]
            lines = [f"TABLE {qt}\nROWS:"]
            for _, r in part.iterrows():
                as_dict = {
                    str(c): (None if pd.isna(r[c]) else (r[c].item() if hasattr(r[c], "item") else r[c]))
                    for c in part.columns
                }
                lines.append(json.dumps(as_dict, ensure_ascii=False))
            docs.append(
                Document(
                    page_content="\n".join(lines),
                    metadata={"type": "row_chunk", "schema": sch, "table": tbl, "chunk_index": i // row_chunk_size}
                )
            )

    # 2) Create embeddings + Chroma store via LangChain
    embeddings = BedrockEmbeddings(
        model_id=embed_model_id,
        region_name=bedrock_region,
    )

    # Wipe & rebuild collections in a single persisted Chroma folder
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=chroma_path
    )
    vectordb.persist()
    return vectordb

def make_retriever(
    chroma_path: str = CHROMA_PATH,
    bedrock_region: str = BEDROCK_REGION,
    embed_model_id: str = EMBED_MODEL_ID,
    k: int = 8,
):
    embeddings = BedrockEmbeddings(
        model_id=embed_model_id,
        region_name=bedrock_region,
    )
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=chroma_path
    )
    return vectordb.as_retriever(search_kwargs={"k": k})

def build_chain(
    bedrock_region: str = BEDROCK_REGION,
    llm_model_id: str = LLM_MODEL_ID,
):
    # Claude 3 Haiku via Bedrock Chat (LangChain wrapper)
    llm = ChatBedrock(
        model_id=llm_model_id,
        region_name=bedrock_region,
        # You can set model_kwargs={"temperature": 0} on older wrappers
    )

    system_msg = (
        "You are a senior SQL developer for Microsoft SQL Server.\n"
        "Use only the provided schema and sample rows.\n"
        "Always qualify tables with their schema (e.g., [dbo].[table]).\n"
        "Do not invent columns or tables.\n"
        "Return ONLY one SQL query in a fenced code block."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            ("human",
             "# Retrieved Schema\n{schema}\n\n# Retrieved Row Samples\n{rows}\n\n"
             "Question: {question}\nWrite the SQL.")
        ]
    )

    return prompt | llm | StrOutputParser()

def ask_sql_langchain(
    question: str,
    retriever,
    chain,
):
    # Pull top docs
    docs = retriever.get_relevant_documents(question)
    schema_texts = [d.page_content for d in docs if d.metadata.get("type") == "schema"]
    row_texts    = [d.page_content for d in docs if d.metadata.get("type") == "row_chunk"]

    # Prepare prompt inputs
    inputs = {
        "schema": "\n\n".join(schema_texts[:6]),  # take a few
        "rows":   "\n\n".join(row_texts[:8]),     # take a few
        "question": question,
    }
    return chain.invoke(inputs)

# ----------- Optional: execute the SQL -----------
def run_sql(conn, sql_text: str) -> pd.DataFrame:
    import re
    m = re.search(r"```(?:sql)?\s*(.*?)```", sql_text, flags=re.DOTALL | re.IGNORECASE)
    sql = m.group(1) if m else sql_text
    return pd.read_sql(sql, conn)


# ----------------- USAGE EXAMPLE -----------------
if __name__ == "__main__":
    import pyodbc

    # 0) MSSQL connection (adjust!)
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost;DATABASE=MyDb;UID=sa;PWD=YourStrong!Passw0rd"
    )

    # 1) Index (run when schema/data changes)
    index_into_chroma_langchain(conn, only_schemas=["dbo"])  # keep focused

    # 2) Build retriever & chain
    retriever = make_retriever()
    chain = build_chain()

    # 3) Ask for SQL
    question = "monthly revenue by product in 2024?"
    sql_text = ask_sql_langchain(question, retriever, chain)
    print("=== Generated SQL ===\n", sql_text)

    # 4) (Optional) Run it
    try:
        df = run_sql(conn, sql_text)
        print(df.head())
    except Exception as e:
        print("Execution error:", e)
