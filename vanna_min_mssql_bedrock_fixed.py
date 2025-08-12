"""
vanna_min_mssql_bedrock_fixed.py
--------------------------------
Minimal Vanna-style RAG for Microsoft SQL Server using ONLY AWS Bedrock:
- Embeddings: Amazon Titan Text Embeddings V2
- LLM: Anthropic Claude 3 Haiku (via Bedrock)
Fix: avoids reserved word 'schema' by aliasing to table_schema.
"""

import json
from typing import List, Dict, Optional
import pandas as pd

# ---------------- Embeddings: Titan V2 wrapper ----------------

class BedrockTitanEmbeddingFunction:
    def __init__(self, region_name: str = "us-east-1", model_id: str = "amazon.titan-embed-text-v2:0",
                 dimensions: Optional[int] = None, normalize: Optional[bool] = None):
        import boto3
        self.client = boto3.client("bedrock-runtime", region_name=region_name)
        self.model_id = model_id
        self.dimensions = dimensions
        self.normalize = normalize

    def __call__(self, texts: List[str]) -> List[List[float]]:
        vecs = []
        for t in texts:
            payload = {"inputText": t}
            if self.dimensions is not None:
                payload["dimensions"] = int(self.dimensions)
            if self.normalize is not None:
                payload["normalize"] = bool(self.normalize)
            resp = self.client.invoke_model(modelId=self.model_id, body=json.dumps(payload))
            body = json.loads(resp["body"].read())
            vecs.append(body["embedding"])
        return vecs

# ---------------- Helpers ----------------

def _qual(schema: str, table: str) -> str:
    return f"[{schema}].[{table}]"

def _get_info_schema(conn) -> pd.DataFrame:
    q = """
    SELECT
        TABLE_SCHEMA  AS table_schema,
        TABLE_NAME    AS table_name,
        COLUMN_NAME   AS column_name,
        DATA_TYPE     AS data_type,
        IS_NULLABLE   AS is_nullable,
        ORDINAL_POSITION
    FROM INFORMATION_SCHEMA.COLUMNS
    ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION
    """
    return pd.read_sql(q, conn)

def _format_table_ddl(schema: str, table: str, df_tbl: pd.DataFrame) -> str:
    lines = [f"TABLE {_qual(schema, table)} ("]
    for _, r in df_tbl.iterrows():
        nulls = "NULL" if str(r["is_nullable"]).upper() == "YES" else "NOT NULL"
        lines.append(f"  [{r['column_name']}] {r['data_type']} {nulls},")
    lines[-1] = lines[-1].rstrip(",")
    lines.append(")")
    return "\n".join(lines)

# ---------------- Index into Chroma ----------------

def index_db(conn,
             chroma_path: str = "./chroma",
             sample_rows_per_table: int = 40,
             chunk_size: int = 5,
             exclude_schemas: Optional[List[str]] = None,
             bedrock_region: str = "us-east-1",
             titan_dimensions: Optional[int] = None,
             titan_normalize: Optional[bool] = None):
    import chromadb
    exclude_schemas = exclude_schemas or ["sys", "INFORMATION_SCHEMA"]
    df = _get_info_schema(conn)

    tables = (
        df[~df["table_schema"].isin(exclude_schemas)]
        .groupby(["table_schema","table_name"]).size().reset_index()[["table_schema","table_name"]]
    )

    client = chromadb.PersistentClient(path=chroma_path)
    embed = BedrockTitanEmbeddingFunction(region_name=bedrock_region,
                                          dimensions=titan_dimensions,
                                          normalize=titan_normalize)
    schema_col = client.get_or_create_collection("schema_chunks", embedding_function=embed)
    rows_col   = client.get_or_create_collection("row_samples",   embedding_function=embed)

    schema_docs, schema_ids, schema_meta = [], [], []
    for _, row in tables.iterrows():
        sch, tbl = row["table_schema"], row["table_name"]
        df_tbl = df[(df["table_schema"]==sch) & (df["table_name"]==tbl)]
        schema_docs.append(_format_table_ddl(sch, tbl, df_tbl))
        schema_ids.append(f"schema::{sch}.{tbl}")
        schema_meta.append({"type":"schema","schema":sch,"table":tbl})
    if schema_docs:
        try: schema_col.delete(where={"type":"schema"})
        except Exception: pass
        schema_col.add(documents=schema_docs, ids=schema_ids, metadatas=schema_meta)

    row_docs, row_ids, row_meta = [], [], []
    for _, row in tables.iterrows():
        sch, tbl = row["table_schema"], row["table_name"]
        qt = _qual(sch, tbl)
        try:
            df_rows = pd.read_sql(f"SELECT TOP {sample_rows_per_table} * FROM {qt} ORDER BY NEWID()", conn)
        except Exception:
            continue
        for i in range(0, len(df_rows), chunk_size):
            part = df_rows.iloc[i:i+chunk_size]
            lines = [f"TABLE {qt}\nROWS:"]
            for _, r in part.iterrows():
                as_dict = {str(c): (None if pd.isna(r[c]) else (r[c].item() if hasattr(r[c],'item') else r[c])) for c in part.columns}
                lines.append(json.dumps(as_dict, ensure_ascii=False))
            row_docs.append("\n".join(lines))
            row_ids.append(f"rows::{sch}.{tbl}::{i//chunk_size}")
            row_meta.append({"type":"row_chunk","schema":sch,"table":tbl,"chunk_index":i//chunk_size})
    if row_docs:
        try: rows_col.delete(where={"type":"row_chunk"})
        except Exception: pass
        rows_col.add(documents=row_docs, ids=row_ids, metadatas=row_meta)

# ---------------- Retrieve ----------------

def retrieve(question: str,
             chroma_path: str = "./chroma",
             k_schema: int = 6,
             k_rows: int = 8,
             bedrock_region: str = "us-east-1",
             titan_dimensions: Optional[int] = None,
             titan_normalize: Optional[bool] = None) -> Dict[str, List[str]]:
    import chromadb
    client = chromadb.PersistentClient(path=chroma_path)
    embed = BedrockTitanEmbeddingFunction(region_name=bedrock_region,
                                          dimensions=titan_dimensions,
                                          normalize=titan_normalize)
    s_col = client.get_or_create_collection("schema_chunks", embedding_function=embed)
    r_col = client.get_or_create_collection("row_samples", embedding_function=embed)

    s = s_col.query(query_texts=[question], n_results=k_schema, where={"type":"schema"})
    r = r_col.query(query_texts=[question], n_results=k_rows, where={"type":"row_chunk"})

    flat = lambda res: [d for group in (res.get("documents") or []) for d in group]
    return {"schema": flat(s), "rows": flat(r)}

# ---------------- Claude Haiku ----------------

def to_sql_with_bedrock_claude(question: str,
                               ctx: Dict[str, List[str]],
                               bedrock_region: str = "us-east-1",
                               model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
                               max_tokens: int = 800,
                               temperature: float = 0.0) -> str:
    import boto3
    system = (
        "You are a senior SQL developer for Microsoft SQL Server.\n"
        "Use only the provided schema. Do not invent columns.\n"
        "Return ONLY one SQL query in a code block."
    )
    content = "# Schema\n" + "\n\n".join(ctx["schema"]) + "\n\n# Row Samples\n" + "\n\n".join(ctx["rows"]) + \
              f"\n\nQuestion: {question}\nWrite the SQL."
    client = boto3.client("bedrock-runtime", region_name=bedrock_region)
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "system": system,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": content}]}
        ]
    }
    resp = client.invoke_model(modelId=model_id, body=json.dumps(payload))
    body = json.loads(resp["body"].read())
    texts = []
    for block in body.get("content", []):
        if block.get("type") == "text":
            texts.append(block.get("text", ""))
    return "\n".join(texts)

# ---------------- Execute SQL ----------------

def run_sql(conn, sql_text: str) -> pd.DataFrame:
    import re
    m = re.search(r"```(?:sql)?\s*(.*?)```", sql_text, flags=re.DOTALL|re.IGNORECASE)
    sql = m.group(1) if m else sql_text
    return pd.read_sql(sql, conn)
