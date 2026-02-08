import sqlite3
import pandas as pd
from src.ollama_client import generate_sql
from src.prompt import PROMPT

from src.config import DB_PATH

def run_text2sql(query):
    sql = generate_sql(PROMPT, query)
    print("\nGenerated SQL:\n", sql)

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(sql, conn)
    conn.close()

    return df
