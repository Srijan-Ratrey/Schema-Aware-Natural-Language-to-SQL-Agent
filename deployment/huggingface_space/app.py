"""
Gradio demo for the Schema-Aware NL → SQL agent, ready for Hugging Face Spaces (free CPU).

The model is pulled from the HF Hub at module load (once, not per-request) and runs on CPU.
Every generated query is passed through a read-only guard before it touches the database, and
results are LIMIT-capped — this demo executes model output against a real DB, so safety is not
optional.

This file is intentionally self-contained so the Space directory is copy-paste deployable.
The serialization + safety helpers mirror `src/schema_serialization.py` in the main repo;
keep them in sync (or `cp ../../src/schema_serialization.py .` and import from it instead).

Configure the model via the MODEL_ID env var (HF Space → Settings → Variables), default below.
"""

import os
import sqlite3
from pathlib import Path

import gradio as gr
import sqlglot
from sqlglot import exp
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
MODEL_ID = os.environ.get("MODEL_ID", "your-username/nl2sql-t5-large-spider")
SAMPLE_DB = Path(__file__).parent / "data" / "sample.db"
MAX_ROWS = 200

# --------------------------------------------------------------------------- #
# Schema serialization + safety (mirror of src/schema_serialization.py)
# --------------------------------------------------------------------------- #
def serialize_schema(tables, foreign_keys=None):
    parts = [f"{t} : {' , '.join(c)}" for t, c in tables.items()]
    schema_str = " | ".join(parts)
    if foreign_keys:
        fks = " | ".join(f"{a} = {b}" for a, b in foreign_keys)
        return f"{schema_str} || {fks}"
    return schema_str


def build_input(question, tables, foreign_keys=None):
    return f"translate to SQL: {question} | schema: {serialize_schema(tables, foreign_keys)}"


def schema_from_sqlite(db_path):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    tables, fks = {}, []
    try:
        names = [r[0] for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        for t in names:
            tables[t] = [r[1] for r in cur.execute(f'PRAGMA table_info("{t}")')]
            for fk in cur.execute(f'PRAGMA foreign_key_list("{t}")'):
                fks.append((f"{t}.{fk[3]}", f"{fk[2]}.{fk[4]}"))
    finally:
        con.close()
    return tables, fks


def is_read_only(sql):
    try:
        statements = [s for s in sqlglot.parse(sql) if s is not None]
    except Exception:
        return False
    if len(statements) != 1:
        return False
    root = statements[0]
    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create,
                 exp.Alter, exp.TruncateTable, exp.Merge, exp.Command)
    if any(root.find(n) for n in forbidden):
        return False
    if isinstance(root, exp.Select):
        return True
    if isinstance(root, exp.With) and root.find(exp.Select) is not None:
        return True
    return root.find(exp.Select) is not None and not isinstance(root, forbidden)


def enforce_limit(sql, max_rows=MAX_ROWS, dialect="sqlite"):
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception:
        return sql
    select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if select is not None and select.args.get("limit") is None:
        select.limit(max_rows, copy=False)
    return tree.sql(dialect=dialect)


# --------------------------------------------------------------------------- #
# Sample database (built once if missing)
# --------------------------------------------------------------------------- #
def ensure_sample_db():
    if SAMPLE_DB.exists():
        return
    SAMPLE_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(SAMPLE_DB)
    con.executescript(
        """
        CREATE TABLE Customer (Id INTEGER PRIMARY KEY, FirstName TEXT, LastName TEXT,
                               City TEXT, Country TEXT);
        CREATE TABLE Supplier (Id INTEGER PRIMARY KEY, CompanyName TEXT, City TEXT, Country TEXT);
        CREATE TABLE Product (Id INTEGER PRIMARY KEY, ProductName TEXT, SupplierId INTEGER,
                              UnitPrice REAL, IsDiscontinued INTEGER,
                              FOREIGN KEY (SupplierId) REFERENCES Supplier(Id));
        CREATE TABLE "Order" (Id INTEGER PRIMARY KEY, OrderDate TEXT, CustomerId INTEGER,
                              TotalAmount REAL, FOREIGN KEY (CustomerId) REFERENCES Customer(Id));
        CREATE TABLE OrderItem (Id INTEGER PRIMARY KEY, OrderId INTEGER, ProductId INTEGER,
                                UnitPrice REAL, Quantity INTEGER,
                                FOREIGN KEY (OrderId) REFERENCES "Order"(Id),
                                FOREIGN KEY (ProductId) REFERENCES Product(Id));

        INSERT INTO Customer VALUES (1,'Maria','Anders','Berlin','Germany'),
            (2,'Ana','Trujillo','Mexico City','Mexico'),(3,'Thomas','Hardy','London','UK'),
            (4,'Christina','Berglund','Lulea','Sweden'),(5,'Hanna','Moos','Mannheim','Germany');
        INSERT INTO Supplier VALUES (1,'Exotic Liquids','London','UK'),
            (2,'New Orleans Cajun','New Orleans','USA'),(3,'Tokyo Traders','Tokyo','Japan');
        INSERT INTO Product VALUES (1,'Chai',1,18.0,0),(2,'Chang',1,19.0,0),
            (3,'Aniseed Syrup',2,10.0,0),(4,'Cajun Seasoning',2,22.0,0),
            (5,'Ikura',3,31.0,0),(6,'Tofu',3,23.25,1);
        INSERT INTO "Order" VALUES (1,'2023-01-15',1,178.0),(2,'2023-02-20',2,62.0),
            (3,'2023-03-05',1,93.0),(4,'2023-04-12',3,124.0);
        INSERT INTO OrderItem VALUES (1,1,1,18.0,5),(2,1,5,31.0,2),(3,2,3,10.0,4),
            (4,3,2,19.0,3),(5,4,4,22.0,2),(6,4,5,31.0,1);
        """
    )
    con.commit()
    con.close()


# --------------------------------------------------------------------------- #
# Model (loaded once at module scope — critical for cold-start on free CPU)
# --------------------------------------------------------------------------- #
print(f"Loading model: {MODEL_ID} (this is slow on first boot)...")
ensure_sample_db()
SCHEMA_TABLES, SCHEMA_FKS = schema_from_sqlite(str(SAMPLE_DB))
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
model.eval()
print("Model loaded.")


@torch.no_grad()
def nl_to_sql(question):
    x = build_input(question, SCHEMA_TABLES, SCHEMA_FKS)
    ids = tok(x, return_tensors="pt", truncation=True, max_length=512).input_ids
    out = model.generate(ids, max_new_tokens=256, num_beams=5)
    return tok.decode(out[0], skip_special_tokens=True)


def run_query(sql):
    con = sqlite3.connect(str(SAMPLE_DB))
    try:
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall()
        return cols, rows
    finally:
        con.close()


def ask(question):
    if not question or not question.strip():
        return "", None, "Enter a question to get started."
    sql = nl_to_sql(question)
    if not is_read_only(sql):
        return sql, None, "⛔ Blocked: only read-only SELECT queries are allowed in this demo."
    safe_sql = enforce_limit(sql)
    try:
        cols, rows = run_query(safe_sql)
    except Exception as e:
        return safe_sql, None, f"Query failed: {e}"
    status = f"✅ {len(rows)} row(s)" + (f" (capped at {MAX_ROWS})" if len(rows) >= MAX_ROWS else "")
    return safe_sql, gr.Dataframe(headers=cols, value=rows), status


SCHEMA_PREVIEW = "\n".join(f"• {t} ({', '.join(c)})" for t, c in SCHEMA_TABLES.items())

with gr.Blocks(title="Schema-Aware NL → SQL") as demo:
    gr.Markdown("# Schema-Aware NL → SQL\nAsk a question in plain English over a sample "
                "e-commerce database. The fine-tuned T5 model generates SQL, which is "
                "safety-checked (read-only) and executed live.")
    gr.Markdown(f"**Sample schema**\n```\n{SCHEMA_PREVIEW}\n```")
    with gr.Row():
        q = gr.Textbox(label="Ask in plain English",
                       placeholder="How many customers are from Germany?", scale=4)
        btn = gr.Button("Generate & Run", variant="primary", scale=1)
    sql_out = gr.Code(label="Generated SQL", language="sql")
    status_out = gr.Markdown()
    table_out = gr.Dataframe(label="Results")
    gr.Examples(
        examples=[
            "How many customers are there?",
            "List all products with a unit price above 20",
            "Show the total amount of each order",
            "Which customers are from Germany?",
            "What are the names of products supplied by Tokyo Traders?",
        ],
        inputs=q,
    )
    btn.click(ask, inputs=q, outputs=[sql_out, table_out, status_out])
    q.submit(ask, inputs=q, outputs=[sql_out, table_out, status_out])

if __name__ == "__main__":
    demo.launch()
