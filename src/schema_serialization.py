"""
Shared schema serialization + SQL safety utilities.

This is the single source of truth for how a database schema is turned into the
text the T5 model sees. Training (`scripts/train_lora.py`), evaluation
(`scripts/evaluate_spider.py`), and serving (`deployment/huggingface_space/app.py`)
all import from here so the train-time and inference-time formats never drift apart.

Why this matters: a "schema-aware" model is only as schema-aware as the schema it
actually receives at inference. If the eval/demo serialize the schema differently
from training, the model is effectively being asked to generalize to a format it
never saw — which silently tanks accuracy.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional


# --------------------------------------------------------------------------- #
# Schema serialization
# --------------------------------------------------------------------------- #
def serialize_schema(
    tables: Dict[str, List[str]],
    foreign_keys: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """
    Flatten a schema into the inline string format the model is trained on.

    Args:
        tables: mapping of ``table_name -> [column, column, ...]``.
        foreign_keys: optional list of ``("table.col", "table.col")`` pairs.

    Returns:
        e.g. ``"book : id , title , rating | author : id , name || book.author_id = author.id"``
    """
    parts = [f"{table} : {' , '.join(cols)}" for table, cols in tables.items()]
    schema_str = " | ".join(parts)

    if foreign_keys:
        fks = " | ".join(f"{a} = {b}" for a, b in foreign_keys)
        return f"{schema_str} || {fks}"
    return schema_str


def build_input(
    question: str,
    tables: Dict[str, List[str]],
    foreign_keys: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """
    Build the full T5 input string for a question + schema.

    The ``translate to SQL:`` prefix cues T5's text-to-text framing. Keep this
    function as the ONLY place that constructs model inputs.
    """
    schema = serialize_schema(tables, foreign_keys)
    return f"translate to SQL: {question} | schema: {schema}"


# System instruction shared by training and inference for the decoder-only (causal) path,
# so the chat prompt never drifts between train and serve. Kept terse on purpose: we want
# just the SQL back, no prose or Markdown fences.
CAUSAL_SYSTEM_PROMPT = (
    "You are an expert data analyst. Given a SQLite database schema and a question, "
    "reply with ONLY a single valid SQLite SELECT query that answers it. "
    "No explanation, no comments, no Markdown code fences."
)


def build_causal_messages(
    question: str,
    tables: Dict[str, List[str]],
    foreign_keys: Optional[List[Tuple[str, str]]] = None,
) -> List[Dict[str, str]]:
    """
    Build chat ``messages`` for a decoder-only instruct model (e.g. Qwen2.5-Coder).

    Returns the system + user turns ready for ``tokenizer.apply_chat_template(...)``. The
    schema is serialized by the same :func:`serialize_schema` used everywhere else, so the
    decoder-only path stays schema-consistent with the T5 path. The assistant turn (the gold
    SQL at train time) is appended by the caller.
    """
    schema = serialize_schema(tables, foreign_keys)
    user = f"Database schema:\n{schema}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": CAUSAL_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# --------------------------------------------------------------------------- #
# Spider `tables.json` → serialized schema
# --------------------------------------------------------------------------- #
def schema_from_spider_tables(table_entry: dict) -> Tuple[Dict[str, List[str]], List[Tuple[str, str]]]:
    """
    Convert one entry of Spider's ``tables.json`` into (tables, foreign_keys).

    Spider's format uses ``column_names_original`` as ``[table_index, column_name]``
    and ``foreign_keys`` as pairs of global column indices.
    """
    table_names = table_entry["table_names_original"]
    column_names = table_entry["column_names_original"]  # [[tbl_idx, col_name], ...]; index 0 is "*"
    column_types = table_entry.get("column_types", [])

    tables: Dict[str, List[str]] = {name: [] for name in table_names}
    for tbl_idx, col_name in column_names:
        if tbl_idx == -1:  # the synthetic "*" column
            continue
        tables[table_names[tbl_idx]].append(col_name)

    foreign_keys: List[Tuple[str, str]] = []
    for a_idx, b_idx in table_entry.get("foreign_keys", []):
        a_tbl, a_col = column_names[a_idx]
        b_tbl, b_col = column_names[b_idx]
        foreign_keys.append(
            (f"{table_names[a_tbl]}.{a_col}", f"{table_names[b_tbl]}.{b_col}")
        )

    return tables, foreign_keys


def schema_from_retriever(schema_dict: dict) -> Tuple[Dict[str, List[str]], List[Tuple[str, str]]]:
    """
    Convert the rich schema produced by ``SchemaRetriever.get_database_schema()``
    into the ``(tables, foreign_keys)`` form used by ``serialize_schema``/``build_input``.

    This is the bridge that makes the agent schema-driven for *any* connected DB:
    the model receives the real tables, columns and relationships introspected from
    the database, not a hand-written demo schema.
    """
    tables: Dict[str, List[str]] = {}
    for tname, tinfo in schema_dict.get("tables", {}).items():
        tables[tname] = [c["name"] for c in tinfo.get("columns", [])]

    foreign_keys: List[Tuple[str, str]] = []
    for rel in schema_dict.get("relationships", []):
        ft, tt = rel.get("from_table"), rel.get("to_table")
        for fc, tc in zip(rel.get("from_columns", []), rel.get("to_columns", [])):
            foreign_keys.append((f"{ft}.{fc}", f"{tt}.{tc}"))

    return tables, foreign_keys


def schema_from_sqlite(db_path: str) -> Tuple[Dict[str, List[str]], List[Tuple[str, str]]]:
    """
    Introspect a live SQLite database into (tables, foreign_keys).

    Used by the demo so the model sees the *actual* connected DB's schema.
    """
    import sqlite3

    con = sqlite3.connect(db_path)
    cur = con.cursor()
    tables: Dict[str, List[str]] = {}
    foreign_keys: List[Tuple[str, str]] = []
    try:
        names = [r[0] for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )]
        for t in names:
            tables[t] = [r[1] for r in cur.execute(f'PRAGMA table_info("{t}")')]
            for fk in cur.execute(f'PRAGMA foreign_key_list("{t}")'):
                # fk row: (id, seq, table, from, to, on_update, on_delete, match)
                ref_table, from_col, to_col = fk[2], fk[3], fk[4]
                foreign_keys.append((f"{t}.{from_col}", f"{ref_table}.{to_col}"))
    finally:
        con.close()
    return tables, foreign_keys


# --------------------------------------------------------------------------- #
# SQL safety guard
# --------------------------------------------------------------------------- #
def is_read_only(sql: str) -> bool:
    """
    Return True only if ``sql`` is a single read-only SELECT (optionally a CTE).

    Uses sqlglot's AST rather than keyword matching, so it correctly rejects
    multi-statement payloads (``SELECT 1; DROP TABLE x``) and write statements
    hidden inside the query. Fails closed: anything that doesn't parse, or parses
    to more than one statement, is rejected.
    """
    import sqlglot
    from sqlglot import exp

    try:
        statements = sqlglot.parse(sql)
    except Exception:
        return False

    # Exactly one non-empty statement allowed.
    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        return False

    root = statements[0]

    # No write/DDL nodes anywhere in the tree.
    forbidden = (
        exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create,
        exp.Alter, exp.TruncateTable, exp.Merge, exp.Command,
    )
    if any(root.find(node) for node in forbidden):
        return False

    # The top-level statement must itself be a SELECT (or a CTE wrapping one).
    if isinstance(root, exp.Select):
        return True
    if isinstance(root, exp.With) and root.find(exp.Select) is not None:
        return True
    if root.find(exp.Select) is not None and not isinstance(root, forbidden):
        # e.g. parenthesized / UNION selects
        return True
    return False


def enforce_limit(sql: str, max_rows: int = 200, dialect: str = "sqlite") -> str:
    """
    Append a ``LIMIT`` to a SELECT that lacks one, so a public demo can't be
    asked to materialize a huge result set. Best-effort; returns the original
    string if it can't be parsed.
    """
    import sqlglot
    from sqlglot import exp

    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception:
        return sql

    select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if select is not None and select.args.get("limit") is None:
        select.limit(max_rows, copy=False)
    return tree.sql(dialect=dialect)
