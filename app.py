import os
import json
import math
import re
import sqlite3

import chromadb
import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# Load local .env if running locally (HF Spaces automatically injects secrets as env vars)
load_dotenv()

# =============================================================================
# 1. Database (3 tables, 10 SKUs)
# =============================================================================

conn = sqlite3.connect(":memory:", check_same_thread=False)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

cur.executescript("""
CREATE TABLE inventory (
    sku_id TEXT PRIMARY KEY, sku_name TEXT NOT NULL, category TEXT,
    current_stock REAL NOT NULL, service_level REAL NOT NULL
);
CREATE TABLE lead_time (
    sku_id TEXT PRIMARY KEY, lead_time_days REAL NOT NULL, supplier_name TEXT,
    FOREIGN KEY (sku_id) REFERENCES inventory(sku_id)
);
CREATE TABLE demand (
    sku_id TEXT PRIMARY KEY, avg_daily_demand REAL NOT NULL, demand_std_dev REAL,
    FOREIGN KEY (sku_id) REFERENCES inventory(sku_id)
);
""")

cur.executemany("INSERT INTO inventory VALUES (?,?,?,?,?)", [
    ("SKU-001","Wireless Headphones",    "Electronics",    200, 98),
    ("SKU-002","Paracetamol 500mg",      "Pharma",         800, 99),
    ("SKU-003","Instant Noodles 12-pack","FMCG",          1200, 95),
    ("SKU-004","Steel Rod 10mm",         "Raw Materials",   300, 92),
    ("SKU-005","Laptop 15-inch",         "Electronics",     50, 99),
    ("SKU-006","Vitamin C Tablets",      "Pharma",         600, 97),
    ("SKU-007","Shampoo 400ml",          "FMCG",           900, 95),
    ("SKU-008","Copper Wire 2.5mm",      "Raw Materials",   150, 90),
    ("SKU-009","Bluetooth Speaker",      "Electronics",    120, 96),
    ("SKU-010","Flour 5kg",              "FMCG",          2000, 95),
])
cur.executemany("INSERT INTO lead_time VALUES (?,?,?)", [
    ("SKU-001",21,"TechImports Ltd"), ("SKU-002", 7,"MedSupply Co"),
    ("SKU-003", 4,"FoodDist Pvt"),    ("SKU-004",14,"SteelWorks Inc"),
    ("SKU-005",30,"GlobalTech"),      ("SKU-006",10,"PharmaPlus"),
    ("SKU-007", 5,"FMCG Direct"),     ("SKU-008",21,"CopperMine Co"),
    ("SKU-009",18,"AudioWorld"),      ("SKU-010", 3,"LocalMills"),
])
cur.executemany("INSERT INTO demand VALUES (?,?,?)", [
    ("SKU-001",  8, 2.0), ("SKU-002", 40,  5.0), ("SKU-003", 80, 10.0),
    ("SKU-004", 15, 3.0), ("SKU-005",  3,  1.0), ("SKU-006", 25,  4.0),
    ("SKU-007", 60, 8.0), ("SKU-008", 10,  2.5), ("SKU-009",  6,  1.5),
    ("SKU-010",150,20.0),
])
conn.commit()

def fetch_sku(sku_id: str) -> dict | None:
    row = cur.execute("""
        SELECT i.sku_id, i.sku_name, i.category,
               i.current_stock, i.service_level,
               l.lead_time_days, l.supplier_name,
               d.avg_daily_demand, d.demand_std_dev
        FROM inventory i
        JOIN lead_time l ON l.sku_id = i.sku_id
        JOIN demand d ON d.sku_id = i.sku_id
        WHERE UPPER(i.sku_id) = UPPER(?)
    """, (sku_id,)).fetchone()
    return dict(row) if row else None

def list_skus() -> list[dict]:
    return [dict(r) for r in cur.execute(
        "SELECT sku_id, sku_name, category, current_stock FROM inventory ORDER BY sku_id"
    ).fetchall()]

# =============================================================================
# 2. RAG Knowledge Base
# =============================================================================

KB = [
    ("ss-formula",  "Safety Stock = Z * sqrt(lead_time) * avg_daily_demand. Z=1.28→90%, 1.65→95%, 2.05→98%, 2.33→99%.", {"topic":"formula"}),
    ("rop-formula", "Reorder Point (ROP) = (avg_daily_demand * lead_time) + safety_stock.",                               {"topic":"formula"}),
    ("desired",     "Desired Stock = (avg_daily_demand * lead_time) + safety_stock. Reorder qty = max(0, Desired-Current).",{"topic":"formula"}),
    ("eoq",         "EOQ = sqrt((2 * Annual Demand * Ordering Cost) / Holding Cost).",                                    {"topic":"formula"}),
    ("electronics", "Electronics: lead times 14-30 days, service level 95-99%, high obsolescence risk.",                  {"topic":"category"}),
    ("fmcg",        "FMCG: lead times 3-7 days, high steady demand, service level 95-98%.",                               {"topic":"category"}),
    ("pharma",      "Pharma: lead times 7-21 days, 99%+ service level, strict FIFO and expiry management.",               {"topic":"category"}),
    ("raw-mat",     "Raw materials: lead times 10-45 days, service level 90-95%.",                                        {"topic":"category"}),
    ("abc",         "ABC Analysis: A-items need tight control; C-items tolerate bulk ordering.",                           {"topic":"best_practice"}),
    ("variability", "High demand variability requires more safety stock for the same service level.",                      {"topic":"best_practice"}),
    ("seasonal",    "Increase safety stock 4-6 weeks before peak season; reduce quickly post-season.",                    {"topic":"best_practice"}),
    ("supplier-risk","Single-source or long international lead times: add 10-20% risk buffer to safety stock.",           {"topic":"best_practice"}),
    ("sl-tradeoff", "Raising service level from 95% to 99% roughly doubles safety stock (Z: 1.65→2.33).",                 {"topic":"best_practice"}),
]

embed = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

chroma = chromadb.Client()
col = chroma.create_collection("inventory_kb", metadata={"hnsw:space": "cosine"})
vecs = embed.embed_documents([t for _, t, _ in KB])
col.add(ids=[i for i, _, _ in KB], embeddings=vecs,
        documents=[t for _, t, _ in KB], metadatas=[m for _, _, m in KB])

def retrieve_context(query: str, n: int = 3) -> str:
    qv = embed.embed_query(query)
    res = col.query(query_embeddings=[qv], n_results=n)
    return "\n".join(f"- {c}" for c in res["documents"][0])

# =============================================================================
# 3. Tools
# =============================================================================

@tool
def db_lookup(sku_id: str) -> dict:
    """Fetch all inventory, lead-time, and demand parameters for a SKU from the database."""
    record = fetch_sku(sku_id)
    return record if record else {"error": f"{sku_id} not found. Use list_available_skus to see valid IDs."}

@tool
def list_available_skus() -> list[dict]:
    """Return all SKUs currently in the inventory database."""
    return list_skus()

@tool
def compute_stock_metrics(avg_daily_demand: float, lead_time_days: float,
                           service_level_pct: float, current_stock: float) -> dict:
    """
    Compute safety stock, desired stock level, and reorder quantity with exact arithmetic.
    Returns: z_score, safety_stock, desired_stock, reorder_quantity.
    """
    z = {90: 1.28, 95: 1.65, 97: 2.05, 98: 2.05, 99: 2.33}.get(int(service_level_pct), 1.65)
    ss = round(z * math.sqrt(lead_time_days) * avg_daily_demand, 2)
    desired = round(avg_daily_demand * lead_time_days + ss, 2)
    reorder_qty = max(0.0, round(desired - current_stock, 2))
    return {
        "z_score": z,
        "safety_stock": ss,
        "desired_stock": desired,
        "reorder_quantity": reorder_qty,
        "reorder_needed": reorder_qty > 0,
    }

TOOLS = [db_lookup, list_available_skus, compute_stock_metrics]
TOOL_MAP = {t.name: t for t in TOOLS}

# =============================================================================
# 4. Agent
# =============================================================================

def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not configured. "
            "Please add GROQ_API_KEY in Space Settings -> Secrets."
        )
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0).bind_tools(TOOLS)

SYSTEM = """You are an inventory planning agent with access to tools and domain knowledge.

When a user asks about a SKU:
1. Use db_lookup to fetch its parameters
2. Use compute_stock_metrics for exact arithmetic
3. Provide a clear recommendation: reorder quantity, urgency, and justification

When the user asks to list SKUs, call list_available_skus.

Domain knowledge (use this for context and best practices):
{rag_context}"""

def run_agent(user_message: str) -> str:
    rag_context = retrieve_context(user_message)
    messages = [
        SystemMessage(content=SYSTEM.format(rag_context=rag_context)),
        HumanMessage(content=user_message),
    ]

    llm = get_llm()

    # ReAct loop
    while True:
        response = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return response.content

        for tc in response.tool_calls:
            fn = TOOL_MAP[tc["name"]]
            result = fn.invoke(tc["args"])
            messages.append(ToolMessage(
                content=json.dumps(result, default=str),
                tool_call_id=tc["id"],
            ))

# =============================================================================
# 5. Gradio Chat Interface
# =============================================================================

def chat_handler(message, _history):
    try:
        return run_agent(message)
    except Exception as e:
        return (
            f"⚠️ **Error:** {e}\n\n"
            "Try: *\"What is the reorder quantity for SKU-001?\"*  \n"
            "Or: **list** to see all SKUs."
        )

demo = gr.ChatInterface(
    fn=chat_handler,
    title="Inventory Planning Agent (DB + RAG + Tools)",
    description=(
        "Ask about any SKU — the agent fetches data from the DB, "
        "runs exact calculations via tools, and uses domain knowledge for advice.  \n"
        "Type **list** to see available SKUs."
    ),
    examples=[
        "list",
        "What is the reorder quantity for SKU-001?",
        "Should I reorder SKU-002? It's a pharma product.",
        "Give me a full analysis for SKU-005.",
        "How much should I order for SKU-010?",
        "Which SKUs are most at risk of stockout?",
    ],
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
