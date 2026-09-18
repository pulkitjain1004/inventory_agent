from __future__ import annotations

import os
import json
import math
import re
import sqlite3
from typing import Optional, List, Dict, Any

import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

# Load local .env if present
load_dotenv()

# =============================================================================
# 1. Database: Specialty Coffee Roastery & Cafe Inventory
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
    ("SKU-001","Ethiopian Yirgacheffe Green Beans (60kg Bag)", "Green Beans",       12, 98),
    ("SKU-002","Colombian Supremo Roasted Espresso (1kg)",     "Roasted Coffee",   150, 99),
    ("SKU-003","Barista Edition Oat Milk (1L Carton)",         "Plant Milk",       420, 99),
    ("SKU-004","Ceremonial Uji Matcha Powder (500g Tin)",      "Specialty Tea",     25, 95),
    ("SKU-005","Compostable 12oz Hot Cups (Case of 1000)",     "Packaging",         40, 95),
    ("SKU-006","Madagascar Vanilla Artisan Syrup (750ml)",     "Syrups & Flavors",  60, 92),
    ("SKU-007","Organic Whole Fresh Milk (2L Jug)",            "Fresh Dairy",      180, 99),
    ("SKU-008","Espresso Machine Cleaning Powder (900g)",      "Cafe Supplies",     15, 90),
    ("SKU-009","Commercial Cold Brew Filter Bags (Pack of 50)","Brewing Gear",     30, 92),
    ("SKU-010","Raw Turbinado Sugar Sticks (Box of 2000)",     "Dry Goods",         50, 90),
])
cur.executemany("INSERT INTO lead_time VALUES (?,?,?)", [
    ("SKU-001", 35, "Direct-Trade Addis Imports"),
    ("SKU-002",  3, "In-House Roastery Batch"),
    ("SKU-003",  7, "OatPure Organics"),
    ("SKU-004", 18, "Kyoto Heritage Teas"),
    ("SKU-005", 10, "EcoCup Packaging Co"),
    ("SKU-006", 12, "Artisan FlavorCraft"),
    ("SKU-007",  2, "Valley Meadow Dairy"),
    ("SKU-008",  5, "BaristaPro Care"),
    ("SKU-009",  8, "BrewTech Equipment"),
    ("SKU-010",  6, "SweetHarvest Direct"),
])
cur.executemany("INSERT INTO demand VALUES (?,?,?)", [
    ("SKU-001",  0.5, 0.1),  # ~0.5 bag (30kg)/day
    ("SKU-002", 22.0, 4.0),  # 22 kg espresso/day
    ("SKU-003", 65.0, 8.0),  # 65 cartons oat milk/day
    ("SKU-004",  2.0, 0.5),  # 2 tins matcha/day
    ("SKU-005",  3.0, 0.6),  # 3 cases (3000 cups)/day
    ("SKU-006",  4.0, 1.0),  # 4 bottles syrup/day
    ("SKU-007", 85.0,12.0),  # 85 jugs whole milk/day
    ("SKU-008",  1.0, 0.2),  # 1 tub cleaner/day
    ("SKU-009",  2.5, 0.4),  # 2.5 packs filter bags/day
    ("SKU-010",  4.0, 0.8),  # 4 boxes sugar/day
])
conn.commit()

def fetch_sku(sku_id: str) -> Optional[Dict[str, Any]]:
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

def list_skus() -> List[Dict[str, Any]]:
    return [dict(r) for r in cur.execute(
        "SELECT sku_id, sku_name, category, current_stock FROM inventory ORDER BY sku_id"
    ).fetchall()]

# =============================================================================
# 2. RAG Knowledge Base: Coffee Roastery & Cafe Supply Chain Rules
# =============================================================================

KB = [
    ("green-coffee",  "Green coffee beans: 30-60 day international lead times, store at 60% humidity, optimal roasting window within 12 months."),
    ("roasted-beans", "Roasted coffee: Peak flavor window is 7 to 28 days post-roast. Roast in micro-batches to avoid staleness; never over-stock."),
    ("fresh-dairy",   "Fresh whole milk: Highly perishable (5-7 day shelf life), 99%+ service level required to prevent morning cafe peak stockout."),
    ("plant-milks",   "Oat & Almond milks: Ambient storage up to 9 months unopened. High steady daily consumption; maintain 7-10 day buffer."),
    ("specialty-tea", "Ceremonial matcha: Sensitive to light and oxygen. Store in airtight cold storage; order in small frequent batches."),
    ("packaging",     "Paper cups & lids: High volume bulk items. Order full pallets to minimize freight costs; keep 2 weeks safety buffer."),
    ("flavor-syrups", "Syrups & flavorings: Shelf-stable (12+ months). Seasonal flavors (pumpkin, hazelnut) require +40% buffer in Q4."),
    ("cold-brew",     "Cold brew filters: Critical summer staple (demand triples May-August). Order ahead of peak warmer months."),
    ("cafe-hygiene",  "Cleaning chemicals & descalers: Mandatory for daily espresso machine backflushing. Zero stockout tolerance for health compliance."),
    ("safety-stock",  "Safety Stock = Z * sqrt(lead_time) * avg_daily_demand. Z=1.28→90%, 1.65→95%, 2.05→98%, 2.33→99% service level."),
    ("reorder-point", "Reorder Point (ROP) = (avg_daily_demand * lead_time) + safety_stock. Trigger purchase orders when stock reaches ROP."),
    ("desired-stock", "Desired Stock = (avg_daily_demand * lead_time) + safety_stock. Reorder qty = max(0, Desired - Current)."),
    ("supplier-risk", "Direct-trade origin coffee (Ethiopia/Colombia): Add 15-20% lead time buffer for port customs and freight delays."),
]

def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())

def retrieve_context(query: str, n: int = 3) -> str:
    query_tokens = set(tokenize(query))
    scored = []
    for doc_id, text in KB:
        doc_tokens = tokenize(text)
        score = sum(1 for t in doc_tokens if t in query_tokens)
        scored.append((score, text))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    top_docs = [t for s, t in scored[:n] if s > 0]
    
    if not top_docs:
        top_docs = [KB[0][1], KB[1][1], KB[9][1]]
        
    return "\n".join(f"- {c}" for c in top_docs)

# =============================================================================
# 3. Tools
# =============================================================================

@tool
def db_lookup(sku_id: str) -> dict:
    """Fetch cafe inventory, lead-time, supplier, and daily demand parameters for a SKU."""
    record = fetch_sku(sku_id)
    return record if record else {"error": f"{sku_id} not found. Type 'list' to see valid coffee & cafe SKUs."}

@tool
def list_available_skus() -> list:
    """Return all specialty coffee, milk, tea, packaging, and cafe supplies in the database."""
    return list_skus()

@tool
def compute_stock_metrics(avg_daily_demand: float, lead_time_days: float,
                           service_level_pct: float, current_stock: float) -> dict:
    """
    Compute exact safety stock, desired batch size, reorder quantity, and cafe runway.
    Returns: z_score, safety_stock, desired_stock, reorder_quantity, runway_days, stockout_risk.
    """
    z = {90: 1.28, 95: 1.65, 97: 2.05, 98: 2.05, 99: 2.33}.get(int(service_level_pct), 1.65)
    ss = round(z * math.sqrt(lead_time_days) * avg_daily_demand, 2)
    desired = round(avg_daily_demand * lead_time_days + ss, 2)
    reorder_qty = max(0.0, round(desired - current_stock, 2))
    runway_days = round(current_stock / max(avg_daily_demand, 0.01), 1)
    
    is_urgent = runway_days <= lead_time_days
    risk_level = "🚨 CRITICAL (Stockout Expected Before Delivery)" if is_urgent else "🟢 HEALTHY BUFFER"
    
    return {
        "z_score": z,
        "safety_stock": ss,
        "desired_stock": desired,
        "reorder_quantity": reorder_qty,
        "reorder_needed": reorder_qty > 0,
        "runway_days": runway_days,
        "stockout_risk": risk_level,
    }

TOOLS = [db_lookup, list_available_skus, compute_stock_metrics]
TOOL_MAP = {t.name: t for t in TOOLS}

# =============================================================================
# 4. Agent Persona
# =============================================================================

def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0).bind_tools(TOOLS)

SYSTEM = """You are RoastOps, the Head of Logistics and Master Roaster Operations Director for an artisan coffee roastery and multi-location specialty cafe chain.

When assisting with inventory decisions:
1. Always call db_lookup to fetch live cafe stock, supplier lead times, and daily consumption.
2. Always call compute_stock_metrics for exact safety stock, reorder quantities, and runway calculations.
3. Integrate domain knowledge (e.g. coffee bean degassing windows, milk shelf-life, direct-trade origin lead times).

Format all recommendations cleanly:
- **Cafe Item Snapshot:** [Item Name] ([Category]) | Supplier: [Name]
- **Operational Stats:** Current Stock: X | Daily Consumption: Y/day | Lead Time: Z days
- **Calculated Buffer:** Safety Stock: X units | Days of Runway: X days | Status: [Risk Status]
- **Roastery / Cafe Action Plan:** [Order Now / Stand By / Roasting Schedule] + [Specific domain justification]

When the user asks to list inventory, call list_available_skus.

Domain Knowledge Context:
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
# 5. Gradio Chat Interface (Ocean Soft Theme)
# =============================================================================

custom_theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.cyan,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
)

def chat_handler(message, _history):
    try:
        return run_agent(message)
    except Exception as e:
        return (
            f"⚠️ **Notice:** {e}\n\n"
            "Try: *\"What is the reorder quantity for SKU-001?\"*  \n"
            "Or: **list** to see all coffee & cafe supplies."
        )

demo = gr.ChatInterface(
    fn=chat_handler,
    title="RoastOps: Specialty Coffee Roastery & Cafe Supply Planner",
    description=(
        "**AI-Powered Coffee & Cafe Logistics Copilot** Real-time bean batch forecasting, "
        "milk shelf-life buffer analysis, and automated procurement recommendations.\n\n"
        "💬 *Ask about green beans, roasted coffee, plant milks, packaging, or type **`list`** to inspect the cafe catalog.*"
    ),
    theme=custom_theme,
    examples=[
        "list",
        "What is the reorder quantity and runway for SKU-001 (Ethiopian Green Beans)?",
        "Should we order more Oat Milk (SKU-003)? It's our top selling plant milk.",
        "Give me a full stock and roast audit for SKU-002 (Colombian Espresso).",
        "Check stock levels for Whole Milk (SKU-007) and Compostable Cups (SKU-005).",
        "Which cafe ingredients or beans are at critical risk of stockout?",
    ],
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"☕ Starting RoastOps Coffee Logistics Engine on port {port}...")
    demo.launch(server_name="0.0.0.0", server_port=port)
