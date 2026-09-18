# ☕ RoastOps — Specialty Coffee Roastery & Cafe Logistics Copilot

An intelligent supply chain and inventory decision assistant tailored for artisan coffee roasteries and cafe chains. Powered by SQLite, in-memory domain knowledge retrieval (RAG), and mathematical tool execution via Groq LLMs.

---

## 🌟 Core Features
- **In-Memory SQLite Database:** Real-time tracking of single-origin green beans, fresh roasted espresso, dairy/plant milks, syrups, and packaging.
- **Coffee Supply Chain RAG:** Embedded domain rules covering bean degassing windows, milk cold-chain perishability, origin customs delays, and container MOQs.
- **Exact Tool Calling:** Computes Safety Stock ($Z \times \sqrt{L} \times D$), Reorder Points, and Days of Runway with exact Python arithmetic.
- **Ocean Soft UI:** Built with Gradio 4 featuring soft modern cyan & blue hues and clean typography.
- **Ultra-Lightweight:** Runs in under 45 MB RAM with instant startup on cloud free tiers (Render / Hugging Face).

---

## 📦 Cafe & Roastery SKU Catalog

| SKU ID | Item Name | Category | Lead Time | Daily Demand |
| :--- | :--- | :--- | :--- | :--- |
| **SKU-001** | Ethiopian Yirgacheffe Green Beans (60kg Bag) | Green Beans | 35 days | 0.5 bags/day |
| **SKU-002** | Colombian Supremo Roasted Espresso (1kg) | Roasted Coffee | 3 days | 22 kg/day |
| **SKU-003** | Barista Edition Oat Milk (1L Carton) | Plant Milk | 7 days | 65 cartons/day |
| **SKU-004** | Ceremonial Uji Matcha Powder (500g Tin) | Specialty Tea | 18 days | 2 tins/day |
| **SKU-005** | Compostable 12oz Hot Cups (Case of 1000) | Packaging | 10 days | 3 cases/day |
| **SKU-006** | Madagascar Vanilla Artisan Syrup (750ml) | Syrups & Flavors | 12 days | 4 bottles/day |
| **SKU-007** | Organic Whole Fresh Milk (2L Jug) | Fresh Dairy | 2 days | 85 jugs/day |
| **SKU-008** | Espresso Machine Cleaning Powder (900g) | Cafe Supplies | 5 days | 1 tub/day |
| **SKU-009** | Commercial Cold Brew Filter Bags (Pack of 50) | Brewing Gear | 8 days | 2.5 packs/day |
| **SKU-010** | Raw Turbinado Sugar Sticks (Box of 2000) | Dry Goods | 6 days | 4 boxes/day |

---

## 🛠️ Local Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your Groq API key in `.env`:
   ```bash
   GROQ_API_KEY=your_groq_api_key_here
   ```
4. Run the application:
   ```bash
   python app.py
   ```
