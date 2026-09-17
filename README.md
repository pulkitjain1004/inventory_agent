---
title: Inventory Planning Agent
emoji: 📦
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# 📦 Inventory Planning Agent (DB + RAG + Tools)

An intelligent inventory management and demand planning assistant powered by SQLite, ChromaDB vector retrieval, and tool-augmented LLM reasoning via Groq.

## Features
- **In-Memory SQLite Database**: Queries real-time stock levels, lead times, and daily demand.
- **RAG Knowledge Base**: Uses ChromaDB + `sentence-transformers/all-MiniLM-L6-v2` embeddings for supply chain domain knowledge.
- **Tool Calling**: Accurate arithmetic calculations for Safety Stock, Reorder Points, and Desired Stock.
- **Gradio Chat UI**: Simple and responsive chat interface.

## Local Setup
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and configure your API key:
   ```bash
   cp .env.example .env
   # Set GROQ_API_KEY=your_key in .env
   ```
4. Run the app:
   ```bash
   python app.py
   ```
