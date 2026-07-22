# Document Assistant Agent

**Live Demo: [docdacity.streamlit.app](https://docdacity.streamlit.app)**

A production-grade multi-agent document assistant built with LangGraph, LangChain, ChromaDB, and Gemini 3.5 Flash. Upload financial or healthcare documents and ask questions, get summaries, or run calculations — all through a live Streamlit web app with full multi-user session isolation.

## Features

- **3 specialised ReAct agents** — Q&A, Summarization, Calculation — each with its own system prompt and tool access
- **Intent-driven routing** with keyword fast-path that skips the LLM for obvious intents
- **ChromaDB semantic vector search** using Gemini embeddings — per-session collection for full user isolation
- **SQLite persistent memory** via LangGraph `SqliteSaver` — conversation survives tab closes and resumption
- **File ingestion** — PDF, DOCX, CSV, and TXT via drag-and-drop upload
- **Multi-user safe** — each browser tab gets its own ChromaDB collection and LangGraph thread ID
- **Secure calculator** — AST-based evaluator, no `eval()` on user input
- **LangSmith tracing** — zero-code, configured via environment variables

## Stack

| Layer | Technology |
|---|---|
| LLM | Gemini 3.5 Flash (free tier) / GPT-4o fallback |
| Embeddings | gemini-embedding-2 / text-embedding-3-large fallback |
| Agent framework | LangChain + LangGraph |
| Vector store | ChromaDB (in-memory, per session) |
| Memory | SqliteSaver (persistent across sessions) |
| UI | Streamlit |
| Tracing | LangSmith |

## Getting Started

### Prerequisites

- Python 3.9 or higher
- A Gemini API key (free at [ai.google.dev](https://ai.google.dev)) — OpenAI key optional as fallback

### Installation

```bash
cd starter
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac / Linux
pip install -r requirements.txt
```

### Environment Setup

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```
GEMINI_API_KEY="your_gemini_key_here"
OPENAI_API_KEY=""
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY="your_langsmith_key_here"
LANGSMITH_PROJECT="DOCUMENT BOT"
```

### Run Locally

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Deployment (Streamlit Cloud)

1. Push this repo to GitHub (must be public for free tier)
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**
3. Set **Main file path** to `starter/app.py`
4. Under **Advanced settings → Secrets**, add:

```toml
GEMINI_API_KEY = "your_key"
OPENAI_API_KEY = ""
LANGSMITH_TRACING = "true"
LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
LANGSMITH_API_KEY = "your_key"
LANGSMITH_PROJECT = "DOCUMENT BOT"
```

5. Click **Deploy**

## Project Structure

```
starter/
├── app.py                    # Streamlit web UI entry point
├── main.py                   # CLI entry point
├── requirements.txt
├── .env.example
└── src/
    ├── agent.py              # LangGraph workflow, nodes, SqliteSaver
    ├── assistant.py          # DocumentAssistant orchestrator class
    ├── retrieval.py          # ChromaRetriever — ChromaDB + Gemini embeddings
    ├── tools.py              # LangChain tools: search, reader, stats, calculator
    ├── prompts.py            # System prompts for each agent
    └── schemas.py            # Pydantic models
```

## Agent Architecture

```
User Input
    │
    ▼
classify_intent  ──(keyword fast-path skips LLM for obvious intents)
    │
    ├──► qa_agent           (document_search, document_reader)
    ├──► summarization_agent (document_search, document_reader, document_statistics)
    └──► calculation_agent  (document_search, document_reader, calculator)
              │
              ▼
        update_memory  ──(only runs when tools were used)
              │
              ▼
             END
```

## Performance Optimisations

- **Keyword fast-path**: common intents like "summarize" or "total" bypass the classification LLM call entirely
- **ReAct agent caching**: `create_react_agent` compiles the internal graph once per LLM instance, reused on every turn
- **Conditional memory updates**: `update_memory` only runs when tools were actually called, skipping redundant LLM calls on conversational turns
- **Chat history truncation**: last 6 messages passed to agents, preventing unbounded token growth
- **Sample document embedding cache**: 5 built-in docs embedded once per process, not per session

## Built With

- [LangChain](https://www.langchain.com) — tool definitions and LLM orchestration
- [LangGraph](https://langchain-ai.github.io/langgraph) — stateful multi-agent workflow and checkpointing
- [ChromaDB](https://www.trychroma.com) — vector store for semantic search
- [Google Gemini](https://ai.google.dev) — LLM and embeddings (free tier)
- [Streamlit](https://streamlit.io) — web UI and deployment
- [LangSmith](https://smith.langchain.com) — tracing and observability
