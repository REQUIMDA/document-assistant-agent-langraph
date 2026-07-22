# Document Assistant — starter/

**Live Demo: [docdacity.streamlit.app](https://docdacity.streamlit.app)**

This directory contains all runnable code for the Document Assistant Agent.

## Quick Start

```bash
cd starter
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac / Linux
pip install -r requirements.txt
cp .env.example .env         # then fill in your API keys
streamlit run app.py
```

## Entry Points

| Command | What it does |
|---|---|
| `streamlit run app.py` | Launches the full Streamlit web UI |
| `python main.py` | Runs the CLI interface |

## Environment Variables

Copy `.env.example` to `.env` and fill in:

```
GEMINI_API_KEY=""          # primary LLM + embeddings (free tier)
OPENAI_API_KEY=""          # fallback if no Gemini key
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=""
LANGSMITH_PROJECT="DOCUMENT BOT"
```

## Source Files

| File | Responsibility |
|---|---|
| `app.py` | Streamlit UI — chat, file upload, session management |
| `main.py` | CLI loop |
| `src/agent.py` | LangGraph workflow, all agent nodes, SqliteSaver checkpointer |
| `src/assistant.py` | `DocumentAssistant` class — orchestrates sessions, LLM, retriever |
| `src/retrieval.py` | `ChromaRetriever` — ChromaDB + Gemini embeddings, file ingestion |
| `src/tools.py` | LangChain tools: `document_search`, `document_reader`, `document_statistics`, `calculator` |
| `src/prompts.py` | System prompts for Q&A, Summarization, Calculation agents and memory |
| `src/schemas.py` | Pydantic models: `UserIntent`, `UpdateMemoryResponse`, `SessionState` |

## Session Isolation

Each browser tab gets:
- Its own ChromaDB collection named `docs_{session_id}` — uploaded documents are never shared between tabs
- Its own LangGraph `thread_id` in SQLite — conversation memory is fully isolated

Sessions persist across tab closes. To resume a session, paste its ID into the **Resume an existing session** field in the sidebar.

## Generated Files

| Path | Contents |
|---|---|
| `checkpoints.sqlite` | LangGraph conversation state (auto-created) |
| `sessions/` | Session metadata JSON files (auto-created) |
| `logs/` | Per-session tool call history (auto-created) |
