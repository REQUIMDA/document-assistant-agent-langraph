import os
import sys
import uuid

import streamlit as st
from dotenv import load_dotenv

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from assistant import DocumentAssistant

# Load .env from the starter/ directory regardless of where the script is run from
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DocDacity Document Assistant",
    page_icon="📄",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session initialisation — runs once per browser tab
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "assistant" not in st.session_state:
    st.session_state.assistant = None

if "messages" not in st.session_state:
    st.session_state.messages = []          # [{role, content, meta}]

if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []    # filenames uploaded this session

if "last_meta" not in st.session_state:
    st.session_state.last_meta = {}         # metadata from the last assistant turn


# ---------------------------------------------------------------------------
# Lazy assistant init
# ---------------------------------------------------------------------------
def get_assistant() -> DocumentAssistant:
    if st.session_state.assistant is None:
        try:
            gemini_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
            openai_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
        except Exception:
            gemini_key = os.getenv("GEMINI_API_KEY", "")
            openai_key = os.getenv("OPENAI_API_KEY", "")

        if not gemini_key and not openai_key:
            st.error("No API key found. Set GEMINI_API_KEY or OPENAI_API_KEY in your .env / Streamlit secrets.")
            st.stop()

        assistant = DocumentAssistant(
            openai_api_key=openai_key or None,
            gemini_api_key=gemini_key or None,
            temperature=0.1,
        )
        assistant.start_session(
            user_id="web_user",
            session_id=st.session_state.session_id,
        )
        st.session_state.assistant = assistant
    return st.session_state.assistant


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📄 DocDacity")
    st.caption("Intelligent Document Assistant")
    st.divider()

    # --- Session management ---
    st.subheader("Session")
    st.code(st.session_state.session_id[:18] + "…", language=None)

    with st.expander("Resume an existing session"):
        resume_id = st.text_input("Paste session ID", key="resume_input", label_visibility="collapsed")
        if st.button("Resume", use_container_width=True):
            if resume_id.strip():
                st.session_state.session_id = resume_id.strip()
                st.session_state.assistant = None
                st.session_state.ingested_files = []
                st.session_state.last_meta = {}
                # Hydrate UI from saved session history
                resumed = get_assistant()
                st.session_state.messages = resumed.get_ui_messages()
                st.rerun()

    st.divider()

    # --- Document upload ---
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop PDF, TXT, CSV or DOCX files here",
        type=["pdf", "txt", "csv", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        assistant = get_assistant()
        for uf in uploaded_files:
            if uf.name not in st.session_state.ingested_files:
                with st.spinner(f"Indexing {uf.name}…"):
                    try:
                        n_chunks = assistant.ingest_uploaded_document(uf.name, uf.read())
                        st.session_state.ingested_files.append(uf.name)
                        st.success(f"✓ {uf.name} — {n_chunks} chunks indexed")
                    except Exception as e:
                        st.error(f"Failed to index {uf.name}: {e}")

    if st.session_state.ingested_files:
        st.caption("Indexed files:")
        for fname in st.session_state.ingested_files:
            st.markdown(f"- {fname}")
    else:
        st.caption("No files uploaded yet. Sample documents are pre-loaded.")

    st.divider()

    # --- Model info ---
    st.subheader("Model")
    assistant_ref = st.session_state.get("assistant")
    if assistant_ref:
        provider = assistant_ref.llm_provider.upper()
        if provider == "GEMINI":
            st.success("Gemini 3.5 Flash + gemini-embedding-2")
        else:
            st.info("GPT-4o + text-embedding-3-large (OpenAI fallback)")
    else:
        st.caption("Initialising on first message…")

    st.divider()
    st.caption("Built with LangChain · LangGraph · ChromaDB")


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
st.title("Document Assistant")

tab_chat, tab_state = st.tabs(["💬 Chat", "🔍 Session State"])

# ---------------------------------------------------------------------------
# Tab 1 — Chat
# ---------------------------------------------------------------------------
with tab_chat:
    # Render existing conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("meta"):
                meta = msg["meta"]
                with st.expander("Details", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        if meta.get("intent"):
                            intent = meta["intent"]
                            st.markdown(f"**Intent:** `{intent.get('intent_type', '—')}`")
                            st.markdown(f"**Confidence:** {intent.get('confidence', 0):.0%}")
                    with col2:
                        if meta.get("tools_used"):
                            st.markdown(f"**Tools used:** {', '.join(meta['tools_used'])}")
                        if meta.get("sources"):
                            st.markdown(f"**Sources:** {', '.join(meta['sources'])}")
                    if meta.get("actions_taken"):
                        st.markdown(f"**Agent flow:** {' → '.join(meta['actions_taken'])}")
                    if meta.get("summary"):
                        st.markdown(f"**Conversation summary:** {meta['summary']}")

    # Chat input
    if prompt := st.chat_input("Ask about your documents…"):
        assistant = get_assistant()

        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process and show assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                result = assistant.process_message(prompt)

            if result["success"]:
                response_text = result.get("response") or "_No response generated._"
                st.markdown(response_text)

                meta = {
                    "intent": result.get("intent"),
                    "tools_used": result.get("tools_used", []),
                    "sources": result.get("sources", []),
                    "actions_taken": result.get("actions_taken", []),
                    "summary": result.get("summary", ""),
                }
                st.session_state.last_meta = meta

                with st.expander("Details", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        if meta["intent"]:
                            intent = meta["intent"]
                            st.markdown(f"**Intent:** `{intent.get('intent_type', '—')}`")
                            st.markdown(f"**Confidence:** {intent.get('confidence', 0):.0%}")
                    with col2:
                        if meta["tools_used"]:
                            st.markdown(f"**Tools used:** {', '.join(meta['tools_used'])}")
                        if meta["sources"]:
                            st.markdown(f"**Sources:** {', '.join(meta['sources'])}")
                    if meta["actions_taken"]:
                        st.markdown(f"**Agent flow:** {' → '.join(meta['actions_taken'])}")
                    if meta["summary"]:
                        st.markdown(f"**Conversation summary:** {meta['summary']}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "meta": meta,
                })
            else:
                err = result.get("error", "Unknown error")
                st.error(f"Error: {err}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Error: {err}",
                    "meta": {},
                })

# ---------------------------------------------------------------------------
# Tab 2 — Session State Inspector
# ---------------------------------------------------------------------------
with tab_state:
    st.subheader("Current Session State")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Session ID**")
        st.code(st.session_state.session_id, language=None)

        st.markdown("**LLM Provider**")
        a = st.session_state.get("assistant")
        st.code(a.llm_provider.upper() if a else "not initialised", language=None)

        st.markdown("**Messages in history**")
        st.code(str(len(st.session_state.messages)), language=None)

    with col_right:
        meta = st.session_state.last_meta
        if meta:
            st.markdown("**Last intent**")
            if meta.get("intent"):
                st.json(meta["intent"])

            st.markdown("**Active documents (sources)**")
            sources = meta.get("sources") or []
            st.code(", ".join(sources) if sources else "none", language=None)

            st.markdown("**Last agent flow**")
            flow = meta.get("actions_taken") or []
            st.code(" → ".join(flow) if flow else "none", language=None)
        else:
            st.info("Send a message first to see session state here.")

    if st.session_state.last_meta.get("summary"):
        st.divider()
        st.markdown("**Conversation Summary**")
        st.info(st.session_state.last_meta["summary"])

    st.divider()
    st.markdown("**Indexed files this session**")
    if st.session_state.ingested_files:
        for f in st.session_state.ingested_files:
            st.markdown(f"- `{f}`")
    else:
        st.caption("No files uploaded yet.")
