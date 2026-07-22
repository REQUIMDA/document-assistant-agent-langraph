import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from langchain_core.messages import BaseMessage

from schemas import SessionState
from retrieval import ChromaRetriever
from tools import get_all_tools, ToolLogger
from agent import create_workflow, AgentState


def _build_llm(gemini_api_key: Optional[str], openai_api_key: Optional[str], temperature: float):
    """Return Gemini LLM if key present, else fall back to OpenAI."""
    if gemini_api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-3.5-flash",
            google_api_key=gemini_api_key,
            temperature=temperature,
        ), "gemini"
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        api_key=openai_api_key,
        model="gpt-4o",
        temperature=temperature,
        base_url="https://openai.vocareum.com/v1",
    ), "openai"


def _build_embedding_model(gemini_api_key: Optional[str], openai_api_key: Optional[str]):
    """Return Gemini embeddings if key present, else fall back to OpenAI."""
    if gemini_api_key:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2",
            google_api_key=gemini_api_key,
        )
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=openai_api_key,
    )


class DocumentAssistant:
    """
    Orchestrates sessions, the LangGraph workflow, ChromaDB retrieval,
    and Gemini (or OpenAI fallback) LLM + embeddings.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        temperature: float = 0.1,
        session_storage_path: str = "./sessions",
    ):
        self.openai_api_key = openai_api_key
        self.gemini_api_key = gemini_api_key
        self.temperature = temperature

        self.llm, self.llm_provider = _build_llm(gemini_api_key, openai_api_key, temperature)
        self.embedding_model = _build_embedding_model(gemini_api_key, openai_api_key)

        self.logs_dir = "./logs"
        self.session_storage_path = session_storage_path
        os.makedirs(session_storage_path, exist_ok=True)

        # Set per start_session()
        self.tool_logger: Optional[ToolLogger] = None
        self.tools: Optional[List] = None
        self.retriever: Optional[ChromaRetriever] = None
        self.workflow = None
        self.current_session: Optional[SessionState] = None

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def start_session(self, user_id: str, session_id: Optional[str] = None) -> str:
        if session_id and self._session_exists(session_id):
            self.current_session = self._load_session(session_id)
            print(f"Resumed session {session_id}")
        else:
            session_id = session_id or str(uuid.uuid4())
            self.current_session = SessionState(
                session_id=session_id,
                user_id=user_id,
                conversation_history=[],
                document_context=[],
            )
            print(f"Started new session {session_id}")

        # Per-session ChromaDB collection — isolates each user's documents
        self.retriever = ChromaRetriever(
            embedding_model=self.embedding_model,
            collection_name=f"docs_{session_id}",
        )

        self.tool_logger = ToolLogger(logs_dir=self.logs_dir, session_id=session_id)
        self.tools = get_all_tools(self.retriever, self.tool_logger)
        self.workflow = create_workflow(self.llm, self.tools)

        return session_id

    def get_ui_messages(self) -> list:
        """Return conversation history as {role, content, meta} dicts for st.session_state.messages."""
        if not self.current_session:
            return []
        messages = []
        for turn in self.current_session.conversation_history:
            user_input = turn.get("user_input", "")
            assistant_response = turn.get("assistant_response", "")
            if user_input:
                messages.append({"role": "user", "content": user_input, "meta": {}})
            if assistant_response:
                messages.append({"role": "assistant", "content": assistant_response, "meta": {
                    "intent": turn.get("intent"),
                    "tools_used": turn.get("tools_used", []),
                    "actions_taken": turn.get("actions_taken", []),
                    "summary": turn.get("conversation_summary", ""),
                }})
        return messages

    def _session_exists(self, session_id: str) -> bool:
        return os.path.exists(
            os.path.join(self.session_storage_path, f"{session_id}.json")
        )

    def _load_session(self, session_id: str) -> SessionState:
        filepath = os.path.join(self.session_storage_path, f"{session_id}.json")
        with open(filepath, "r") as f:
            data = json.load(f)
        return SessionState(**data)

    def _save_session(self) -> None:
        if not self.current_session:
            return
        filepath = os.path.join(
            self.session_storage_path,
            f"{self.current_session.session_id}.json",
        )
        session_dict = self.current_session.dict()

        def _serialize(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj

        with open(filepath, "w") as f:
            json.dump(session_dict, f, indent=2, default=_serialize)

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------

    def ingest_uploaded_document(self, filename: str, file_bytes: bytes) -> int:
        """
        Chunk, embed, and store an uploaded file in this session's ChromaDB collection.
        Returns the number of chunks ingested.
        """
        if not self.retriever:
            raise ValueError("No active session. Call start_session() first.")
        return self.retriever.ingest_file(filename, file_bytes)

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _get_checkpoint_state(self, config) -> dict:
        """Single SQLite read per turn — reused by both summary and history helpers."""
        if not self.current_session or not self.current_session.conversation_history:
            return {}
        try:
            return self.workflow.get_state(config).values
        except Exception:
            return {}

    def _get_conversation_summary(self, config) -> str:
        return self._get_checkpoint_state(config).get("conversation_summary", "No previous conversation.")

    def _get_conversation_history(self, config) -> List[BaseMessage]:
        return self._get_checkpoint_state(config).get("messages", [])

    # ------------------------------------------------------------------
    # Message processing
    # ------------------------------------------------------------------

    def process_message(self, user_input: str) -> Dict[str, Any]:
        if not self.current_session:
            raise ValueError("No active session. Call start_session() first.")

        config = {
            "configurable": {
                "thread_id": self.current_session.session_id,
                "llm": self.llm,
                "tools": self.tools,
            }
        }

        initial_state: AgentState = {
            "messages": [],
            "user_input": user_input,
            "intent": None,
            "next_step": "classify_intent",
            "conversation_summary": self._get_conversation_summary(config),
            "active_documents": self.current_session.document_context,
            "current_response": None,
            "tools_used": [],
            "session_id": self.current_session.session_id,
            "user_id": self.current_session.user_id,
            "actions_taken": [],
        }

        try:
            final_state = self.workflow.invoke(initial_state, config=config)

            raw_content = final_state.get("messages")[-1].content if final_state.get("messages") else None
            if isinstance(raw_content, list):
                raw_content = "".join(
                    b["text"] if isinstance(b, dict) and "text" in b else str(b)
                    for b in raw_content
                )

            if final_state.get("messages"):
                self.current_session.conversation_history.append({
                    "user_input": user_input,
                    "assistant_response": raw_content or "",
                    "intent": final_state.get("intent").dict() if final_state.get("intent") else None,
                    "actions_taken": final_state.get("actions_taken", []),
                    "tools_used": final_state.get("tools_used", []),
                    "conversation_summary": final_state.get("conversation_summary", ""),
                    "timestamp": datetime.now().isoformat(),
                })
                self.current_session.last_updated = datetime.now()
                if final_state.get("active_documents"):
                    self.current_session.document_context = list(set(
                        self.current_session.document_context
                        + final_state["active_documents"]
                    ))
                self._save_session()

            return {
                "success": True,
                "response": raw_content,
                "intent": final_state.get("intent").dict() if final_state.get("intent") else None,
                "tools_used": final_state.get("tools_used", []),
                "sources": final_state.get("active_documents", []),
                "actions_taken": final_state.get("actions_taken", []),
                "summary": final_state.get("conversation_summary", ""),
                "llm_provider": self.llm_provider,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": None,
            }
