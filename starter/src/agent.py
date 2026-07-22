import os
from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent, tools_condition, ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
import re
import operator
from schemas import UserIntent, UpdateMemoryResponse
from prompts import get_intent_classification_prompt, get_chat_prompt_template, MEMORY_SUMMARY_PROMPT
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

# TODO: The AgentState class is already implemented for you.  Study the
# structure to understand how state flows through the LangGraph
# workflow.  See README.md Task 2.1 for detailed explanations of
# each property.
class AgentState(TypedDict):
    """
    The agent state object
    """
    # Current conversation
    user_input: Optional[str]
    messages: Annotated[List[BaseMessage], add_messages]

    # Intent and routing
    intent: Optional[UserIntent]
    next_step: str

    # Memory and context
    conversation_summary: str
    active_documents: Optional[List[str]]

    # Current task state
    current_response: Optional[Dict[str, Any]]
    tools_used: List[str]

    # Session management
    session_id: Optional[str]
    user_id: Optional[str]

    # TODO: Modify actions_taken to use an operator.add reducer
    actions_taken: Annotated[List[str], operator.add]


# Keyed by (id(llm), id(tools)) — tools id changes when documents are uploaded,
# ensuring the cached agent always reflects the current retriever state.
# Capped at 8 entries to prevent unbounded growth across uploads.
_AGENT_CACHE: Dict[tuple, Any] = {}
_AGENT_CACHE_MAX = 8


def _get_react_agent(llm, tools):
    key = (id(llm), id(tools))
    if key not in _AGENT_CACHE:
        if len(_AGENT_CACHE) >= _AGENT_CACHE_MAX:
            _AGENT_CACHE.pop(next(iter(_AGENT_CACHE)))
        _AGENT_CACHE[key] = create_react_agent(model=llm, tools=tools)
    return _AGENT_CACHE[key]


def _trim_history(messages: List[BaseMessage], max_messages: int = 6) -> List[BaseMessage]:
    """Keep only the most recent N messages to prevent unbounded prompt growth."""
    return messages[-max_messages:] if len(messages) > max_messages else messages


def _run_react(agent, messages: List[BaseMessage]) -> tuple:
    result = agent.invoke({"messages": messages}, {"recursion_limit": 6})
    tools_used = [t.name for t in result.get("messages", []) if isinstance(t, ToolMessage) and t.name]
    return result, tools_used


_SUMMARIZE_KEYWORDS = {"summarize", "summary", "summarise", "brief", "overview", "tldr", "outline"}
_CALC_KEYWORDS = {"total", "sum", "calculate", "how much", "average", "how many", "tally", "grand total", "add up"}


def _fast_classify(user_input: str) -> Optional[str]:
    """Keyword-based fast path — returns intent_type or None to fall through to LLM."""
    text = user_input.lower()
    words = set(text.split())
    if words & _SUMMARIZE_KEYWORDS:
        return "summarization"
    if any(kw in text for kw in _CALC_KEYWORDS):
        return "calculation"
    return None


def classify_intent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Classify user intent and update next_step. Also records that this
    function executed by appending "classify_intent" to actions_taken.
    """
    user_input = state.get("user_input", "")

    # Fast keyword path — skips a Gemini call for obvious intents
    fast = _fast_classify(user_input)
    if fast:
        intent = UserIntent(intent_type=fast, confidence=0.9, reasoning="keyword match")
        return {
            "actions_taken": ["classify_intent"],
            "intent": intent,
            "next_step": f"{fast}_agent",
        }

    llm = config.get("configurable").get("llm")
    history = state.get("messages", [])
    structured_llm = llm.with_structured_output(UserIntent)

    def _msg_text(content):
        if isinstance(content, list):
            return " ".join(b["text"] if isinstance(b, dict) and "text" in b else str(b) for b in content)
        return str(content) if content else ""

    prompt = get_intent_classification_prompt().format(
        user_input=user_input,
        conversation_history="\n".join(
            [_msg_text(msg.content) for msg in history if hasattr(msg, "content")]
        )
    )

    intent = structured_llm.invoke(prompt)
    intent_type = intent.intent_type if intent.intent_type in ("qa", "summarization", "calculation") else "qa"

    return {
        "actions_taken": ["classify_intent"],
        "intent": intent,
        "next_step": f"{intent_type}_agent",
    }

def qa_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    llm = config.get("configurable").get("llm")
    tools = config.get("configurable").get("tools")
    messages = get_chat_prompt_template("qa").invoke({
        "input": state["user_input"],
        "chat_history": _trim_history(state.get("messages", [])),
    }).to_messages()
    result, tools_used = _run_react(_get_react_agent(llm, tools), messages)
    return {
        "messages": result.get("messages", []),
        "actions_taken": ["qa_agent"],
        "current_response": result,
        "tools_used": tools_used,
        "next_step": "update_memory" if tools_used else "end",
    }


def summarization_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    llm = config.get("configurable").get("llm")
    tools = config.get("configurable").get("tools")
    messages = get_chat_prompt_template("summarization").invoke({
        "input": state["user_input"],
        "chat_history": _trim_history(state.get("messages", [])),
    }).to_messages()
    result, tools_used = _run_react(_get_react_agent(llm, tools), messages)
    return {
        "messages": result.get("messages", []),
        "actions_taken": ["summarization_agent"],
        "current_response": result,
        "tools_used": tools_used,
        "next_step": "update_memory" if tools_used else "end",
    }


def calculation_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    llm = config.get("configurable").get("llm")
    tools = config.get("configurable").get("tools")
    messages = get_chat_prompt_template("calculation").invoke({
        "input": state["user_input"],
        "chat_history": _trim_history(state.get("messages", [])),
    }).to_messages()
    result, tools_used = _run_react(_get_react_agent(llm, tools), messages)
    return {
        "messages": result.get("messages", []),
        "actions_taken": ["calculation_agent"],
        "current_response": result,
        "tools_used": tools_used,
        "next_step": "update_memory" if tools_used else "end",
    }


# TODO: Finish implementing the update_memory function. Refer to README.md Task 2.4
def update_memory(state: AgentState,config: RunnableConfig) -> AgentState:
    """
    Update conversation memory and record the action.
    """

    llm = config.get("configurable").get("llm")

    prompt_with_history = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            MEMORY_SUMMARY_PROMPT
        ),
        MessagesPlaceholder("chat_history"),
    ]).invoke({
        "chat_history": _trim_history(state.get("messages", [])),
    })

    structured_llm = llm.with_structured_output(UpdateMemoryResponse)

    response = structured_llm.invoke(prompt_with_history)

    return {
        "actions_taken": ["update_memory"],
        "conversation_summary": response.summary,
        "active_documents": response.document_ids,
        "next_step": "end",
    }


def should_continue(state: AgentState) -> str:
    """Router function"""
    return state.get("next_step", "end")


def create_workflow(llm, tools):
    """
    Creates the LangGraph agents.
    Compiles the workflow with an InMemorySaver checkpointer to persist state.
    """

    workflow = StateGraph(AgentState)

    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("qa_agent", qa_agent)
    workflow.add_node("summarization_agent", summarization_agent)
    workflow.add_node("calculation_agent", calculation_agent)
    workflow.add_node("update_memory", update_memory)

    workflow.set_entry_point("classify_intent")

    workflow.add_conditional_edges(
        "classify_intent",
        should_continue,
        {
            "qa_agent": "qa_agent",
            "summarization_agent": "summarization_agent",
            "calculation_agent": "calculation_agent",
            "end": END,
        }
    )

    for node in ("qa_agent", "summarization_agent", "calculation_agent"):
        workflow.add_conditional_edges(node, should_continue, {
            "update_memory": "update_memory",
            "end": END,
        })

    workflow.add_edge("update_memory", END)

    db_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints.sqlite")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    checkpointer = SqliteSaver(conn)
    return workflow.compile(checkpointer=checkpointer)