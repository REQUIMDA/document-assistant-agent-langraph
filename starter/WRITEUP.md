# Report-Building Agent - Student Writeup

## a. Architecture & Routing Decisions

The system uses a LangGraph StateGraph with five nodes connected in a conditional pipeline:

    classify_intent --> [qa_agent | summarization_agent | calculation_agent] --> update_memory --> END

**Why LangGraph?**
LangGraph StateGraph makes routing explicit and inspectable. Each node receives the full AgentState and returns only the fields it modifies. The Annotated[List[str], operator.add] reducer on actions_taken means each node appends its name without overwriting what prior nodes wrote, so the full execution path is preserved in state.

**Routing logic (should_continue):**
After classify_intent runs it sets next_step to one of "qa_agent", "summarization_agent", or "calculation_agent". add_conditional_edges reads that value and dispatches to the correct node. After any agent node runs it sets next_step = "update_memory", so a direct add_edge handles those transitions.

**Why one entry point?**
All queries pass through classify_intent first. This ensures every turn is categorised before any expensive LLM + tool calls happen, and the intent classification drives which system prompt and structured output schema are used downstream.

**ReAct sub-agents:**
Each task agent (qa, summarization, calculation) uses create_react_agent from langgraph.prebuilt. This gives each agent its own tool-calling loop across the four tools (calculator, document_search, document_reader, document_statistics), while still returning a strongly-typed Pydantic response enforced by response_format=Schema.

**Graph definition (from agent.py):**

    workflow = StateGraph(AgentState)
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("qa_agent", qa_agent)
    workflow.add_node("summarization_agent", summarization_agent)
    workflow.add_node("calculation_agent", calculation_agent)
    workflow.add_node("update_memory", update_memory)
    workflow.set_entry_point("classify_intent")
    workflow.add_conditional_edges("classify_intent", should_continue, {
        "qa_agent": "qa_agent",
        "summarization_agent": "summarization_agent",
        "calculation_agent": "calculation_agent",
        "end": END,
    })
    workflow.add_edge("qa_agent", "update_memory")
    workflow.add_edge("summarization_agent", "update_memory")
    workflow.add_edge("calculation_agent", "update_memory")
    workflow.add_edge("update_memory", END)
    return workflow.compile(checkpointer=InMemorySaver())

---

## b. State & Memory

**State object (AgentState):**

| Field | Type | Purpose |
|---|---|---|
| user_input | str | Raw query for the current turn |
| messages | Annotated[List[BaseMessage], add_messages] | Full message list, merged across turns by LangGraph |
| intent | UserIntent | Pydantic model: intent_type, confidence, reasoning |
| next_step | str | Router signal read by should_continue |
| conversation_summary | str | Rolling summary produced by update_memory |
| active_documents | List[str] | Document IDs referenced across turns |
| current_response | Dict | Raw output from the last agent node |
| tools_used | List[str] | Names of tools called this turn |
| actions_taken | Annotated[List[str], operator.add] | Append-only audit log of every node executed |
| session_id / user_id | str | Links the LangGraph thread to the persisted session file |

**Within-session memory (LangGraph checkpointer):**
InMemorySaver is passed to workflow.compile(). Every workflow.invoke() call uses config["configurable"]["thread_id"] = session_id, so LangGraph replays and extends the message history for that thread across multiple turns within the same process run. The add_messages reducer on the messages field merges new messages into the existing list rather than overwriting it.

**Cross-session memory (file persistence):**
DocumentAssistant._save_session() writes a JSON file to ./sessions/<session_id>.json after every turn. It stores a serialisable summary of each turn: user input, classified intent, actions taken, tools used, conversation summary, and timestamp. On restart, start_session(user_id, session_id) reloads this file so document context carries over.

**Memory summarisation (update_memory node):**
After every agent response, update_memory calls the LLM with the full message history and MEMORY_SUMMARY_PROMPT, producing an UpdateMemoryResponse with a rolling summary string and a list of document_ids. The summary is stored in AgentState.conversation_summary and injected into subsequent prompts via _get_conversation_summary(), preventing unbounded context growth over long sessions.

---

## c. Structured Output

Structured outputs are enforced at two levels:

**1. Intent classification (classify_intent):**

    structured_llm = llm.with_structured_output(UserIntent)
    intent = structured_llm.invoke(prompt)

UserIntent is a Pydantic model with a Literal["qa","summarization","calculation","unknown"] field, a float confidence score constrained to ge=0.0, le=1.0, and a str reasoning field. LangChain .with_structured_output() uses OpenAI function-calling, so the model is forced to return a JSON object matching the schema. Invalid values raise a ValidationError before any routing happens.

**2. Agent responses (qa / summarization / calculation):**
Each ReAct agent is created with response_format=Schema:

    agent = create_react_agent(model=llm_with_tools, tools=tools, response_format=AnswerResponse)

| Agent | Schema | Key enforced fields |
|---|---|---|
| qa_agent | AnswerResponse | question: str, answer: str, sources: List[str], confidence: float (0-1), timestamp: datetime |
| summarization_agent | SummarizationResponse | summary: str, key_points: List[str], document_ids: List[str], original_length: int |
| calculation_agent | CalculationResponse | expression: str, result: float, explanation: str, units: Optional[str] |
| update_memory | UpdateMemoryResponse | summary: str, document_ids: List[str] |

All schemas use pydantic.Field with descriptions and range constraints. Downstream code can safely access .answer, .result, .summary etc. without any string parsing.

---

## d. Example Conversations

### Example 1 - Q&A

    User:       What is the total amount in invoice INV-001?
    Intent:     qa (confidence: 0.97)
    Tools used: document_reader
    Actions:    classify_intent -> qa_agent -> update_memory

    Assistant:  Invoice INV-001 (Invoice #12345) for Acme Corporation:
                  - Consulting Services:      5000.00
                  - Software Development:    12500.00
                  - Support and Maintenance:  2500.00
                  - Subtotal:                20000.00
                  - Tax (10%):               2000.00
                  - Total Due:              22000.00
                Source: INV-001  |  Confidence: 0.97

---

### Example 2 - Summarization

    User:       Summarize all contracts in the system
    Intent:     summarization (confidence: 0.95)
    Tools used: document_search, document_reader
    Actions:    classify_intent -> summarization_agent -> update_memory

    Assistant:  CONTRACT SUMMARY - CON-001 (Service Agreement)
                  Parties:   DocDacity Solutions Inc. <-> Healthcare Partners LLC
                  Term:      12 months from January 1, 2024
                  Value:     180000.00 total (15000.00/month)
                  Services:  Document Processing, 24/7 Support, Analytics, Compliance
                  Exit:      60 days written notice from either party
                Key points:
                  - Total committed spend: 180000.00
                  - Runs January 2024 through December 2024

---

### Example 3 - Calculation

    User:       What is the combined total of all invoices?
    Intent:     calculation (confidence: 0.99)
    Tools used: document_search, document_reader, calculator
    Actions:    classify_intent -> calculation_agent -> update_memory

    Assistant:  Retrieved all invoices and extracted totals:
                  INV-001 (Acme Corporation):   22000.00
                  INV-002 (TechStart Inc.):     69300.00
                  INV-003 (Global Corp):       214500.00

                Calculator input:   22000 + 69300 + 214500
                Calculator result:  305800

                Combined total of all invoices: 305800.00
                Sources: INV-001, INV-002, INV-003

---

### Example 4 - Multi-turn conversation (memory in action)

    Turn 1
    User:      What documents mention Acme Corporation?
    Intent:    qa (confidence: 0.96)
    Tools:     document_search
    Actions:   classify_intent -> qa_agent -> update_memory
    Response:  INV-001 (Invoice #12345) is billed to Acme Corporation for
               consulting, software development, and support. Total due: 22000.00.

    Turn 2
    User:      What was the tax amount on that invoice?
    Intent:    qa (confidence: 0.98)
    Tools:     document_reader
    Actions:   classify_intent -> qa_agent -> update_memory
    Note:      "that invoice" resolved via conversation_summary, which recorded
               INV-001 as the active document from Turn 1.
    Response:  The tax on INV-001 was 2000.00 (10% of the 20000.00 subtotal).
