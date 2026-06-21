# Document Assistant

A conversational document processing assistant built with LangChain and LangGraph. It can answer questions about documents, summarize content, and perform calculations on financial data — all through a natural language interface.

## Getting Started

### Dependencies

- Python 3.9 or higher
- An OpenAI API key
- The packages listed in `starter/requirements.txt`

### Installation

1. Navigate to the project folder:

```
cd starter
```

2. Create and activate a virtual environment:

```
python -m venv venv
venv\Scripts\activate
```

3. Install the required packages:

```
pip install -r requirements.txt
```

4. Set up your environment file:

```
cp .env.example .env
```

Open `.env` and fill in your OpenAI API key.

### Running the Assistant

```
python main.py
```

You will be prompted for a user ID, then you can start typing queries. Type `/help` for available commands and `/quit` to exit.

## Project Structure

```
starter/
├── src/
│   ├── schemas.py        # Pydantic response models
│   ├── retrieval.py      # Document retrieval logic
│   ├── tools.py          # LangChain tools (calculator, search, reader)
│   ├── prompts.py        # Prompt templates for each agent
│   ├── agent.py          # LangGraph workflow and node functions
│   └── assistant.py      # Top-level assistant class
├── logs/                 # Auto-generated tool call history per session
├── sessions/             # Auto-generated session state files
├── docs/
│   └── langgraph_agent_architecture.png
├── main.py               # Entry point
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variable template
└── WRITEUP.md            # Implementation documentation
```

## Agent Architecture

The assistant routes every user message through a LangGraph workflow:

![LangGraph agent architecture](./docs/langgraph_agent_architecture.png)

```
classify_intent --> [qa_agent | summarization_agent | calculation_agent] --> update_memory --> END
```

Each incoming message is first classified by intent, then handed to the appropriate agent, and finally the conversation memory is updated before the turn ends.

## What It Can Do

- **Answer questions** about specific documents — amounts, dates, parties, status
- **Summarize** one or more documents into structured key points
- **Calculate** totals, averages, or any arithmetic across document values
- **Remember context** across turns in a session using LangGraph's checkpointer

## Built With

- [LangChain](https://www.langchain.com) - LLM orchestration and tool definitions
- [LangGraph](https://langchain-ai.github.io/langgraph) - Stateful multi-agent workflow
- [OpenAI](https://openai.com) - Underlying language model (gpt-4o)
- [Pydantic](https://docs.pydantic.dev) - Structured output schemas

## License

[License](../LICENSE.md)
