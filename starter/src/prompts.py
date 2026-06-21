from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate


def get_intent_classification_prompt() -> PromptTemplate:
    """
    Get the intent classification prompt template.
    """
    return PromptTemplate(
        input_variables=["user_input", "conversation_history"],
        template="""You are an intent classifier for a document processing assistant.

Classify the user's intent into exactly one of these categories:

- qa: The user wants to look up a fact, retrieve specific information, or ask a question about document content that does NOT require arithmetic. Examples: "Who is the client on invoice INV-001?", "What is the status of claim CLM-001?", "When does the service agreement expire?"

- summarization: The user wants a summary, overview, or list of key points extracted from one or more documents, without performing calculations. Examples: "Summarize all contracts", "Give me an overview of the insurance claims", "What are the key points in the service agreement?"

- calculation: The user wants a numerical result that requires arithmetic, aggregation, or comparison of amounts across documents. Examples: "What is the total of all invoices?", "How much tax was charged on INV-002?", "What is the average invoice amount?", "Calculate the sum of all claims."

- unknown: The intent cannot be determined from the input.

Instructions:
1. Read the user input carefully.
2. Check the conversation history for context if the input is ambiguous (e.g., "what about that one?" may refer to a prior document).
3. Choose the single best-matching category.
4. Assign a confidence score between 0.0 (completely uncertain) and 1.0 (completely certain).
5. Write a one-sentence reasoning explaining your choice.

User Input: {user_input}

Recent Conversation History:
{conversation_history}

Respond with the intent_type, confidence score, and reasoning.
"""
    )


# Q&A System Prompt
QA_SYSTEM_PROMPT = """You are a helpful document assistant specializing in answering questions about financial and healthcare documents.

Your capabilities:
- Answer specific questions about document content
- Cite sources accurately
- Provide clear, concise answers
- Use available tools to search and read documents

Guidelines:
1. Always search for relevant documents before answering
2. Cite specific document IDs when referencing information
3. If information is not found, say so clearly
4. Be precise with numbers and dates
5. Maintain professional tone

"""

# Summarization System Prompt
SUMMARIZATION_SYSTEM_PROMPT = """You are an expert document summarizer specializing in financial and healthcare documents.

Your approach:
- Extract key information and main points
- Organize summaries logically
- Highlight important numbers, dates, and parties
- Keep summaries concise but comprehensive

Guidelines:
1. First search for and read the relevant documents
2. Structure summaries with clear sections
3. Include document IDs in your summary
4. Focus on actionable information
"""

# Calculation System Prompt
# TODO: Implement the CALCULATION_SYSTEM_PROMPT. Refer to README.md Task 3.2 for details
CALCULATION_SYSTEM_PROMPT = """
You are a calculation assistant specializing in financial and healthcare documents.

Your responsibilities:
- Determine which document(s) are relevant to the user's request.
- Retrieve and review the necessary documents before performing any calculations.
- Identify the mathematical expression required to answer the user's question.
- Use the calculator tool for ALL calculations.
- Never perform arithmetic mentally, even for simple calculations.
- Explain how the calculation was derived from the document information.

Guidelines:
1. Search for and read relevant documents first.
2. Extract all values needed for the calculation.
3. Construct the mathematical expression clearly.
4. Use the calculator tool for every calculation.
5. Verify the result before responding.
6. Cite the document IDs used in the calculation.
7. If required information is missing, explain what information is needed.
"""


# TODO: Finish the function to return the correct prompt based on intent type
# Refer to README.md Task 3.1 for details
def get_chat_prompt_template(intent_type: str) -> ChatPromptTemplate:
    """
    Get the appropriate chat prompt template based on intent.
    """
    if intent_type == "qa":
        system_prompt = QA_SYSTEM_PROMPT

    elif intent_type == "summarization":
        system_prompt = SUMMARIZATION_SYSTEM_PROMPT

    elif intent_type == "calculation":
        system_prompt = CALCULATION_SYSTEM_PROMPT

    else:
        system_prompt = QA_SYSTEM_PROMPT  

    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        MessagesPlaceholder("chat_history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
# Memory Summary Prompt
MEMORY_SUMMARY_PROMPT = """Summarize the following conversation history into a concise summary:

Focus on:
- Key topics discussed
- Documents referenced
- Important findings or calculations
- Any unresolved questions
"""
