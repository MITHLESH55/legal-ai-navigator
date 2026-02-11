import os
import logging
from typing import TypedDict, List, Annotated, Literal
from operator import itemgetter, add  # <-- Import 'add'

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from llm_factory import get_llm  # <-- Use LLM factory
from tools import all_tools  # <-- Imports all 6 tools: uploaded docs, public search, knowledge graph, court cases, Google Drive, browser redirect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- 1. Define the Agent State ---
class AgentState(MessagesState):
    """Extended state for the agent with additional fields."""
    input: str
    case_id: str  # Add this field for document filtering
    pending_tool_approval: dict = None  # For human-in-the-loop


# --- 2. Define the Graph Nodes ---

# --- 2. Define the Graph Nodes ---

# Get the LLM with tool binding
llm = get_llm(model_size="large", temperature=0)
llm_with_tools = llm.bind_tools(all_tools)

# System prompt for the agent
SYSTEM_PROMPT = """
You are VoiceLegal, an expert Indian legal assistant AI with a ReAct (Reasoning + Acting) workflow.

**HOW YOU WORK (ReAct Loop):**
1. **Reason**: Analyze the user's query
2. **Act**: Call tools to gather information (if needed)
3. **Observe**: Receive tool results
4. **Reason Again**: Process the results and decide next action
5. **Act Again**: Either call more tools OR provide final answer

**CRITICAL: Always process tool results before responding to the user!**

**ROUTING RULES:**

1. **ANSWER DIRECTLY (NO TOOLS)** for:
   - General legal definitions (e.g., "What is IPC 302?", "Explain habeas corpus")
   - Legal procedures (e.g., "How to file an FIR?", "What is bail?")
   - Constitutional articles, fundamental rights
   - Basic legal concepts everyone should know
   - Simple statements and memory (e.g., "Remember this", "Note that", "What did I say?")
   - Follow-up questions about previous conversation
   
   👉 If you can answer from your training OR conversation history, DO NOT call tools.

2. **USE TOOLS AND PROCESS RESULTS** for queries requiring external data:
   
   **IMPORTANT ReAct Pattern:**
   ```
   User: "Fetch from Google Drive and summarize in 2 lines"
   
   Step 1 (Act): Call search_google_drive_files
   Step 2 (Observe): Receive: {"status": "Success", "files": [{"name": "legal.pdf", "id": "123", ...}]}
   Step 3 (Reason): I now have file information. The user wants a summary.
   Step 4 (Act): Based on the file name "legal.pdf" and metadata, provide a 2-line summary:
                  "Found 1 legal document in your Drive folder: legal.pdf (68KB, uploaded on Oct 22).
                   This appears to be a legal document ready for review."
   ```
   
   **More Examples:**
   - "Find similar cases and explain" →
     * Call search_public_cases_and_news_serpapi
     * Receive results
     * Analyze and explain the findings
   
   - "Who was plaintiff in my case?" →
     * Call search_knowledge_graph
     * Receive entities
     * Answer with the plaintiff's name
   
   **Available Tools:**
   
   📄 **Private Documents:**
   - `search_uploaded_documents`: Search user's uploaded case files
   - `search_knowledge_graph`: Query extracted entities (judges, advocates, parties)
   
   🌐 **Public Information:**
   - `search_public_cases_and_news_serpapi`: Search web for legal news, similar cases
   
   ⚖️ **Court Cases:**
   - `search_court_cases`: Intelligent court case lookup (District/Supreme/Consumer)
     Pass the user's EXACT query as parameter
   
   ☁️ **Google Drive:**
   - `search_google_drive_files`: List PDF files in authorized Drive folder
     * DEFAULT: auto_sync=False (only lists files)
     * AFTER APPROVAL: Call again with auto_sync=True to download and index NEW files
     * User must approve before files are downloaded and processed into Qdrant/Neo4j
   
   **Google Drive Workflow:**
   1. User asks: "fetch from drive"
   2. Request approval to access Drive
   3. User approves → Call search_google_drive_files(auto_sync=False) first
   4. Observe results, then call search_google_drive_files(auto_sync=True) to process NEW files
   5. Summarize what was found and processed
   
   🌐 **Browser:**
   - `open_browser_search`: Open external browser for user (when explicitly requested)

**Response Style:**
- Be direct and confident
- NO phrases like "I'm happy to help", "our team", "I can tell you"
- NO disclaimers about being an AI or consulting a lawyer
- Process ALL tool results before responding
- Provide actionable information based on what you learned

**Remember: The loop continues until you have enough information to answer the user's question completely!**
"""

tool_node = ToolNode(all_tools)

# Tools that require human confirmation
TOOLS_REQUIRING_APPROVAL = {
    "search_google_drive_files",
    "open_browser_search"
}

# --- 3. Define the Graph Nodes ---

def request_tool_approval(state: AgentState):
    """
    Request human approval for sensitive tool calls using a small LLM.
    This node pauses execution and waits for user confirmation.
    """
    messages = state.get("messages", [])
    last_message = messages[-1]
    
    # Get tool calls requiring approval
    tools_to_approve = []
    for tool_call in last_message.tool_calls:
        if tool_call["name"] in TOOLS_REQUIRING_APPROVAL:
            tools_to_approve.append(tool_call)
    
    if not tools_to_approve:
        return {}
    
    # Use small model to generate a clear approval request
    small_llm = get_llm(model_size="small", temperature=0)
    
    # Build approval message
    approval_messages = []
    for tool_call in tools_to_approve:
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})
        
        # Generate user-friendly message using small LLM
        prompt = f"""Generate a clear, concise approval request for this action. 
Be brief and specific. No extra explanation.

Tool: {tool_name}
Arguments: {tool_args}

Format:
"Would you like me to [action description]? (Yes/No)"

Examples:
- "Would you like me to search your Google Drive for 'contract.pdf'? (Yes/No)"
- "Would you like me to open a browser search for 'Supreme Court judgments IPC 302'? (Yes/No)"
"""
        
        approval_msg = small_llm.invoke(prompt).content.strip()
        approval_messages.append({
            "tool_call_id": tool_call["id"],
            "tool_name": tool_name,
            "tool_args": tool_args,
            "approval_message": approval_msg
        })
    
    # Interrupt and wait for user decision
    logger.info(f"Requesting approval for {len(approval_messages)} tool(s)")
    
    user_decision = interrupt({
        "type": "tool_approval_request",
        "tools": approval_messages,
        "message": "The following actions require your approval:"
    })
    
    # User decision should be a dict like: {"approved": True/False}
    return {"pending_tool_approval": user_decision}

def call_model(state: AgentState):
    """Calls the LLM with tools."""
    logger.info("Node: call_model")
    
    # Check if we're handling approval response
    approval_decision = state.get("pending_tool_approval")
    if approval_decision:
        # If user denied approval, send message back to agent
        if not approval_decision.get("approved", False):
            logger.info("User denied tool approval")
            return {
                "messages": [
                    AIMessage(content="I understand. Let me know if you'd like me to help with something else.")
                ],
                "pending_tool_approval": None
            }
        else:
            logger.info("User approved tool execution - clearing approval flag")
            # Just clear the approval flag and continue to invoke the model
            # The tools will execute via the graph edge, and we'll be called again with results
            return {"pending_tool_approval": None}
    
    # Build messages list with system prompt and conversation history
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    
    # Add the user input if this is the first call
    if state.get("input"):
        messages.append(HumanMessage(content=state["input"]))
    
    # Add any existing conversation history (includes tool results)
    if state.get("messages"):
        messages.extend(state["messages"])
    
    # Call the LLM with bound tools
    logger.info(f"Calling LLM with {len(messages)} messages in history")
    response = llm_with_tools.invoke(messages)
    
    # Log what the agent decided
    if hasattr(response, "tool_calls") and response.tool_calls:
        logger.info(f"Agent decided to call {len(response.tool_calls)} tool(s): {[tc['name'] for tc in response.tool_calls]}")
    else:
        logger.info("Agent decided to respond directly (no tool calls)")
    
    # Return the response to be added to messages
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Router function to decide if we should call tools or end."""
    messages = state.get("messages", [])
    if not messages:
        return END
    
    last_message = messages[-1]
    
    # If the LLM makes tool calls, check if approval is needed
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        # Check if any tool requires approval
        needs_approval = any(
            tool_call["name"] in TOOLS_REQUIRING_APPROVAL 
            for tool_call in last_message.tool_calls
        )
        
        if needs_approval:
            logger.info("Decision: Request approval for sensitive tools")
            return "request_approval"
        else:
            logger.info("Decision: Execute tools directly")
            return "tools"
    
    # Otherwise, we're done
    logger.info("Decision: End (no tools needed)")
    return END


def create_agent_executor():
    workflow = StateGraph(AgentState)

    # Define nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("request_approval", request_tool_approval)
    workflow.add_node("tools", tool_node)

    # Set entry point
    workflow.set_entry_point("agent")

    # Add conditional edges from agent
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "request_approval": "request_approval",
            "tools": "tools",
            END: END
        }
    )

    # After approval, execute tools
    workflow.add_edge("request_approval", "tools")
    
    # After tools are called, go back to the agent
    workflow.add_edge("tools", "agent")

    # Add memory checkpointer for conversation persistence and interrupts
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    return app


# Singleton instance of the agent graph with memory
agent_executor = create_agent_executor()