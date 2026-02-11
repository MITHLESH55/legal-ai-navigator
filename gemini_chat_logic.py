# gemini_chat_logic.py
import os
import logging
from typing import TypedDict, List, Annotated, Literal, Sequence, Optional, Any
from operator import add
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, HumanMessage, AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool # For type hinting
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver # Using in-memory checkpointer (comes with langgraph)
from langgraph.types import interrupt, Command

# Import only the tools needed for /chat-g
from browser_tool import open_browser_search
from drive_tool import search_google_drive_files
# Import DB functions for saving messages WITHIN the graph flow
from db_utils import save_chat_message, get_chat_history

# Import Gemini File API uploader
import google.generativeai as genai
import time

logger = logging.getLogger(__name__)

# --- Gemini Configuration ---
GEMINI_MODEL_NAME = "gemini-1.5-flash"


# Tools specifically for the Gemini agent
gemini_tools: Sequence[BaseTool] = [
    open_browser_search,
    search_google_drive_files,
    # Grounding is enabled via ToolConfig below
]

# --- State Definition for Gemini Agent ---
# We use MessagesState directly as it handles history. Add other fields if needed.
class AgentState_G(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add] # Use Sequence and AnyMessage
    # Store approval state separately if needed, though interrupts handle pausing
    # pending_tool_approval_g: Optional[dict] = None

# --- Gemini LLM Initialization ---
def get_gemini_llm_with_config():
    """Initializes the Gemini LLM with grounding, tools, and thinking config."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")

    # Note: Grounding with Google Search is available through the google-generativeai SDK
    # directly. In LangChain, we'll configure the model and let it use its native capabilities.
    # The model will automatically use grounding when appropriate based on the query.
    
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME,
        google_api_key=api_key,
        temperature=0.1,
        convert_system_message_to_human=True, # Important for Gemini
        # Grounding is enabled natively in Gemini 2.0+ models
        # Additional configuration can be done via google.generativeai SDK if needed
    )
    # Bind the Browser and Drive tools
    return llm.bind_tools(gemini_tools)

gemini_llm_with_tools = get_gemini_llm_with_config()

# --- Nodes for Gemini LangGraph ---

# Tools requiring approval for THIS agent
TOOLS_REQUIRING_APPROVAL_G = {
    "open_browser_search",
    "search_google_drive_files"
}

gemini_tool_node = ToolNode(gemini_tools)

# --- Define Nodes for Gemini Agent Graph ---

# System prompt (optional but good practice)
SYSTEM_PROMPT_G = """You are a helpful legal AI assistant. You have the following capabilities:

1. **File Access**: You can directly read and analyze any files uploaded by the user (PDFs, images, etc.). When a file is provided in the conversation, you can access its full content and extract information from it.

2. **Tools**: You have access to tools for browser search and Google Drive search. Use these when you need external information or to search documents in Google Drive.

When a user uploads a file and asks about its contents, analyze the file directly and provide comprehensive answers. You have full access to read and interpret the file contents."""

def should_continue_g(state: AgentState_G) -> Literal["tools_g", "__end__"]:
    """Router to decide whether to call tools or end."""
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes tool calls, route to the tool node
    if last_message.tool_calls:
        logger.info("Decision: Routing to tools_g node.")
        return "tools_g"
    # Otherwise, respond back to the user
    logger.info("Decision: Routing to __end__ node.")
    return "__end__"

async def call_gemini_model_node(state: AgentState_G, config):
    """Invokes the Gemini model, adding system prompt if first turn."""
    logger.info("Node: call_gemini_model_node")
    messages = state['messages']

    # Add system prompt only if it's the beginning of the conversation
    # Check if system prompt already exists by looking for the prompt text
    has_system_prompt = False
    for m in messages:
        if isinstance(m, HumanMessage):
            # Handle both string content and list content (multimodal)
            if isinstance(m.content, str):
                if "You are a helpful" in m.content and "assistant" in m.content:
                    has_system_prompt = True
                    break
            elif isinstance(m.content, list):
                # Check if any text part contains the system prompt
                for part in m.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_content = part.get("text", "")
                        if "You are a helpful" in text_content and "assistant" in text_content:
                            has_system_prompt = True
                            break
                if has_system_prompt:
                    break
    
    if not has_system_prompt:
         # Convert system prompt to HumanMessage for Gemini compatibility
         invocation_messages = [HumanMessage(content=SYSTEM_PROMPT_G)] + list(messages)
         logger.info(f"Added system prompt to conversation. Total messages: {len(invocation_messages)}")
    else:
         invocation_messages = list(messages)
         logger.info(f"System prompt already exists. Total messages: {len(invocation_messages)}")

    response = await gemini_llm_with_tools.ainvoke(invocation_messages, config=config)
    # We return a list, because this will be added to the existing list
    return {"messages": [response]}

# We don't need a separate approval node with LangGraph's native interrupts.
# The main agent decides, and the ToolNode can be configured to interrupt.

# --- Create and Compile Gemini Agent Graph ---

# Use MemorySaver for LangGraph state persistence in memory
# Note: For production use with persistence across restarts, install langgraph-checkpoint-sqlite
# and use: from langgraph.checkpoint.sqlite import SqliteSaver
# gemini_memory = SqliteSaver.from_conn_string("gemini_agent_checkpoints.db")
gemini_memory = MemorySaver()

def create_gemini_agent_executor():
    """Creates the LangGraph agent executor for the Gemini endpoint."""
    workflow_g = StateGraph(AgentState_G)

    # Add nodes
    workflow_g.add_node("agent_g", call_gemini_model_node)
    workflow_g.add_node("tools", gemini_tool_node) # Using standard ToolNode - must be named "tools" for tools_condition

    # Define edges
    workflow_g.set_entry_point("agent_g")

    # Conditional edge: after agent call, decide to call tools or end
    workflow_g.add_conditional_edges(
        "agent_g",
        tools_condition, # Use LangGraph's built-in condition (expects node named "tools")
    )

    # Edge from tools back to agent to process results
    workflow_g.add_edge("tools", "agent_g")

    logger.info("Compiling Gemini agent executor with MemorySaver checkpointer...")
    # Compile with checkpointer for persistence and interrupt support
    # Note: tools_condition expects a node named "tools", so we use that name above
    app_g = workflow_g.compile(
        checkpointer=gemini_memory,
        # Interrupt before executing tools for human-in-the-loop approval
        interrupt_before=["tools"], # Must match the node name
    )
    logger.info("Gemini agent executor compiled.")
    return app_g

# Singleton instance for the Gemini agent
agent_executor_g = create_gemini_agent_executor()

# --- NEW: Direct Gemini Invocation with Files (bypasses LangGraph but keeps tools) ---
async def invoke_gemini_with_files_directly(
    user_message: str,
    gemini_file_objs: List[Any],
    session_id: str
) -> str:
    """
    Directly invoke Gemini API with files, bypassing LangGraph.
    This allows native file handling while maintaining access to browser and drive tools.
    
    Args:
        user_message: The user's text query
        gemini_file_objs: List of Gemini file objects from genai.upload_file()
        session_id: Session ID for history retrieval
    
    Returns:
        The model's response text
    """
    logger.info("Invoking Gemini directly with files (bypassing LangGraph, tools still available)")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found")
    
    genai.configure(api_key=api_key)
    
    # Convert LangChain tools to Gemini function declarations
    from google.ai.generativelanguage_v1beta.types import FunctionDeclaration, Schema, Type
    
    tool_declarations = []
    for tool in gemini_tools:
        # Get tool schema
        schema_dict = tool.args_schema.schema() if tool.args_schema else {"type": "object", "properties": {}}
        
        # Convert to Gemini format
        properties = {}
        required = schema_dict.get("required", [])
        
        for prop_name, prop_schema in schema_dict.get("properties", {}).items():
            prop_type = prop_schema.get("type", "string")
            # Map JSON schema types to Gemini types
            type_mapping = {
                "string": Type.STRING,
                "number": Type.NUMBER,
                "integer": Type.INTEGER,
                "boolean": Type.BOOLEAN,
                "array": Type.ARRAY,
                "object": Type.OBJECT
            }
            
            properties[prop_name] = Schema(
                type=type_mapping.get(prop_type, Type.STRING),
                description=prop_schema.get("description", "")
            )
        
        func_decl = FunctionDeclaration(
            name=tool.name,
            description=tool.description,
            parameters=Schema(
                type=Type.OBJECT,
                properties=properties,
                required=required
            )
        )
        tool_declarations.append(func_decl)
    
    # Create model with tools
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL_NAME,
        tools=tool_declarations,
        system_instruction=SYSTEM_PROMPT_G
    )
    
    # Get conversation history
    history_messages = get_chat_history(session_id)
    history = []
    for msg_data in history_messages:
        role = "user" if msg_data.get("role") == "user" else "model"
        content = msg_data.get("content", "")
        if content:
            history.append({"role": role, "parts": [content]})
    
    # Start chat with history
    chat = model.start_chat(history=history)
    
    # Prepare message: files first, then text
    message_parts = gemini_file_objs + [user_message]
    
    logger.info(f"Sending {len(gemini_file_objs)} file(s) + text to Gemini with {len(tool_declarations)} tools available")
    
    try:
        # Send message
        response = chat.send_message(message_parts)
        
        # Check if model wants to use tools
        max_tool_iterations = 5
        iteration = 0
        
        while iteration < max_tool_iterations:
            if not response.candidates or not response.candidates[0].content.parts:
                break
                
            # Check for function calls
            function_calls = []
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    function_calls.append(part.function_call)
            
            if not function_calls:
                # No more tool calls, return the text response
                break
            
            # Execute all function calls
            function_responses = []
            for fc in function_calls:
                logger.info(f"Gemini called tool: {fc.name} with args: {dict(fc.args)}")
                
                # Find and execute the tool
                tool_result = None
                for tool in gemini_tools:
                    if tool.name == fc.name:
                        try:
                            # Execute the tool
                            tool_result = tool.func(**dict(fc.args))
                            logger.info(f"Tool {fc.name} returned: {str(tool_result)[:200]}...")
                        except Exception as tool_err:
                            logger.error(f"Error executing tool {fc.name}: {tool_err}")
                            tool_result = f"Error: {str(tool_err)}"
                        break
                
                if tool_result is None:
                    tool_result = f"Tool {fc.name} not found"
                
                # Create function response
                from google.ai.generativelanguage_v1beta.types import FunctionResponse, Part
                function_responses.append(
                    Part(
                        function_response=FunctionResponse(
                            name=fc.name,
                            response={"result": str(tool_result)}
                        )
                    )
                )
            
            # Send function responses back to model
            response = chat.send_message(function_responses)
            iteration += 1
        
        # Extract final text response
        if response.candidates and response.candidates[0].content.parts:
            text_parts = []
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    text_parts.append(part.text)
            
            final_response = "".join(text_parts)
            logger.info(f"Gemini response: {final_response[:200]}...")
            return final_response
        else:
            logger.warning("No text response from Gemini")
            return "I apologize, but I couldn't generate a response."
    
    except Exception as e:
        logger.error(f"Error invoking Gemini with files: {e}", exc_info=True)
        raise

# --- Gemini File API Helper ---
def upload_file_to_gemini(file_path: str, display_name: Optional[str] = None) -> Optional[Any]:
    """Uploads a file to the Gemini API, waits for processing, and returns the file object or None."""
    logger.info(f"Uploading file to Gemini API: {file_path}")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not set for file upload.")
        return None

    # Configure the genai client (only needs to be done once, but safe to repeat)
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to configure google.generativeai: {e}")
        return None

    if not display_name:
        display_name = os.path.basename(file_path)

    try:
        logger.info(f"Attempting upload for {display_name} from {file_path}...")
        # upload_file handles opening and reading the file
        file_obj = genai.upload_file(path=file_path, display_name=display_name)
        logger.info(f"File submitted for upload: {file_obj.name} ({file_obj.display_name}). Initial state: {file_obj.state.name}")

        # Wait for the file to become ACTIVE
        wait_time = 5
        total_wait = 0
        max_wait = 120 # Wait up to 2 minutes
        while file_obj.state.name == "PROCESSING" and total_wait < max_wait:
            logger.info(f"Waiting {wait_time}s for Gemini file processing...")
            time.sleep(wait_time)
            total_wait += wait_time
            try:
                file_obj = genai.get_file(name=file_obj.name) # Refresh state
                logger.info(f"Current file state: {file_obj.state.name}")
            except Exception as get_e:
                logger.error(f"Error refreshing file state for {file_obj.name}: {get_e}")
                # Decide how to handle this - maybe break and assume failure?
                break # Exit loop on error fetching status

        if file_obj.state.name == "ACTIVE":
            logger.info(f"Gemini file processing complete and ACTIVE: {file_obj.name}")
            return file_obj
        elif file_obj.state.name == "FAILED":
            logger.error(f"Gemini file processing FAILED for {file_path}. Error: {getattr(file_obj, 'error', 'Unknown')}")
            # Optionally delete the failed file from Gemini backend
            try:
                 genai.delete_file(name=file_obj.name)
                 logger.info(f"Deleted failed Gemini file {file_obj.name}")
            except Exception as del_e:
                 logger.error(f"Could not delete failed Gemini file {file_obj.name}: {del_e}")
            return None
        else:
             logger.warning(f"Gemini file {file_obj.name} finished in unexpected state: {file_obj.state.name} after {total_wait}s")
             return None # Treat PROCESSING after timeout as failure for now

    except Exception as e:
        logger.error(f"Error uploading file '{display_name}' to Gemini: {e}", exc_info=True)
        return None
