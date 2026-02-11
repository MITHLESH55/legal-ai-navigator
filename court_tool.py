"""
Court supervisor tool - wraps the multi-agent court system as a single tool.
"""

import logging
from langchain.tools import tool
from supervisor_agent import court_supervisor, SupervisorState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@tool
def search_court_cases(query: str, case_id: str = None) -> str:
    """
    Search for court cases across District Courts, Supreme Court, and Consumer Forums.
    
    This intelligent tool automatically:
    1. Determines which court type your query is about
    2. Routes to the appropriate specialized court agent
    3. May ask you to clarify which court type if ambiguous
    
    Use this tool when the user asks about:
    - Any court case (District, Supreme, or Consumer Forum)
    - CNR numbers, diary numbers, case numbers
    - Party names, advocate names, filing numbers
    - Cause lists, orders, hearings
    - Court locations and information
    
    Examples:
    - "Find cases involving XYZ party in 2023"
    - "Get Supreme Court diary number 12345/2023"
    - "Show me consumer case CC/123/2023"
    - "What's the cause list for tomorrow?"
    """
    logger.info(f"Tool: search_court_cases, query: {query}")
    
    try:
        # Create supervisor state
        state = SupervisorState(
            input=query,
            case_id=case_id or "",
            messages=[],
            court_type="unknown",
            summary="",
            needs_clarification=False
        )
        
        # Create config with thread ID for memory
        config = {
            "configurable": {
                "thread_id": case_id or "default_thread"
            }
        }
        
        # Invoke supervisor
        result = court_supervisor.invoke(state, config=config)
        
        # Check if there's an interrupt (human-in-the-loop needed)
        if "__interrupt__" in result:
            interrupt_data = result["__interrupt__"][0]
            # Format the clarification request nicely
            return f"""
I need clarification: {interrupt_data.value['question']}

Options: {', '.join(interrupt_data.value['options'])}

Your query: {interrupt_data.value['query']}

Please specify which court type and ask your question again.
"""
        
        # Return the summary
        return result.get('summary', 'No response from court system')
        
    except Exception as e:
        logger.error(f"Error in search_court_cases: {e}")
        return f"Error searching court cases: {str(e)}"
