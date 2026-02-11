"""
Supervisor agent that coordinates between court-specific agents.
Handles routing and human-in-the-loop confirmation.
"""

import logging
from typing import TypedDict, Annotated, List, Literal
from operator import add

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

from llm_factory import get_llm
from court_agents import district_court_agent, supreme_court_agent, consumer_forum_agent, CourtAgentState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Supervisor State ---
class SupervisorState(TypedDict):
    input: str
    case_id: str
    messages: Annotated[List[BaseMessage], add]
    court_type: str  # "district", "supreme", "consumer", or "unknown"
    summary: str
    needs_clarification: bool


# --- Helper: Determine Court Type ---
def determine_court_type(state: SupervisorState) -> str:
    """
    Analyzes the query to determine which court type is relevant.
    Returns: "district", "supreme", "consumer", or "needs_clarification"
    """
    query = state['input'].lower()
    
    # Check for explicit mentions
    if any(word in query for word in ["supreme court", "sc ", "diary number", "aor"]):
        logger.info("Detected: Supreme Court query")
        return "supreme"
    
    if any(word in query for word in ["consumer forum", "consumer case", "consumer complaint"]):
        logger.info("Detected: Consumer Forum query")
        return "consumer"
    
    if any(word in query for word in ["district court", "cnr", "filing number", "cause list"]):
        logger.info("Detected: District Court query")
        return "district"
    
    # Default to district court for ambiguous case queries
    if any(word in query for word in ["case", "advocate", "party", "petitioner", "defendant"]):
        logger.info("Defaulting to: District Court (ambiguous case query)")
        return "district"
    
    logger.info("Court type unclear - needs clarification")
    return "needs_clarification"


# --- Node: Route to Court Agent ---
def route_to_court_agent(state: SupervisorState):
    """Routes the query to the appropriate court agent"""
    logger.info(f"Node: route_to_court_agent (type: {state['court_type']})")
    
    court_type = state['court_type']
    
    # Select the appropriate agent
    if court_type == "district":
        agent = district_court_agent
    elif court_type == "supreme":
        agent = supreme_court_agent
    elif court_type == "consumer":
        agent = consumer_forum_agent
    else:
        return {"summary": "Could not determine court type. Please specify if this is a District Court, Supreme Court, or Consumer Forum query."}
    
    # Invoke the subgraph
    court_state = CourtAgentState(
        input=state['input'],
        case_id=state.get('case_id', ''),
        agent_scratchpad=[],
        summary='',
        court_type=court_type
    )
    
    result = agent.invoke(court_state)
    
    return {
        "summary": result.get('summary', 'No response from court agent'),
        "messages": [AIMessage(content=result.get('summary', 'No response'))]
    }


# --- Node: Ask for Clarification (Human-in-the-Loop) ---
def ask_court_clarification(state: SupervisorState):
    """
    Uses human-in-the-loop to ask user which court type they're asking about.
    """
    logger.info("Node: ask_court_clarification (Human-in-the-loop)")
    
    # Interrupt and ask user
    clarification_question = {
        "question": "Which court type is this query about?",
        "options": ["District Court", "Supreme Court", "Consumer Forum"],
        "query": state['input']
    }
    
    # This will pause execution and wait for user input
    user_choice = interrupt(clarification_question)
    
    logger.info(f"User selected: {user_choice}")
    
    # Map user choice to court type
    court_type_map = {
        "District Court": "district",
        "Supreme Court": "supreme",
        "Consumer Forum": "consumer",
        "district": "district",
        "supreme": "supreme",
        "consumer": "consumer"
    }
    
    selected_court = court_type_map.get(user_choice, "district")
    
    return {
        "court_type": selected_court,
        "needs_clarification": False,
        "messages": [HumanMessage(content=f"User selected: {user_choice}")]
    }


# --- Node: Analyze Query ---
def analyze_query(state: SupervisorState):
    """Determines if this is a court-specific query"""
    logger.info("Node: analyze_query")
    
    query = state['input'].lower()
    
    # Check if it's a court API query
    court_keywords = [
        "cnr", "case", "court", "supreme", "district", "consumer", 
        "filing", "advocate", "party", "judge", "diary", "aor",
        "cause list", "hearing", "order", "bench"
    ]
    
    is_court_query = any(keyword in query for keyword in court_keywords)
    
    if not is_court_query:
        return {
            "court_type": "not_applicable",
            "summary": "This doesn't appear to be a court-related query. Please specify if you need information about District Court, Supreme Court, or Consumer Forum cases.",
            "messages": [AIMessage(content="Not a court query")]
        }
    
    # Determine court type
    court_type = determine_court_type(state)
    
    return {
        "court_type": court_type,
        "needs_clarification": court_type == "needs_clarification"
    }


# --- Conditional Edge: Should Clarify? ---
def should_clarify(state: SupervisorState) -> Literal["clarify", "route", "end"]:
    """Decides if we need to ask user for clarification"""
    
    if state.get('court_type') == 'not_applicable':
        return "end"
    
    if state.get('needs_clarification', False):
        return "clarify"
    
    return "route"


# --- Build Supervisor Graph ---
def create_supervisor_graph():
    """Creates the supervisor graph with human-in-the-loop"""
    
    workflow = StateGraph(SupervisorState)
    
    # Add nodes
    workflow.add_node("analyze_query", analyze_query)
    workflow.add_node("ask_court_clarification", ask_court_clarification)
    workflow.add_node("route_to_court_agent", route_to_court_agent)
    
    # Set entry point
    workflow.set_entry_point("analyze_query")
    
    # Add conditional routing
    workflow.add_conditional_edges(
        "analyze_query",
        should_clarify,
        {
            "clarify": "ask_court_clarification",
            "route": "route_to_court_agent",
            "end": END
        }
    )
    
    # After clarification, route to agent
    workflow.add_edge("ask_court_clarification", "route_to_court_agent")
    
    # After agent response, end
    workflow.add_edge("route_to_court_agent", END)
    
    # Compile with checkpointer for memory
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# Export the supervisor
court_supervisor = create_supervisor_graph()
