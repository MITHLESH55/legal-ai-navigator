"""
Court-specific agents for District Court, Supreme Court, and Consumer Forum.
Each agent handles tools specific to its court type.
"""

import logging
from typing import TypedDict, Annotated, List
from operator import add

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain.agents import create_agent
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from llm_factory import get_llm
from tools import (
    # District Court tools
    fetch_ecourts_case_by_cnr,
    search_district_court_by_party,
    search_district_court_by_advocate,
    search_district_court_by_filing_number,
    search_district_court_by_advocate_number,
    get_district_court_cause_list,
    search_ecourts_by_party_name,
    search_ecourts_by_advocate,
    get_ecourts_states,
    get_ecourts_districts,
    get_ecourts_courts,
    
    # Supreme Court tools
    fetch_supreme_court_case_by_diary,
    search_supreme_court_by_party,
    get_supreme_court_orders_on_date,
    search_supreme_court_by_aor,
    
    # Consumer Forum tools
    fetch_consumer_forum_case,
    get_consumer_forum_benches,
    get_consumer_forum_districts,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- State for Court Agents ---
class CourtAgentState(TypedDict):
    input: str
    case_id: str
    agent_scratchpad: Annotated[List[BaseMessage], add]
    summary: str
    court_type: str  # "district", "supreme", or "consumer"


# --- District Court Agent ---
def create_district_court_agent():
    """Agent specialized in District Court queries"""
    # Use small/fast model for routing decisions (saves API calls)
    llm = get_llm(model_size="small", temperature=0)
    
    system_prompt = """
You are a District Court specialist AI assistant.

Your expertise covers:
- District Court case searches by CNR, party name, advocate, filing number
- Cause lists and hearing schedules
- State, district, and court information
- Civil and criminal cases in district courts

Available tools:
- fetch_ecourts_case_by_cnr: Get case by 16-digit CNR number
- search_district_court_by_party: Search by party name, year, stage
- search_district_court_by_advocate: Search by advocate name
- search_district_court_by_filing_number: Search by filing number and year
- search_district_court_by_advocate_number: Search by advocate registration
- get_district_court_cause_list: Get hearings scheduled for a date
- get_ecourts_states: List all states
- get_ecourts_districts: Get districts in a state
- get_ecourts_courts: Get courts in a district

Be direct and efficient. Use tools when needed, answer directly when you can.
"""
    
    district_court_tools = [
        fetch_ecourts_case_by_cnr,
        search_district_court_by_party,
        search_district_court_by_advocate,
        search_district_court_by_filing_number,
        search_district_court_by_advocate_number,
        get_district_court_cause_list,
        search_ecourts_by_party_name,
        search_ecourts_by_advocate,
        get_ecourts_states,
        get_ecourts_districts,
        get_ecourts_courts,
    ]
    
    return create_agent(llm, district_court_tools, system_prompt=system_prompt)


# --- Supreme Court Agent ---
def create_supreme_court_agent():
    """Agent specialized in Supreme Court queries"""
    # Use small/fast model for routing
    llm = get_llm(model_size="small", temperature=0)
    
    system_prompt = """
You are a Supreme Court specialist AI assistant.

Your expertise covers:
- Supreme Court case lookups by diary number
- Party searches in Supreme Court
- Daily orders from Supreme Court
- AOR (Advocate on Record) searches

Available tools:
- fetch_supreme_court_case_by_diary: Get case by diary number and year
- search_supreme_court_by_party: Search by party name (petitioner/respondent)
- get_supreme_court_orders_on_date: Get all orders on a specific date
- search_supreme_court_by_aor: Search by AOR registration number

Be direct and efficient. Use tools when needed, answer directly when you can.
"""
    
    supreme_court_tools = [
        fetch_supreme_court_case_by_diary,
        search_supreme_court_by_party,
        get_supreme_court_orders_on_date,
        search_supreme_court_by_aor,
    ]
    
    return create_agent(llm, supreme_court_tools, system_prompt=system_prompt)


# --- Consumer Forum Agent ---
def create_consumer_forum_agent():
    """Agent specialized in Consumer Forum queries"""
    # Use small/fast model for routing
    llm = get_llm(model_size="small", temperature=0)
    
    system_prompt = """
You are a Consumer Forum specialist AI assistant.

Your expertise covers:
- Consumer Forum case lookups
- Consumer Forum benches and locations
- Consumer Forum districts

Available tools:
- fetch_consumer_forum_case: Get case details by case number
- get_consumer_forum_benches: List all consumer forum benches
- get_consumer_forum_districts: Get districts for consumer forums

Be direct and efficient. Use tools when needed, answer directly when you can.
"""
    
    consumer_forum_tools = [
        fetch_consumer_forum_case,
        get_consumer_forum_benches,
        get_consumer_forum_districts,
    ]
    
    return create_agent(llm, consumer_forum_tools, system_prompt=system_prompt)


# --- Build Specialized Agent Graphs ---
def build_court_agent_graph(agent_type: str):
    """Build a graph for a specific court agent"""
    
    if agent_type == "district":
        agent = create_district_court_agent()
        tools = [
            fetch_ecourts_case_by_cnr, search_district_court_by_party,
            search_district_court_by_advocate, search_district_court_by_filing_number,
            search_district_court_by_advocate_number, get_district_court_cause_list,
            search_ecourts_by_party_name, search_ecourts_by_advocate,
            get_ecourts_states, get_ecourts_districts, get_ecourts_courts
        ]
    elif agent_type == "supreme":
        agent = create_supreme_court_agent()
        tools = [
            fetch_supreme_court_case_by_diary, search_supreme_court_by_party,
            get_supreme_court_orders_on_date, search_supreme_court_by_aor
        ]
    elif agent_type == "consumer":
        agent = create_consumer_forum_agent()
        tools = [
            fetch_consumer_forum_case, get_consumer_forum_benches,
            get_consumer_forum_districts
        ]
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    def agent_node(state: CourtAgentState):
        logger.info(f"Node: {agent_type}_court_agent")
        return {"agent_scratchpad": [agent.invoke(state)]}
    
    def should_call_tools(state: CourtAgentState) -> str:
        last_message = state['agent_scratchpad'][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            logger.info(f"{agent_type} agent: Calling tools")
            return "call_tools"
        logger.info(f"{agent_type} agent: Generating answer")
        return "generate_answer"
    
    def call_tools(state: CourtAgentState):
        logger.info(f"Node: {agent_type}_call_tools")
        tool_node = ToolNode(tools)
        return tool_node.invoke(state)
    
    def generate_answer(state: CourtAgentState):
        logger.info(f"Node: {agent_type}_generate_answer")
        last_message = state['agent_scratchpad'][-1]
        
        if isinstance(last_message, AIMessage) and not last_message.tool_calls:
            # Direct answer
            return {"summary": last_message.content}
        
        # Generate from tool results
        llm = get_llm(model_size="large", temperature=0.1)
        summary_prompt = f"""
Based on the {agent_type} court information retrieved, provide a clear, direct answer.

User question: {{question}}

Retrieved information: {{context}}

Provide a direct, confident answer. No fluff, no disclaimers.
"""
        # Get tool results from scratchpad
        tool_results = [msg.content for msg in state['agent_scratchpad'] if hasattr(msg, 'content')]
        context = "\n".join(str(r) for r in tool_results)
        
        answer = llm.invoke(summary_prompt.format(question=state['input'], context=context))
        return {"summary": answer.content}
    
    # Build graph
    workflow = StateGraph(CourtAgentState)
    
    workflow.add_node("agent", agent_node)
    workflow.add_node("call_tools", call_tools)
    workflow.add_node("generate_answer", generate_answer)
    
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_call_tools, {
        "call_tools": "call_tools",
        "generate_answer": "generate_answer"
    })
    workflow.add_edge("call_tools", "agent")  # Loop back after tools
    workflow.add_edge("generate_answer", END)
    
    return workflow.compile()


# Export compiled graphs
district_court_agent = build_court_agent_graph("district")
supreme_court_agent = build_court_agent_graph("supreme")
consumer_forum_agent = build_court_agent_graph("consumer")
