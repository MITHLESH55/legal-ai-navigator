import os
import json
import requests
import logging
from langchain_community.utilities import SerpAPIWrapper
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from data_stores import vector_store, graph_store # Import our new data_stores
from llm_factory import get_llm  # <-- Use LLM factory
from ecourts_cache import (  # <-- Import cache helpers
    get_states_cached,
    get_districts_cached,
    get_consumer_districts_cached,
    get_consumer_benches_cached,
    get_cache_info
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Tool 1: Vector RAG (Qdrant) ---
@tool
def search_uploaded_documents(query: str, case_id: str = None) -> str:
    """
    Searches the user's **private, uploaded documents** for information
    using semantic vector search.
    Use this tool for "fuzzy" or "similarity" based queries about
    the user's specific case files.
    """
    logger.info(f"Tool: search_uploaded_documents, query: {query}, case_id: {case_id}")
    if not vector_store:
        return "Vector store is not initialized. Please check Qdrant configuration."
    return vector_store.search_documents(query, case_id)

# --- Tool 2: Public Web Search (SerpAPI) ---
@tool
def search_public_cases_and_news_serpapi(query: str) -> str:
    """
    Searches the **public internet** for general legal information, news articles,
    and discussions on similar cases or legal precedents. Returns the search results 
    directly to the agent for processing.
    
    Use this tool when the user asks for "similar cases," "legal news," or general 
    information not specific to their own files, AND they want you to process/summarize the results.
    
    DO NOT use this tool if the user explicitly asks to "open a browser," "show me in a new tab," 
    or "search the web for me" - use open_browser_search instead for those requests.
    """
    logger.info(f"Tool: search_public_cases_and_news_serpapi, query: {query}")
    try:
        serp_api_key = os.getenv("SERPAPI_API_KEY")
        if not serp_api_key:
            return "SerpAPI is not configured. Missing SERPAPI_API_KEY."
        
        search = SerpAPIWrapper(serpapi_api_key=serp_api_key)
        result = search.run(f"Indian law {query}")
        return f"Public web search results: {result}"
    except Exception as e:
        return f"Error during web search: {e}"

# --- Tool 3: e-Courts API - Case Lookup by CNR ---
@tool
def fetch_ecourts_case_by_cnr(cnr_number: str) -> str:
    """
    Fetches specific case details from District Court using CNR (Case Number Reference).
    Use this tool when the user provides a **CNR Number**.
    The CNR is a unique identifier for each case in the Indian court system.
    """
    logger.info(f"Tool: fetch_ecourts_case_by_cnr, cnr: {cnr_number}")
    
    api_token = os.getenv("ECOURTS_API_TOKEN")
    if not api_token:
        return "e-Courts API is not configured. Missing ECOURTS_API_TOKEN in .env file."
    
    try:
        url = "https://court-api.kleopatra.io/api/core/live/district-court/case"
        headers = {
            "authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
        payload = {"cnr": cnr_number}
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            return json.dumps({
                "status": "Success",
                "cnr_number": cnr_number,
                "case_details": data
            }, indent=2)
        elif response.status_code == 404:
            return f"Case not found with CNR: {cnr_number}"
        else:
            logger.error(f"e-Courts API error: {response.status_code} - {response.text}")
            return f"Error fetching case details: HTTP {response.status_code}"
            
    except Exception as e:
        logger.error(f"e-Courts API request failed: {e}")
        return f"Failed to connect to e-Courts API: {str(e)}"

# --- Tool 4: e-Courts API - Search by Party Name ---
@tool
def search_ecourts_by_party_name(party_name: str, state_code: str, district_code: str) -> str:
    """
    Searches for cases by party name (plaintiff/defendant/petitioner/respondent) 
    in District Courts.
    Requires: party_name, state_code, and district_code.
    Use this when user wants to find cases involving a specific person or organization.
    """
    logger.info(f"Tool: search_ecourts_by_party_name, party: {party_name}, state: {state_code}, district: {district_code}")
    
    api_token = os.getenv("ECOURTS_API_TOKEN")
    if not api_token:
        return "e-Courts API is not configured. Missing ECOURTS_API_TOKEN in .env file."
    
    try:
        url = "https://court-api.kleopatra.io/api/core/live/district-court/search/party"
        headers = {
            "authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "party_name": party_name,
            "state_code": state_code,
            "district_code": district_code
        }
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            return json.dumps({
                "status": "Success",
                "party_name": party_name,
                "results": data
            }, indent=2)
        else:
            logger.error(f"e-Courts API error: {response.status_code}")
            return f"Error searching by party name: HTTP {response.status_code}"
            
    except Exception as e:
        logger.error(f"e-Courts party search failed: {e}")
        return f"Failed to search by party name: {str(e)}"

# --- Tool 5: e-Courts API - Search by Advocate Name ---
@tool
def search_ecourts_by_advocate(advocate_name: str, state_code: str, district_code: str) -> str:
    """
    Searches for cases by advocate/lawyer name in District Courts.
    Requires: advocate_name, state_code, and district_code.
    Use this when user wants to find cases handled by a specific advocate.
    """
    logger.info(f"Tool: search_ecourts_by_advocate, advocate: {advocate_name}, state: {state_code}, district: {district_code}")
    
    api_token = os.getenv("ECOURTS_API_TOKEN")
    if not api_token:
        return "e-Courts API is not configured. Missing ECOURTS_API_TOKEN in .env file."
    
    try:
        url = "https://court-api.kleopatra.io/api/core/live/district-court/search/advocate"
        headers = {
            "authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "advocate_name": advocate_name,
            "state_code": state_code,
            "district_code": district_code
        }
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            return json.dumps({
                "status": "Success",
                "advocate_name": advocate_name,
                "results": data
            }, indent=2)
        else:
            logger.error(f"e-Courts API error: {response.status_code}")
            return f"Error searching by advocate name: HTTP {response.status_code}"
            
    except Exception as e:
        logger.error(f"e-Courts advocate search failed: {e}")
        return f"Failed to search by advocate name: {str(e)}"

# --- Helper Tool 6: Get e-Courts States ---
@tool
def get_ecourts_states() -> str:
    """
    Fetches the list of all states and UTs with district courts in India.
    Use this when you need to know available states or their codes.
    Uses 7-day cache to minimize API calls.
    """
    logger.info("Tool: get_ecourts_states")
    
    # Try to load from cache first
    cached_states = get_states_cached()
    if cached_states:
        logger.info(f"✅ Loaded {len(cached_states)} states from cache")
        return json.dumps({
            "status": "Success (cached)",
            "count": len(cached_states),
            "states": cached_states
        }, indent=2)
    
    # Fallback to API if cache miss
    logger.warning("Cache miss - fetching from API")
    api_token = os.getenv("ECOURTS_API_TOKEN")
    if not api_token:
        return "e-Courts API is not configured. Missing ECOURTS_API_TOKEN in .env file."
    
    try:
        url = "https://court-api.kleopatra.io/api/core/static/district-court/states"
        headers = {"authorization": f"Bearer {api_token}"}
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            states = response.json()
            logger.info("  Fetched from API. Consider running: python cache_ecourts_data.py")
            return json.dumps({
                "status": "Success (API)",
                "count": len(states) if isinstance(states, list) else 0,
                "states": states,
                "note": "Cache not available. Run: python cache_ecourts_data.py"
            }, indent=2)
        else:
            logger.error(f"e-Courts States API error: {response.status_code}")
            return f"Error fetching states: HTTP {response.status_code}"
            
    except Exception as e:
        logger.error(f"Error fetching e-Courts states: {e}")
        return f"Error: {str(e)}"

# --- Helper Tool 7: Get Districts for a State ---
@tool
def get_ecourts_districts(state_codes: list) -> str:
    """
    Fetches districts for specified state(s).
    Provide state_codes as a list (e.g., ["DL", "MH"]).
    Use this when user needs to know districts in a state.
    """
    logger.info(f"Tool: get_ecourts_districts, states: {state_codes}")
    
    api_token = os.getenv("ECOURTS_API_TOKEN")
    if not api_token:
        return "e-Courts API is not configured. Missing ECOURTS_API_TOKEN in .env file."
    
    try:
        url = "https://court-api.kleopatra.io/api/core/static/district-court/districts"
        headers = {
            "authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
        payload = {"state_codes": state_codes}
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            districts = response.json()
            return json.dumps({
                "status": "Success",
                "state_codes": state_codes,
                "districts": districts
            }, indent=2)
        else:
            logger.error(f"e-Courts Districts API error: {response.status_code}")
            return f"Error fetching districts: HTTP {response.status_code}"
            
    except Exception as e:
        logger.error(f"Error fetching districts: {e}")
        return f"Error: {str(e)}"

# --- Helper Tool 8: Get Courts in a Complex ---
@tool
def get_ecourts_courts(complex_codes: list) -> str:
    """
    Fetches courts for specified complex(es).
    Provide complex_codes as a list.
    Use this when user needs specific court information.
    """
    logger.info(f"Tool: get_ecourts_courts, complexes: {complex_codes}")
    
    api_token = os.getenv("ECOURTS_API_TOKEN")
    if not api_token:
        return "e-Courts API is not configured. Missing ECOURTS_API_TOKEN in .env file."
    
    try:
        url = "https://court-api.kleopatra.io/api/core/static/district-court/courts"
        headers = {
            "authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
        payload = {"complex_codes": complex_codes}
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            courts = response.json()
            return json.dumps({
                "status": "Success",
                "complex_codes": complex_codes,
                "courts": courts
            }, indent=2)
        else:
            logger.error(f"e-Courts Courts API error: {response.status_code}")
            return f"Error fetching courts: HTTP {response.status_code}"
            
    except Exception as e:
        logger.error(f"Error fetching courts: {e}")
        return f"Error: {str(e)}"

# --- Tool 4: Knowledge Graph RAG (Neo4j) ---
@tool
def search_knowledge_graph(question: str) -> str:
    """
    Searches the Neo4j knowledge graph for entity relationships.
    Use this when user asks about relationships like:
    - "Who represented the plaintiff?"
    - "Which judges cited this precedent?"
    - "What are all entities in my case?"
    - "What statutes were alleged in the case?"
    
    Args:
        question: The user's question about entity relationships
    """
    if not graph_store:
        return "Knowledge graph is not initialized."
    
    logger.info(f"Tool: search_knowledge_graph, question: {question}")
    
    try:
        # Use large model for better Cypher generation
        llm = get_llm(model_size="large", temperature=0)
        
        # Enhanced prompt with new schema
        cypher_prompt = ChatPromptTemplate.from_template("""
You are a Neo4j Cypher expert. Convert this natural language question into a Cypher query.

**ENHANCED GRAPH SCHEMA:**

**Node Labels (all also have :Entity):**
- Case: Central case node (properties: case_number, year, case_type)
- Judge: Judges (properties: name, role)
- Advocate: Lawyers/Advocates (properties: name, role)
- Plaintiff: Parties filing case (properties: name)
- Defendant: Parties being sued/accused (properties: name)
- Statute: Legal statutes (properties: section_number, act_name)
- Precedent: Legal precedents cited (properties: citation, year)
- Court: Court where filed (properties: location, level)
- Outcome: Case outcomes (properties: result)
- Document: Source documents (properties: case_id, file_name)

**Relationship Types:**
- REPRESENTS: Advocate represents Party
- SUED: Plaintiff sued Defendant
- CITED_IN: Precedent cited in Case
- RULED_IN_FAVOR_OF: Judge ruled in favor of Party
- ALLEGED_VIOLATION_OF: Party alleged violation of Statute
- FILED_IN: Case filed in Court
- PRESIDED_BY: Case presided by Judge
- APPLIES_TO: Statute applies to Case
- RESULTED_IN: Case resulted in Outcome
- PART_OF: Entity part of Case
- APPEARS_IN: Entity appears in Document

**Query Examples:**

Question: "Who represented the plaintiff?"
Query: MATCH (p:Plaintiff)-[:REPRESENTED_BY|REPRESENTS]-(a:Advocate) RETURN a.name AS Advocate, p.name AS Plaintiff LIMIT 10

Question: "What statutes were violated in case CRL_123_2023?"
Query: MATCH (c:Case {{id: 'Case_CRL_123_2023'}})<-[:PART_OF]-(d:Defendant)-[:ALLEGED_VIOLATION_OF]->(s:Statute) RETURN s.section_number, s.act_name LIMIT 10

Question: "Which judge presided over the case?"
Query: MATCH (c:Case)<-[:PRESIDED_BY]-(j:Judge) RETURN j.name AS Judge, c.case_number AS CaseNumber LIMIT 10

Question: "What precedents were cited?"
Query: MATCH (p:Precedent)-[:CITED_IN]->(c:Case) RETURN p.citation, p.year, c.case_number LIMIT 10

Question: "Show all entities in my case"
Query: MATCH (e:Entity)-[:PART_OF]->(c:Case) RETURN DISTINCT labels(e) AS Type, e.id AS EntityID, e LIMIT 20

**IMPORTANT:**
- Use specific labels (Judge, Advocate, Case, etc.) NOT just Entity
- Use specific relationship types (REPRESENTS, SUED, etc.) NOT generic RELATIONSHIP
- Query properties when available (name, case_number, section_number)
- Always LIMIT results to 10-20

User Question: {question}

Generate ONLY the Cypher query, no explanation or markdown.

Cypher Query:
""")
        
        chain = cypher_prompt | llm
        cypher_query = chain.invoke({"question": question}).content.strip()
        
        # Remove markdown code blocks if present
        cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()
        
        logger.info(f"Generated Cypher: {cypher_query}")
        
        # Execute the query
        results = graph_store.run_query(cypher_query)
        
        if not results:
            return "No matching entities found in the knowledge graph."
        
        # Format results
        formatted = "Knowledge Graph Results:\n"
        for i, record in enumerate(results[:10], 1):
            formatted += f"{i}. {record}\n"
        
        return formatted
        
    except Exception as e:
        logger.error(f"Error in KG tool: {e}")
        return f"Error querying knowledge graph: {str(e)}"


# --- NEW KLEOPATRA API TOOLS ---

# District Court Tools
@tool
def search_district_court_by_party(name: str, year: str, stage: str = "BOTH", district_id: str = None, complex_id: str = None) -> str:
    """
    Searches District Court cases by party name.
    
    Args:
        name: Party name to search for
        year: Year of case filing
        stage: Case stage - "PENDING", "DISPOSED", or "BOTH" (default)
        district_id: District ID (optional, provide either district_id or complex_id)
        complex_id: Complex ID (optional, provide either district_id or complex_id)
    
    Use when user asks about cases involving a specific party in district courts.
    """
    logger.info(f"Tool: search_district_court_by_party, name: {name}, year: {year}")
    
    api_token = os.getenv("ECOURTS_API_TOKEN")
    if not api_token:
        return "e-Courts API not configured. Missing ECOURTS_API_TOKEN."
    
    try:
        url = "https://court-api.kleopatra.io/api/core/live/district-court/search/party"
        headers = {
            "authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "name": name,
            "stage": stage,
            "year": year
        }
        if district_id:
            payload["districtId"] = district_id
        if complex_id:
            payload["complexId"] = complex_id
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            return json.dumps(response.json(), indent=2)
        else:
            return f"Error: HTTP {response.status_code} - {response.text}"
    except Exception as e:
        return f"API request failed: {str(e)}"


@tool
def search_district_court_by_advocate(name: str, stage: str = "BOTH", district_id: str = None, complex_id: str = None) -> str:
    """
    Searches District Court cases by advocate name.
    
    Args:
        name: Advocate name to search for
        stage: Case stage - "PENDING", "DISPOSED", or "BOTH" (default)
        district_id: District ID (optional)
        complex_id: Complex ID (optional)
    
    Use when user asks about cases handled by a specific advocate.
    """
    logger.info(f"Tool: search_district_court_by_advocate, name: {name}")
    
    api_token = os.getenv("ECOURTS_API_TOKEN")
    if not api_token:
        return "e-Courts API not configured."
    
    try:
        url = "https://court-api.kleopatra.io/api/core/live/district-court/search/advocate"
        headers = {
            "authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
        payload = {"name": name, "stage": stage}
        if district_id:
            payload["districtId"] = district_id
        if complex_id:
            payload["complexId"] = complex_id
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            return json.dumps(response.json(), indent=2)
        else:
            return f"Error: HTTP {response.status_code}"
    except Exception as e:
        return f"API request failed: {str(e)}"


@tool
def search_district_court_by_filing_number(filing_number: str, filing_year: str, district_id: str = None, complex_id: str = None) -> str:
    """
    Searches District Court cases by filing number.
    
    Args:
        filing_number: Filing number of the case
        filing_year: Year of filing
        district_id: District ID (optional)
        complex_id: Complex ID (optional)
    
    Use when user provides a case filing number.
    """
    logger.info(f"Tool: search_district_court_by_filing_number, number: {filing_number}")
    
    api_token = os.getenv("ECOURTS_API_TOKEN")
    if not api_token:
        return "e-Courts API not configured."
    
    try:
        url = "https://court-api.kleopatra.io/api/core/live/district-court/search/filing"
        headers = {
            "authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "filingNumber": filing_number,
            "filingYear": filing_year
        }
        if district_id:
            payload["districtId"] = district_id
        if complex_id:
            payload["complexId"] = complex_id
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            return json.dumps(response.json(), indent=2)
        else:
            return f"Error: HTTP {response.status_code}"
    except Exception as e:
        return f"API request failed: {str(e)}"


@tool
def search_district_court_by_advocate_number(state: str, number: str, year: str, stage: str = "BOTH", district_id: str = None, complex_id: str = None) -> str:
    """
    Searches District Court cases by advocate registration number.
    
    Args:
        state: State code where advocate is registered
        number: Advocate registration number
        year: Registration year
        stage: Case stage - "PENDING", "DISPOSED", or "BOTH"
        district_id: District ID (optional)
        complex_id: Complex ID (optional)
    
    Use when user provides advocate registration details.
    """
    logger.info(f"Tool: search_district_court_by_advocate_number, state: {state}, number: {number}")
    
    api_token = os.getenv("ECOURTS_API_TOKEN")
    if not api_token:
        return "e-Courts API not configured."
    
    try:
        url = "https://court-api.kleopatra.io/api/core/live/district-court/search/advocate-number"
        headers = {
            "authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "advocate": {
                "state": state,
                "number": number,
                "year": year
            },
            "stage": stage
        }
        if district_id:
            payload["districtId"] = district_id
        if complex_id:
            payload["complexId"] = complex_id
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            return json.dumps(response.json(), indent=2)
        else:
            return f"Error: HTTP {response.status_code}"
    except Exception as e:
        return f"API request failed: {str(e)}"


@tool
def get_district_court_cause_list(date: str, case_type: str, court_id: str) -> str:
    """
    Retrieves cause list for a specific district court on a given date.
    
    Args:
        date: Date in DD-MM-YYYY format (e.g., "20-10-2025")
        case_type: Type of case - "CRIMINAL" or "CIVIL"
        court_id: Court ID
    
    Use when user asks about cases scheduled for hearing on a specific date.
    """
    logger.info(f"Tool: get_district_court_cause_list, date: {date}, type: {case_type}")
    
    api_token = os.getenv("ECOURTS_API_TOKEN")
    if not api_token:
        return "e-Courts API not configured."
    
    try:
        url = "https://court-api.kleopatra.io/api/core/live/district-court/cause-list"
        headers = {
            "authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "date": date,
            "type": case_type,
            "courtId": court_id
        }
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            return json.dumps(response.json(), indent=2)
        else:
            return f"Error: HTTP {response.status_code}"
    except Exception as e:
        return f"API request failed: {str(e)}"


# Supreme Court Tools
@tool
def fetch_supreme_court_case_by_diary(diary_number: str, year: str) -> str:
    """
    Fetches Supreme Court case details using diary number.
    
    Args:
        diary_number: Diary number of the case
        year: Year of the case
    
    Use when user provides Supreme Court diary number.
    """
    logger.info(f"Tool: fetch_supreme_court_case_by_diary, diary: {diary_number}, year: {year}")
    
    api_token = os.getenv("ECOURTS_API_TOKEN")
    if not api_token:
        return "e-Courts API not configured."
    
    try:
        url = "https://court-api.kleopatra.io/api/core/live/supreme-court/case"
        headers = {
            "authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "diaryNumber": diary_number,
            "year": year
        }
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            return json.dumps(response.json(), indent=2)
        else:
            return f"Error: HTTP {response.status_code}"
    except Exception as e:
        return f"API request failed: {str(e)}"


@tool
def search_supreme_court_by_party(name: str, party_type: str, year: str, stage: str) -> str:
    """
    Searches Supreme Court cases by party name.
    
    Args:
        name: Party name to search
        party_type: "ANY", "PETITIONER", or "RESPONDENT"
        year: Year of case
        stage: "PENDING" or "DISPOSED"
    
    Use when user asks about Supreme Court cases involving a specific party.
    """
    logger.info(f"Tool: search_supreme_court_by_party, name: {name}, type: {party_type}")
    
    api_token = os.getenv("ECOURTS_API_TOKEN")
    if not api_token:
        return "e-Courts API not configured."
    
    try:
        url = "https://court-api.kleopatra.io/api/core/live/supreme-court/search/party"
        headers = {
            "authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "name": name,
            "type": party_type,
            "year": year,
            "stage": stage
        }
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            return json.dumps(response.json(), indent=2)
        else:
            return f"Error: HTTP {response.status_code}"
    except Exception as e:
        return f"API request failed: {str(e)}"


@tool
def get_supreme_court_orders_on_date(date: str) -> str:
    """
    Retrieves all orders issued by Supreme Court on a specific date.
    
    Args:
        date: Date in DD-MM-YYYY format (e.g., "20-10-2025")
    
    Use when user asks about Supreme Court orders on a particular date.
    """
    logger.info(f"Tool: get_supreme_court_orders_on_date, date: {date}")
    
    api_token = os.getenv("ECOURTS_API_TOKEN")
    if not api_token:
        return "e-Courts API not configured."
    
    try:
        url = "https://court-api.kleopatra.io/api/core/live/supreme-court/orders-on-date"
        headers = {
            "authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
        payload = {"date": date}
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            return json.dumps(response.json(), indent=2)
        else:
            return f"Error: HTTP {response.status_code}"
    except Exception as e:
        return f"API request failed: {str(e)}"


@tool
def search_supreme_court_by_aor(aor_number: str, year: str, stage: str) -> str:
    """
    Searches Supreme Court cases by Advocate on Record (AOR) number.
    
    Args:
        aor_number: AOR registration number
        year: Year
        stage: "PENDING" or "DISPOSED"
    
    Use when user provides Supreme Court AOR details.
    """
    logger.info(f"Tool: search_supreme_court_by_aor, aor: {aor_number}")
    
    api_token = os.getenv("ECOURTS_API_TOKEN")
    if not api_token:
        return "e-Courts API not configured."
    
    try:
        url = "https://court-api.kleopatra.io/api/core/live/supreme-court/search/aor"
        headers = {
            "authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "number": aor_number,
            "year": year,
            "stage": stage
        }
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            return json.dumps(response.json(), indent=2)
        else:
            return f"Error: HTTP {response.status_code}"
    except Exception as e:
        return f"API request failed: {str(e)}"


# Consumer Forum Tools
@tool
def fetch_consumer_forum_case(case_number: str) -> str:
    """
    Fetches Consumer Forum case details using case number.
    
    Args:
        case_number: Consumer Forum case number
    
    Use when user asks about Consumer Forum cases.
    """
    logger.info(f"Tool: fetch_consumer_forum_case, case: {case_number}")
    
    api_token = os.getenv("ECOURTS_API_TOKEN")
    if not api_token:
        return "e-Courts API not configured."
    
    try:
        url = "https://court-api.kleopatra.io/api/core/live/consumer-forum/case"
        headers = {
            "authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
        payload = {"caseNumber": case_number}
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            return json.dumps(response.json(), indent=2)
        else:
            return f"Error: HTTP {response.status_code}"
    except Exception as e:
        return f"API request failed: {str(e)}"


@tool
def get_consumer_forum_benches() -> str:
    """
    Retrieves list of all Consumer Forum benches in India.
    
    Use when user asks about available Consumer Forum locations.
    """
    logger.info("Tool: get_consumer_forum_benches")
    
    api_token = os.getenv("ECOURTS_API_TOKEN")
    if not api_token:
        return "e-Courts API not configured."
    
    try:
        url = "https://court-api.kleopatra.io/api/core/static/consumer-forum/benches"
        headers = {"authorization": f"Bearer {api_token}"}
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            return json.dumps(response.json(), indent=2)
        else:
            return f"Error: HTTP {response.status_code}"
    except Exception as e:
        return f"API request failed: {str(e)}"


@tool
def get_consumer_forum_districts(bench_id: str = None, bench_ids: list = None, get_all: bool = False) -> str:
    """
    Retrieves Consumer Forum districts for specified benches or all districts.
    Uses 7-day cache to minimize API calls.
    
    Args:
        bench_id: Single bench ID (optional)
        bench_ids: List of bench IDs (optional)
        get_all: Get all districts (default: False)
    
    Use when user needs district information for Consumer Forums.
    """
    logger.info("Tool: get_consumer_forum_districts")
    
    # Try to load from cache first (if getting all districts)
    if get_all and not bench_id and not bench_ids:
        cached_districts = get_consumer_districts_cached()
        if cached_districts:
            logger.info(f"✅ Loaded {len(cached_districts)} consumer districts from cache")
            return json.dumps({
                "status": "Success (cached)",
                "count": len(cached_districts),
                "districts": cached_districts
            }, indent=2)
    
    # Fallback to API
    logger.warning("Cache miss or specific query - fetching from API")
    api_token = os.getenv("ECOURTS_API_TOKEN")
    if not api_token:
        return "e-Courts API not configured."
    
    try:
        url = "https://court-api.kleopatra.io/api/core/static/consumer-forum/districts"
        headers = {
            "authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
        payload = {}
        if bench_id:
            payload["benchId"] = bench_id
        if bench_ids:
            payload["benchIds"] = bench_ids
        if get_all:
            payload["all"] = True
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            logger.info("⚠️  Fetched from API. Consider running: python cache_ecourts_data.py")
            return json.dumps({
                "status": "Success (API)",
                "data": data,
                "note": "Cache not available. Run: python cache_ecourts_data.py"
            }, indent=2)
        else:
            return f"Error: HTTP {response.status_code}"
    except Exception as e:
        return f"API request failed: {str(e)}"


# Import court supervisor tool
from court_tool import search_court_cases

# Import Google Drive tool
from drive_tool import search_google_drive_files

# Import browser redirect tool
from browser_tool import open_browser_search

# List of all tools for the agent (simplified with supervisor)
all_tools = [
    # Core tools
    search_uploaded_documents,
    search_public_cases_and_news_serpapi,
    search_knowledge_graph,
    
    # Court supervisor (wraps all court-specific agents with human-in-the-loop)
    search_court_cases,
    
    # Google Drive integration
    search_google_drive_files,
    
    # Browser redirect for web searches
    open_browser_search,
]

# Legacy court tools kept for backwards compatibility (not exposed to main agent)
legacy_court_tools = [
    fetch_ecourts_case_by_cnr,
    search_ecourts_by_party_name,
    search_ecourts_by_advocate,
    get_ecourts_states,
    get_ecourts_districts,
    get_ecourts_courts,
    search_district_court_by_party,
    search_district_court_by_advocate,
    search_district_court_by_filing_number,
    search_district_court_by_advocate_number,
    get_district_court_cause_list,
    fetch_supreme_court_case_by_diary,
    search_supreme_court_by_party,
    get_supreme_court_orders_on_date,
    search_supreme_court_by_aor,
    fetch_consumer_forum_case,
    get_consumer_forum_benches,
    get_consumer_forum_districts,
]