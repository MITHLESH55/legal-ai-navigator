import logging
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
import os
import json
from data_stores import graph_store
from llm_factory import get_llm  # <-- Use LLM factory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialize spaCy NER Model (for mid-size documents 11-20 pages) ---
import spacy

NER_MODEL_NAME = "en_legal_ner_trf"  # The name after installation
nlp_ner = None
try:
    logger.info(f"Loading spaCy NER model '{NER_MODEL_NAME}'...")
    # Option 1: Try loading directly by name (if installed correctly via pip)
    nlp_ner = spacy.load(NER_MODEL_NAME)
    logger.info(f"spaCy NER model '{NER_MODEL_NAME}' loaded successfully using spacy.load().")
except OSError:
    logger.warning(f"Could not load spaCy model '{NER_MODEL_NAME}' directly.")
    # Option 2: Try importing and loading as a module (fallback)
    try:
        import en_legal_ner_trf
        nlp_ner = en_legal_ner_trf.load()
        logger.info(f"spaCy NER model '{NER_MODEL_NAME}' loaded successfully using module import.")
    except ImportError:
        logger.error(f"Failed to import '{NER_MODEL_NAME}'. Make sure it's installed via pip from the .whl file.")
        nlp_ner = None
    except Exception as e_mod:
        logger.error(f"Failed to load NER model via module import: {e_mod}. NER graph building will be disabled.", exc_info=True)
        nlp_ner = None
except Exception as e_load:
    logger.error(f"Failed to load NER model '{NER_MODEL_NAME}': {e_load}. NER graph building will be disabled.", exc_info=True)
    nlp_ner = None
# --- End spaCy NER Model Initialization ---

# --- 1. Define the Enhanced Graph Structure (Pydantic models) ---

class Node(BaseModel):
    """A node in the knowledge graph with enhanced schema."""
    id: str = Field(description="A unique identifier for the node (e.g., 'Advocate_John_Doe', 'IPC_302', 'Case_123_2023').")
    type: str = Field(description="The specific type/label of the entity (e.g., 'Judge', 'Advocate', 'Plaintiff', 'Defendant', 'Statute', 'Precedent', 'Case', 'Court').")
    properties: Optional[Dict[str, Any]] = Field(default={}, description="Additional properties specific to the node type (e.g., case_number, section_number, date).")
    
    @field_validator('properties', mode='before')
    @classmethod
    def parse_properties(cls, v):
        """Convert JSON string to dictionary if needed"""
        if isinstance(v, str):
            try:
                # Try to parse as JSON
                parsed = json.loads(v)
                if isinstance(parsed, dict):
                    return parsed
                logger.warning(f"Parsed JSON is not a dict: {type(parsed)}, returning empty dict")
                return {}
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse properties JSON string: {e}, returning empty dict")
                return {}
        return v or {}

class Relationship(BaseModel):
    """A relationship between two nodes in the knowledge graph with enhanced schema."""
    source: str = Field(description="The id of the source node.")
    target: str = Field(description="The id of the target node.")
    type: str = Field(description="The specific relationship type (e.g., 'REPRESENTS', 'SUED', 'CITED_IN', 'RULED_IN_FAVOR_OF', 'ALLEGED_VIOLATION_OF', 'FILED_IN', 'PART_OF').")
    properties: Optional[Dict[str, Any]] = Field(default={}, description="Additional properties for the relationship (e.g., date, description).")
    
    @field_validator('properties', mode='before')
    @classmethod
    def parse_properties(cls, v):
        """Convert JSON string to dictionary if needed"""
        if isinstance(v, str):
            try:
                # Try to parse as JSON
                parsed = json.loads(v)
                if isinstance(parsed, dict):
                    return parsed
                logger.warning(f"Parsed JSON is not a dict: {type(parsed)}, returning empty dict")
                return {}
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse properties JSON string: {e}, returning empty dict")
                return {}
        return v or {}

class Graph(BaseModel):
    """The full knowledge graph with nodes and relationships."""
    nodes: List[Node]
    relationships: List[Relationship]

# --- 2. LLM Chain for Extraction with Enhanced Schema ---

def get_graph_extraction_chain():
    """Creates an LLM chain that extracts a Graph object from text with enhanced schema."""
    try:
        llm = get_llm(model_size="large", temperature=0)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are an expert legal analyst. Extract legal entities and relationships from the given text 
to build a structured knowledge graph using SPECIFIC labels and relationship types.

**CRITICAL: Use Specific Node Types (Labels):**
- **Case**: The central case being discussed (e.g., 'Case_CRL_123_2023')
- **Judge**: Judges involved (e.g., 'Judge_Ramesh_Kumar')
- **Advocate**: Lawyers/Advocates (e.g., 'Advocate_Priya_Sharma')
- **Plaintiff**: The party filing the case (e.g., 'Plaintiff_John_Doe')
- **Defendant**: The party being sued/accused (e.g., 'Defendant_ABC_Corp')
- **Statute**: Legal statutes/sections (e.g., 'IPC_302', 'CrPC_Section_125')
- **Precedent**: Legal precedents/cases cited (e.g., 'Precedent_Kesavananda_Bharati')
- **Court**: Court where case is filed (e.g., 'Court_Delhi_High_Court')
- **Outcome**: Case outcome (e.g., 'Outcome_Acquittal', 'Outcome_Conviction')

**Node Properties (add when available):**
- Case: case_number, year, case_type
- Statute: section_number, act_name
- Precedent: citation, year
- Court: location, level (e.g., 'District', 'High Court', 'Supreme Court')
- Judge/Advocate/Party: name, role

**CRITICAL: Use Specific Relationship Types:**
- **REPRESENTS**: Advocate represents a party (Advocate)-[REPRESENTS]->(Plaintiff/Defendant)
- **SUED**: One party sues another (Plaintiff)-[SUED]->(Defendant)
- **CITED_IN**: Precedent cited in case (Precedent)-[CITED_IN]->(Case)
- **RULED_IN_FAVOR_OF**: Judge ruled in favor (Judge)-[RULED_IN_FAVOR_OF]->(Party)
- **ALLEGED_VIOLATION_OF**: Party alleged violation of statute (Party)-[ALLEGED_VIOLATION_OF]->(Statute)
- **FILED_IN**: Case filed in court (Case)-[FILED_IN]->(Court)
- **PRESIDED_BY**: Judge presides over case (Case)-[PRESIDED_BY]->(Judge)
- **APPLIES_TO**: Statute applies to case (Statute)-[APPLIES_TO]->(Case)
- **RESULTED_IN**: Case resulted in outcome (Case)-[RESULTED_IN]->(Outcome)
- **PART_OF**: Entity is part of case (Entity)-[PART_OF]->(Case)

**Relationship Properties (add when available):**
- FILED_IN: date, filing_number
- RULED_IN_FAVOR_OF: date, order_details
- CITED_IN: context, relevance

**IMPORTANT:**
1. **Always extract a central Case node** if the text discusses a specific case
2. Connect all relevant entities (judges, advocates, parties, statutes) to the Case node
3. Use specific relationship types, NOT generic "RELATIONSHIP"
4. Extract only important entities - ignore generic terms
5. Use clear, unique IDs (e.g., 'Case_CRL_123_2023', not just 'Case1')

**CRITICAL FORMAT REQUIREMENTS:**
- The 'properties' field MUST be a valid JSON object (dictionary), NOT a JSON string
- If a node has no properties, use an empty object: {{}}, NOT null or a string
- DO NOT wrap the JSON object in quotes - it should be a native dictionary
- Examples of CORRECT format:
  * {{"case_number": "123/2023", "case_type": "CRL Appeal"}}
  * {{"name": "John Doe", "role": "Defendant"}}
  * {{}} (empty object for no properties)
- Examples of WRONG format:
  * '{{"case_number": "123/2023"}}' (wrapped in quotes - this is a string!)
  * "case_number: 123/2023, case_type: CRL Appeal" (not JSON)
  * null (not a dictionary)
  * "name: John Doe" (not JSON)

**Examples:**
Text: "In CRL Appeal 123/2023, Judge Ramesh Kumar heard arguments from Advocate Priya Sharma representing the accused Mr. John Doe, who was charged under IPC Section 302. The court cited the precedent of State v. Kumar (2020) and acquitted the accused."

Extract:
Nodes:
- id: "Case_CRL_123_2023", type: "Case", properties: {{"case_number": "123/2023", "case_type": "CRL Appeal"}}
- id: "Judge_Ramesh_Kumar", type: "Judge", properties: {{"name": "Ramesh Kumar"}}
- id: "Advocate_Priya_Sharma", type: "Advocate", properties: {{"name": "Priya Sharma"}}
- id: "Defendant_John_Doe", type: "Defendant", properties: {{"name": "John Doe"}}
- id: "Statute_IPC_302", type: "Statute", properties: {{"section_number": "302", "act_name": "IPC"}}
- id: "Precedent_State_v_Kumar_2020", type: "Precedent", properties: {{"citation": "State v. Kumar (2020)"}}
- id: "Outcome_Acquittal", type: "Outcome", properties: {{"result": "Acquitted"}}

Relationships:
- source: "Case_CRL_123_2023", target: "Judge_Ramesh_Kumar", type: "PRESIDED_BY", properties: {{}}
- source: "Advocate_Priya_Sharma", target: "Defendant_John_Doe", type: "REPRESENTS", properties: {{}}
- source: "Defendant_John_Doe", target: "Statute_IPC_302", type: "ALLEGED_VIOLATION_OF", properties: {{}}
- source: "Precedent_State_v_Kumar_2020", target: "Case_CRL_123_2023", type: "CITED_IN", properties: {{}}
- source: "Case_CRL_123_2023", target: "Outcome_Acquittal", type: "RESULTED_IN", properties: {{}}
- source: "Judge_Ramesh_Kumar", target: "Case_CRL_123_2023", type: "PART_OF", properties: {{}}
- source: "Advocate_Priya_Sharma", target: "Case_CRL_123_2023", type: "PART_OF", properties: {{}}
            """),
            ("human", "Text: {input}")
        ])
        
        # Use .with_structured_output to force the LLM to return our Graph model
        return prompt | llm.with_structured_output(Graph)
    except Exception as e:
        logger.error(f"Error creating graph extraction chain: {e}")
        return None

# --- 2B. Helper Function to Convert NER Output to Graph ---

def _convert_ner_to_graph(spacy_doc) -> Graph:
    """
    Converts spaCy NER doc.ents output to a Graph object with nodes only.
    
    Args:
        spacy_doc: The processed spaCy Doc object containing entities.
    
    Returns:
        Graph object with nodes (relationships will be empty for NER-based extraction)
    """
    nodes = []
    seen_entities = set()
    
    # Entity type mapping (adjust based on actual labels from en_legal_ner_trf)
    entity_type_mapping = {
        'COURT': 'Court',
        'PETITIONER': 'Plaintiff',
        'RESPONDENT': 'Defendant',
        'JUDGE': 'Judge',
        'LAWYER': 'Advocate',
        'WITNESS': 'Witness',
        'STATUTE': 'Statute',
        'PROVISION': 'Statute',  # Map provision also to Statute
        'CASE_NUMBER': 'Case',  # Maybe map this to Case? Check consistency
        'PRECEDENT': 'Precedent',
        'DATE': 'Date', 
        'ORG': 'Organization',
        'GPE': 'Location',  # Geo-Political Entity
        'PERSON': 'Person',  # Generic Person if not lawyer/judge etc.
        # Add other relevant mappings based on model's labels
    }
    
    for ent in spacy_doc.ents:
        ner_label = ent.label_
        word = ent.text.strip()
        
        # Skip empty words
        if not word:
            continue
        
        # Map NER label to our schema type
        node_type = entity_type_mapping.get(ner_label, 'Entity')  # Default to generic Entity
        
        # Create a normalized ID
        normalized_word = word.replace(' ', '_').replace('.', '').replace(',', '').replace('(', '').replace(')', '')
        node_id = f"{node_type}_{normalized_word}"
        
        # Avoid duplicates based on ID
        if node_id in seen_entities:
            continue
        seen_entities.add(node_id)
        
        # Create node with properties
        node = Node(
            id=node_id,
            type=node_type,
            properties={
                'name': word,
                'ner_label': ner_label,  # Store original NER label
                'extraction_method': 'spacy_ner'
            }
        )
        nodes.append(node)
    
    logger.info(f"Converted {len(nodes)} spaCy NER entities to graph nodes")
    return Graph(nodes=nodes, relationships=[])  # No relationships from NER

# --- 3. Function to store graph in Neo4j with Enhanced Schema ---

def store_graph_in_neo4j(graph: Graph, case_id: str, file_name: str, chunk_id: str):
    """Takes a Graph object and writes it to Neo4j using enhanced schema with specific labels and relationships."""
    if not graph_store:
        logger.error("Graph store not initialized.")
        return

    # Create a "Document" node to link all entities to their source
    # Use SET instead of ON CREATE SET to ensure properties are always updated
    doc_id = f"{case_id}_{file_name}_{chunk_id}"
    doc_node_query = """
    MERGE (d:Document {id: $doc_id})
    SET d.case_id = $case_id, 
        d.file_name = $file_name, 
        d.chunk_id = $chunk_id,
        d.updated_at = datetime()
    RETURN d
    """
    result = graph_store.run_query(doc_node_query, {
        "doc_id": doc_id,
        "case_id": case_id,
        "file_name": file_name,
        "chunk_id": chunk_id
    })
    
    if result:
        logger.info(f"Created/Updated Document node: {doc_id} (case_id: {case_id}, file_name: {file_name})")
    else:
        logger.error(f"Failed to create Document node: {doc_id}")

    # Create nodes with specific labels using APOC or dynamic labels
    # For each node, create with both the specific type label and a generic Entity label
    for node in graph.nodes:
        try:
            # Use CALL apoc.create.node for dynamic labels if APOC is available
            # Otherwise, use conditional MERGE based on type
            node_query = f"""
            MERGE (n:{node.type}:Entity {{id: $id}})
            SET n += $properties
            WITH n
            MATCH (d:Document {{id: $doc_id}})
            MERGE (n)-[:APPEARS_IN]->(d)
            """
            
            params = {
                "id": node.id,
                "properties": node.properties or {},
                "doc_id": f"{case_id}_{file_name}_{chunk_id}"
            }
            
            graph_store.run_query(node_query, params)
            
        except Exception as e:
            logger.error(f"Error creating node {node.id}: {e}")
    
    # Create relationships with specific types (skip if empty, e.g., from NER extraction)
    if graph.relationships:
        for rel in graph.relationships:
            try:
                # Use APOC for dynamic relationship types if available
                # Otherwise, construct query dynamically (be careful with Cypher injection)
                rel_query = f"""
                MATCH (source:Entity {{id: $source_id}})
                MATCH (target:Entity {{id: $target_id}})
                MERGE (source)-[r:{rel.type}]->(target)
                SET r += $properties
                """
                
                params = {
                    "source_id": rel.source,
                    "target_id": rel.target,
                    "properties": rel.properties or {}
                }
                
                graph_store.run_query(rel_query, params)
                
            except Exception as e:
                logger.error(f"Error creating relationship {rel.type} from {rel.source} to {rel.target}: {e}")
    else:
        logger.info("No relationships to create (likely NER-based extraction)")
    
    # Link Document to Case node if a Case node was created
    try:
        case_link_query = """
        MATCH (d:Document {id: $doc_id})
        MATCH (c:Case)
        WHERE (c)-[:APPEARS_IN]->(d)
        MERGE (d)-[:PART_OF]->(c)
        """
        graph_store.run_query(case_link_query, {
            "doc_id": f"{case_id}_{file_name}_{chunk_id}"
        })
    except Exception as e:
        logger.debug(f"No Case node found or error linking Document to Case: {e}")

# --- 4. Main public function ---

def extract_and_store_graph(documents: list, case_id: str, file_name: str):
    """
    Main function to extract graph data from documents and store in Neo4j.
    Uses three-tier approach based on document size:
    - ≤10 pages: LLM-based extraction with structured output
    - 11-20 pages: NER-based extraction using local model
    - >20 pages: Rejected (no graph extraction)
    """
    # Read page limit from environment
    max_pages = int(os.getenv("MAX_DOCUMENT_PAGES", "20"))
    
    # Determine the number of pages in the document
    num_pages = 0
    if documents:
        # Find maximum page number from metadata
        for doc in documents:
            page_num = doc.metadata.get("page", 0)
            if page_num > num_pages:
                num_pages = page_num
    
    logger.info(f"Document '{file_name}' has {num_pages} pages. Max allowed: {max_pages}")
    
    # Three-tier logic based on document size
    if num_pages > max_pages:
        # Tier 3: Reject documents over 20 pages
        logger.warning(f"Document '{file_name}' exceeds {max_pages} page limit. Skipping graph extraction.")
        return
    
    elif num_pages > 10:
        # Tier 2: Use NER for mid-size documents (11-20 pages)
        logger.info(f"Document '{file_name}' has {num_pages} pages (11-20 range). Using NER extraction...")
        
        if nlp_ner is None:
            logger.error("spaCy NER model not available. Cannot process mid-size document. Skipping graph extraction.")
            return
        
        try:
            # Concatenate all document chunks into a single text
            full_text = "\n".join([doc.page_content for doc in documents])
            
            # Truncate if text is too long (spaCy can handle larger texts than transformers)
            # But still set a reasonable limit
            max_chars = 1000000  # 1M chars - spaCy can handle this, adjust if needed
            if len(full_text) > max_chars:
                logger.warning(f"Document text is {len(full_text)} chars, truncating to {max_chars} for NER processing")
                full_text = full_text[:max_chars]
            
            # Run spaCy NER pipeline
            logger.info("Running spaCy NER model on document text...")
            spacy_doc = nlp_ner(full_text)
            logger.info(f"spaCy NER extraction complete. Found {len(spacy_doc.ents)} entities.")
            
            # Convert spaCy output to Graph
            graph = _convert_ner_to_graph(spacy_doc)
            
            if graph.nodes:
                # Store the extracted graph in Neo4j
                store_graph_in_neo4j(graph, case_id, file_name, "full_document_ner")
                logger.info(f"Stored NER-based graph (found {len(graph.nodes)} nodes).")
            else:
                logger.info(f"No entities extracted from document '{file_name}' using NER.")
                
        except Exception as e:
            logger.error(f"Failed to process graph with spaCy NER for document '{file_name}': {e}", exc_info=True)
    
    else:
        # Tier 1: Use LLM for small documents (≤10 pages)
        logger.info(f"Document '{file_name}' has {num_pages} pages (≤10 range). Using LLM extraction...")
        
        chain = get_graph_extraction_chain()
        if not chain:
            logger.error("Could not create LLM extraction chain. Aborting graph build.")
            return
        
        try:
            # Concatenate all document chunks into a single text
            batched_text = "\n---\n".join([doc.page_content for doc in documents])
            
            # Single LLM call for entire document
            logger.info("Invoking LLM for graph extraction on full document...")
            
            try:
                graph = chain.invoke({"input": batched_text})
                logger.info(f"LLM extraction successful: {len(graph.nodes)} nodes, {len(graph.relationships)} relationships")
            except Exception as llm_error:
                logger.error(f"LLM invocation or validation failed: {llm_error}", exc_info=True)
                logger.error(f"Error type: {type(llm_error).__name__}")
                # Log first few validation errors for debugging
                if hasattr(llm_error, 'errors'):
                    error_list = llm_error.errors()
                    logger.error(f"First validation error sample: {error_list[0] if error_list else 'N/A'}")
                # Continue execution - don't crash the app
                logger.warning(f"Skipping graph extraction for {file_name} due to validation errors")
                return
            
            if graph.nodes:
                # Store the extracted graph in Neo4j
                store_graph_in_neo4j(graph, case_id, file_name, "full_document_llm")
                logger.info(f"Stored LLM-based graph (found {len(graph.nodes)} nodes, {len(graph.relationships)} relationships).")
            else:
                logger.info(f"No entities extracted from document '{file_name}' using LLM.")
                
        except Exception as e:
            logger.error(f"Failed to process graph with LLM for document '{file_name}': {e}")
    
    logger.info("Graph extraction complete.")