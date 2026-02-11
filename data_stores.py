import os
import uuid
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from neo4j import GraphDatabase
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. Qdrant Vector Store Class ---

class VectorStore:
    def __init__(self):
        # Read Qdrant configuration for local instance
        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME")
        
        # Read embedding model name
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
        
        if not self.collection_name:
            raise ValueError("QDRANT_COLLECTION_NAME must be set in .env")

        logger.info(f"Initializing Qdrant client (local) at {self.qdrant_host}:{self.qdrant_port}...")
        self.client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
        
        # Initialize FastEmbed embeddings (local, no API key needed)
        logger.info(f"Initializing FastEmbed embeddings with model: {self.embedding_model_name}...")
        self.embeddings = FastEmbedEmbeddings(model_name=self.embedding_model_name)
        
        # Determine embedding size dynamically
        logger.info("Determining embedding size...")
        self._embedding_size = len(self.embeddings.embed_query("test"))
        logger.info(f"Embedding size: {self._embedding_size}")
        
        self.get_or_create_collection()

    def get_or_create_collection(self):
        try:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=self._embedding_size, distance=models.Distance.COSINE),
            )
            logger.info(f"Qdrant collection '{self.collection_name}' created.")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(f"Qdrant collection '{self.collection_name}' already exists.")
            else:
                logger.error(f"Failed to create Qdrant collection: {e}")

    def index_documents(self, documents: list, case_id: str, file_name: str) -> bool:
        """Indexes LangChain Document objects into Qdrant using local FastEmbed embeddings."""
        try:
            points = []
            for i, doc in enumerate(documents): 
                try:
                    embedding = self.embeddings.embed_query(doc.page_content)
                    points.append(
                        models.PointStruct(
                            id=str(uuid.uuid4()),
                            vector=embedding,
                            payload={
                                "text": doc.page_content,
                                "source": file_name,
                                "case_id": case_id,
                                "page_number": doc.metadata.get("page", 0),
                                "chunk_index": i,
                            },
                        )
                    )
                except Exception as embed_error:
                    logger.error(f"Error embedding chunk {i} for {file_name}: {embed_error}")
                    continue  # Skip this chunk if embedding fails
            
            if points:
                self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
            logger.info(f"Successfully indexed {len(points)} chunks to Qdrant for {file_name}.")
            return True
        except Exception as e:
            logger.error(f"Error indexing to Qdrant: {e}")
            return False

    def search_documents(self, query: str, case_id: str = None, limit: int = 5) -> str:
        """Searches Qdrant and returns a string context."""
        try:
            query_embedding = self.embeddings.embed_query(query)
            filter_conditions = []
            if case_id:
                filter_conditions.append(models.FieldCondition(key="case_id", match=models.MatchValue(value=case_id)))
            
            search_filter = models.Filter(must=filter_conditions) if filter_conditions else None
            
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=limit,
                with_payload=True,
            )
            
            context = ""
            for result in search_result:
                context += f"Source: {result.payload['source']}, Page: {result.payload['page_number']}\n"
                context += f"Content: {result.payload['text']}\n---\n"
            
            return context or "No relevant documents found in the uploaded files."
        except Exception as e:
            logger.error(f"Error searching Qdrant: {e}")
            return "Error while searching Qdrant."

# --- 2. Neo4j Knowledge Graph Class ---

class Neo4jGraph:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USERNAME")
        self.password = os.getenv("NEO4J_PASSWORD")
        
        if not self.uri or not self.user or not self.password:
            raise ValueError("NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD must be set in .env")
        
        logger.info("Initializing Neo4j driver...")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self.create_constraints()

    def close(self):
        self.driver.close()

    def create_constraints(self):
        """Creates unique constraints for the enhanced schema to prevent duplicate nodes."""
        with self.driver.session() as session:
            try:
                # Base constraint for all entities
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE")
                
                # Specific constraints for each entity type
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Case) REQUIRE n.id IS UNIQUE")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Judge) REQUIRE n.id IS UNIQUE")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Advocate) REQUIRE n.id IS UNIQUE")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Plaintiff) REQUIRE n.id IS UNIQUE")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Defendant) REQUIRE n.id IS UNIQUE")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Statute) REQUIRE n.id IS UNIQUE")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Precedent) REQUIRE n.id IS UNIQUE")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Court) REQUIRE n.id IS UNIQUE")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Outcome) REQUIRE n.id IS UNIQUE")
                
                # Document constraint
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
                
                logger.info("Neo4j enhanced schema constraints ensured.")
            except Exception as e:
                logger.error(f"Error setting Neo4j constraints: {e}")

    def run_query(self, query: str, params: dict = {}) -> list:
        """Runs a Cypher query and returns the results."""
        with self.driver.session() as session:
            try:
                result = session.run(query, params)
                return [record.data() for record in result]
            except Exception as e:
                logger.error(f"Neo4j query failed: {e}")
                return []

# --- 3. Document Loading Logic ---

def load_and_split_pdf(file_path: str) -> list:
    """Loads a PDF and splits it into LangChain Document objects."""
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split PDF into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        logger.error(f"Error loading/splitting PDF {file_path}: {e}")
        return []

# --- 4. Singleton Instances ---

try:
    vector_store = VectorStore()
    graph_store = Neo4jGraph()
except ValueError as e:
    logger.error(f"Failed to initialize data stores: {e}. Some features will be disabled.")
    vector_store = None
    graph_store = None