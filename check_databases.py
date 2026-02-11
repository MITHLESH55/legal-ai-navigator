"""
Diagnostic Script to Check Neo4j and Qdrant Database Status
Usage: python check_databases.py
"""
import os
import logging
from dotenv import load_dotenv
from data_stores import vector_store, graph_store

load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_neo4j():
    """Check Neo4j database for Document nodes and their properties."""
    print("\n" + "="*80)
    print("CHECKING NEO4J DATABASE")
    print("="*80)
    
    if not graph_store:
        print("❌ Graph store not initialized!")
        return
    
    # Check all Document nodes
    query = """
    MATCH (d:Document)
    RETURN d.id AS id, d.case_id AS case_id, d.file_name AS file_name, 
           d.chunk_id AS chunk_id, d.updated_at AS updated_at
    ORDER BY d.updated_at DESC
    """
    
    results = graph_store.run_query(query)
    
    if not results:
        print("⚠️  No Document nodes found in Neo4j")
    else:
        print(f"✅ Found {len(results)} Document node(s):\n")
        for i, record in enumerate(results, 1):
            print(f"Document {i}:")
            print(f"  ID: {record.get('id', 'N/A')}")
            print(f"  Case ID: {record.get('case_id', 'MISSING ❌')}")
            print(f"  File Name: {record.get('file_name', 'MISSING ❌')}")
            print(f"  Chunk ID: {record.get('chunk_id', 'N/A')}")
            print(f"  Updated At: {record.get('updated_at', 'N/A')}")
            
            # Check if properties exist
            if not record.get('case_id'):
                print("  ⚠️  WARNING: case_id is missing or NULL!")
            if not record.get('file_name'):
                print("  ⚠️  WARNING: file_name is missing or NULL!")
            print()
    
    # Check all entity nodes connected to documents
    entity_query = """
    MATCH (d:Document)<-[:APPEARS_IN]-(n:Entity)
    RETURN d.file_name AS document, count(n) AS entity_count
    ORDER BY document
    """
    
    entity_results = graph_store.run_query(entity_query)
    
    if entity_results:
        print(f"\n✅ Found entities linked to documents:")
        for record in entity_results:
            print(f"  {record['document']}: {record['entity_count']} entities")
    else:
        print("\n⚠️  No entities found linked to documents")
    
    # Check for specific entity types
    type_query = """
    MATCH (n:Entity)
    RETURN labels(n) AS labels, count(n) AS count
    ORDER BY count DESC
    """
    
    type_results = graph_store.run_query(type_query)
    
    if type_results:
        print(f"\n✅ Entity types found:")
        for record in type_results:
            labels = [l for l in record['labels'] if l != 'Entity']
            if labels:
                print(f"  {', '.join(labels)}: {record['count']} nodes")
    
    print("\n" + "="*80)


def check_qdrant():
    """Check Qdrant vector store for indexed documents."""
    print("\n" + "="*80)
    print("CHECKING QDRANT VECTOR STORE")
    print("="*80)
    
    if not vector_store:
        print("❌ Vector store not initialized!")
        return
    
    try:
        # Get collection info
        collection_info = vector_store.client.get_collection(
            collection_name=vector_store.collection_name
        )
        
        print(f"✅ Collection: {vector_store.collection_name}")
        print(f"   Vectors: {collection_info.points_count}")
        print(f"   Embedding Model: {vector_store.embedding_model_name}")
        print(f"   Embedding Size: {vector_store._embedding_size}")
        
        if collection_info.points_count == 0:
            print("\n⚠️  No vectors indexed yet!")
        else:
            # Scroll to get some sample points
            scroll_result = vector_store.client.scroll(
                collection_name=vector_store.collection_name,
                limit=10,
                with_payload=True
            )
            
            points = scroll_result[0]
            
            if points:
                print(f"\n✅ Sample of indexed documents:")
                
                # Group by source file
                sources = {}
                for point in points:
                    source = point.payload.get('source', 'Unknown')
                    case_id = point.payload.get('case_id', 'Unknown')
                    if source not in sources:
                        sources[source] = {'case_id': case_id, 'chunks': 0}
                    sources[source]['chunks'] += 1
                
                for source, info in sources.items():
                    print(f"  File: {source}")
                    print(f"    Case ID: {info['case_id']}")
                    print(f"    Chunks: {info['chunks']}")
                
                # Test search
                print("\n✅ Testing search functionality...")
                test_query = "legal"
                search_results = vector_store.search_documents(test_query, limit=3)
                
                if "No relevant documents found" in search_results:
                    print("  ⚠️  Search returned no results")
                else:
                    lines = search_results.split('\n')[:5]
                    print("  Search results (first 5 lines):")
                    for line in lines:
                        if line.strip():
                            print(f"    {line}")
    
    except Exception as e:
        print(f"❌ Error checking Qdrant: {e}")
    
    print("\n" + "="*80)


def check_document_sync(file_name="legal.pdf"):
    """Check if a specific document exists in both databases."""
    print("\n" + "="*80)
    print(f"CHECKING SYNC STATUS FOR: {file_name}")
    print("="*80)
    
    # Check Neo4j
    if not graph_store:
        print("❌ Cannot check Neo4j - Graph store not initialized!")
    else:
        neo4j_query = """
        MATCH (d:Document)
        WHERE d.file_name = $file_name
        RETURN d.id AS id, d.case_id AS case_id, d.file_name AS file_name
        """
        
        neo4j_results = graph_store.run_query(neo4j_query, {"file_name": file_name})
        
        if neo4j_results:
            print(f"✅ Found in Neo4j:")
            for record in neo4j_results:
                print(f"  Document ID: {record['id']}")
                print(f"  Case ID: {record['case_id']}")
        else:
            print(f"❌ NOT found in Neo4j")
    
    # Check Qdrant
    if not vector_store:
        print("\n❌ Cannot check Qdrant - Vector store not initialized!")
    else:
        try:
            scroll_result = vector_store.client.scroll(
                collection_name=vector_store.collection_name,
                scroll_filter={
                    "must": [
                        {"key": "source", "match": {"value": file_name}}
                    ]
                },
                limit=100,
                with_payload=True
            )
            
            points = scroll_result[0]
            
            if points:
                print(f"\n✅ Found in Qdrant:")
                print(f"  Chunks: {len(points)}")
                if points:
                    print(f"  Case ID: {points[0].payload.get('case_id', 'N/A')}")
                    print(f"  Sample text: {points[0].payload.get('text', '')[:100]}...")
            else:
                print(f"\n❌ NOT found in Qdrant")
        
        except Exception as e:
            print(f"❌ Error checking Qdrant: {e}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    print("\n🔍 Starting Database Diagnostics...\n")
    
    # Check both databases
    check_neo4j()
    check_qdrant()
    check_document_sync()
    
    print("\n✅ Diagnostic check complete!\n")
