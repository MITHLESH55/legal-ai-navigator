"""
Script to Re-index Documents from Neo4j into Qdrant
This script queries Neo4j for all documents and re-indexes them into Qdrant.
"""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from data_stores import vector_store, graph_store, load_and_split_pdf

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_unique_documents_from_neo4j():
    """Get unique documents from Neo4j (avoiding duplicates)."""
    query = """
    MATCH (d:Document)
    RETURN DISTINCT d.file_name AS file_name, d.case_id AS case_id
    ORDER BY d.file_name
    """
    results = graph_store.run_query(query)
    
    # Group by file_name to get unique files with their case_ids
    unique_docs = {}
    for record in results:
        file_name = record.get('file_name')
        case_id = record.get('case_id')
        if file_name and case_id:
            if file_name not in unique_docs:
                unique_docs[file_name] = []
            unique_docs[file_name].append(case_id)
    
    return unique_docs


def reindex_from_local_files():
    """Re-index documents from local files if they exist."""
    if not graph_store:
        logger.error("❌ Graph store not initialized!")
        return
    
    if not vector_store:
        logger.error("❌ Vector store not initialized!")
        return
    
    # Get unique documents from Neo4j
    unique_docs = get_unique_documents_from_neo4j()
    
    if not unique_docs:
        logger.warning("⚠️  No documents found in Neo4j to re-index")
        return
    
    logger.info(f"📄 Found {len(unique_docs)} unique document(s) to re-index:")
    for file_name, case_ids in unique_docs.items():
        logger.info(f"  - {file_name} (case_ids: {', '.join(case_ids)})")
    
    # Check for local PDF files
    possible_locations = [
        Path('./'),
        Path('./uploads'),
        Path('../'),
        Path('./docs'),
    ]
    
    indexed_count = 0
    
    for file_name, case_ids in unique_docs.items():
        # Try to find the file locally
        file_found = False
        pdf_path = None
        
        for location in possible_locations:
            potential_path = location / file_name
            if potential_path.exists():
                pdf_path = potential_path
                file_found = True
                logger.info(f"✅ Found {file_name} at {pdf_path}")
                break
        
        if not file_found:
            logger.warning(f"⚠️  Could not find {file_name} locally. Skipping...")
            logger.info(f"   Searched in: {[str(loc) for loc in possible_locations]}")
            continue
        
        # Use the first case_id for indexing
        case_id = case_ids[0]
        
        try:
            # Load and split the PDF
            logger.info(f"📖 Loading and splitting {file_name}...")
            documents = load_and_split_pdf(str(pdf_path))
            
            if not documents:
                logger.error(f"❌ No documents extracted from {file_name}")
                continue
            
            # Index into Qdrant
            logger.info(f"🔄 Indexing {len(documents)} chunks into Qdrant...")
            success = vector_store.index_documents(documents, case_id, file_name)
            
            if success:
                logger.info(f"✅ Successfully re-indexed {file_name}")
                indexed_count += 1
            else:
                logger.error(f"❌ Failed to index {file_name}")
        
        except Exception as e:
            logger.error(f"❌ Error processing {file_name}: {e}")
            continue
    
    logger.info(f"\n✅ Re-indexing complete! Successfully indexed {indexed_count}/{len(unique_docs)} documents")


def manual_reindex(pdf_path: str, case_id: str):
    """Manually re-index a specific PDF file."""
    if not vector_store:
        logger.error("❌ Vector store not initialized!")
        return
    
    pdf_file = Path(pdf_path)
    
    if not pdf_file.exists():
        logger.error(f"❌ File not found: {pdf_path}")
        return
    
    try:
        # Load and split the PDF
        logger.info(f"📖 Loading and splitting {pdf_file.name}...")
        documents = load_and_split_pdf(str(pdf_file))
        
        if not documents:
            logger.error(f"❌ No documents extracted from {pdf_file.name}")
            return
        
        # Index into Qdrant
        logger.info(f"🔄 Indexing {len(documents)} chunks into Qdrant...")
        success = vector_store.index_documents(documents, case_id, pdf_file.name)
        
        if success:
            logger.info(f"✅ Successfully indexed {pdf_file.name}")
        else:
            logger.error(f"❌ Failed to index {pdf_file.name}")
    
    except Exception as e:
        logger.error(f"❌ Error processing {pdf_file.name}: {e}")


if __name__ == "__main__":
    import sys
    
    print("\n🔄 Document Re-indexing Tool\n")
    
    if len(sys.argv) > 1:
        # Manual mode: python reindex_documents.py <pdf_path> <case_id>
        if len(sys.argv) < 3:
            print("Usage: python reindex_documents.py <pdf_path> <case_id>")
            print("   Or: python reindex_documents.py  (to auto-reindex from Neo4j)")
            sys.exit(1)
        
        pdf_path = sys.argv[1]
        case_id = sys.argv[2]
        
        print(f"📄 Manual re-indexing: {pdf_path} (case_id: {case_id})\n")
        manual_reindex(pdf_path, case_id)
    else:
        # Auto mode: re-index all documents from Neo4j
        print("📄 Auto re-indexing from Neo4j database\n")
        reindex_from_local_files()
    
    print("\n✅ Done!\n")
