"""
Simple Query Script to Check Neo4j Document Nodes
Run this with: curl or directly in Neo4j Browser
"""

# Query 1: Check all Document nodes
print("Query 1: Check all Document nodes")
print("-" * 60)
query1 = """
MATCH (d:Document)
RETURN d.id AS id, d.case_id AS case_id, d.file_name AS file_name, 
       d.chunk_id AS chunk_id, d.updated_at AS updated_at
ORDER BY d.updated_at DESC
LIMIT 25
"""
print(query1)
print("\n")

# Query 2: Check Document nodes missing properties
print("Query 2: Check Document nodes with missing properties")
print("-" * 60)
query2 = """
MATCH (d:Document)
WHERE d.case_id IS NULL OR d.file_name IS NULL
RETURN d.id AS id, d.case_id AS case_id, d.file_name AS file_name
"""
print(query2)
print("\n")

# Query 3: Count entities per document
print("Query 3: Count entities linked to each document")
print("-" * 60)
query3 = """
MATCH (d:Document)
OPTIONAL MATCH (d)<-[:APPEARS_IN]-(n:Entity)
RETURN d.file_name AS document, d.case_id AS case_id, count(n) AS entity_count
ORDER BY entity_count DESC
"""
print(query3)
print("\n")

# Query 4: Check specific file
print("Query 4: Check for legal.pdf")
print("-" * 60)
query4 = """
MATCH (d:Document)
WHERE d.file_name = 'legal.pdf'
RETURN d
"""
print(query4)
print("\n")

# Query 5: List all entity types
print("Query 5: List all entity types")
print("-" * 60)
query5 = """
MATCH (n:Entity)
WITH labels(n) AS labels, count(n) AS count
RETURN labels, count
ORDER BY count DESC
"""
print(query5)
print("\n")

print("=" * 60)
print("INSTRUCTIONS:")
print("1. Open Neo4j Browser at http://localhost:7474")
print("2. Copy and paste each query above")
print("3. Run them to check the database status")
print("=" * 60)
