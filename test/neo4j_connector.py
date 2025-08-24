from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, basic_auth
import json
from datetime import datetime
import re
from config import Config

class Neo4jConnector:
    def __init__(self):
        self.driver = None
        self.current_database = None
    
    def connect(self, database: str = "neo4j") -> bool:
        """Connect to Neo4j database using direct driver"""
        try:
            self.driver = GraphDatabase.driver(
                Config.NEO4J_URI,
                auth=basic_auth(Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD),
                max_connection_lifetime=30 * 60  # 30 minutes
            )
            
            # Test connection with the specified database
            with self.driver.session(database=database) as session:
                try:
                    result = session.run("RETURN 1 as test")
                    if result.single()["test"] == 1:
                        self.current_database = database
                        print(f"✅ Connected to Neo4j database: {database}")
                        return True
                except Exception as e:
                    if "Database does not exist" in str(e):
                        print(f"⚠️ Database '{database}' does not exist. Creating it...")
                        if self.create_database(database):
                            return self.connect(database)  # Reconnect to new database
                    else:
                        raise e
            
            return False
            
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            self._print_neo4j_setup_instructions()
            return False
    
    def _sanitize_database_name(self, name: str) -> str:
        """Sanitize database name to meet Neo4j requirements"""
        # Remove special characters, keep only alphanumeric and underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Ensure it starts with a letter
        if sanitized and not sanitized[0].isalpha():
            sanitized = 'db_' + sanitized
        # Limit length
        return sanitized[:63]  # Neo4j has a length limit
    
    def _print_neo4j_setup_instructions(self):
        """Print Neo4j setup instructions"""
        print("\n" + "="*60)
        print("NEO4J SETUP INSTRUCTIONS")
        print("="*60)
        print("1. Download and install Neo4j Desktop: https://neo4j.com/download/")
        print("2. Start Neo4j database")
        print("3. Default connection details (update config.py if different):")
        print(f"   URI: {Config.NEO4J_URI}")
        print(f"   Username: {Config.NEO4J_USERNAME}")
        print("   Password: [your_password]")
        print("4. Database names must use only: a-z, A-Z, 0-9, underscores")
        print("5. No special characters or spaces allowed")
        print("="*60)
    
    def database_exists(self, database_name: str) -> bool:
        """Check if a database exists"""
        try:
            with self.driver.session(database="system") as session:
                result = session.run(
                    "SHOW DATABASES WHERE name = $name", 
                    {"name": database_name}
                )
                return bool(list(result))
        except Exception as e:
            print(f"❌ Error checking database existence: {e}")
            return False
    
    def create_database(self, database_name: str) -> bool:
        """Create a new database with proper sanitization"""
        # Sanitize the database name
        sanitized_name = self._sanitize_database_name(database_name)
        
        if sanitized_name != database_name:
            print(f"⚠️ Sanitized database name: '{database_name}' -> '{sanitized_name}'")
            database_name = sanitized_name
        
        try:
            with self.driver.session(database="system") as session:
                # Check if database already exists
                if self.database_exists(database_name):
                    print(f"✅ Database '{database_name}' already exists")
                    return True
                
                # Create the database
                session.run(f"CREATE DATABASE `{database_name}`")
                print(f"✅ Created database: {database_name}")
                
                # Wait for database to be ready
                import time
                time.sleep(2)  # Give Neo4j time to create the database
                
                return True
                
        except Exception as e:
            print(f"❌ Error creating database '{database_name}': {e}")
            print("💡 Database names must use only: a-z, A-Z, 0-9, underscores")
            print("💡 No special characters or spaces allowed")
            return False
    
    def push_graph_data(self, graph_documents: List[Any], clear_existing: bool = True) -> bool:
        """Push graph data to Neo4j using direct Cypher queries"""
        if not self.driver:
            print("❌ Not connected to Neo4j")
            return False
        
        try:
            if clear_existing:
                self._clear_database()
                print("🗑️ Cleared existing data")
            
            total_nodes = 0
            total_relationships = 0
            
            with self.driver.session(database=self.current_database) as session:
                # Process each graph document
                for graph_doc in graph_documents:
                    # Add nodes
                    for node in graph_doc.nodes:
                        # Sanitize node type (Neo4j labels can't have spaces)
                        node_type = node.type.replace(' ', '_')
                        
                        query = f"""
                        MERGE (n:`{node_type}` {{id: $id}})
                        SET n.name = $name,
                            n.type = $type,
                            n.created_at = datetime()
                        """
                        session.run(query, {
                            "id": node.id,
                            "name": node.id,
                            "type": node.type
                        })
                        total_nodes += 1
                    
                    # Add relationships
                    for rel in graph_doc.relationships:
                        # Sanitize relationship type
                        rel_type = rel.type.replace(' ', '_')
                        source_type = rel.source.type.replace(' ', '_')
                        target_type = rel.target.type.replace(' ', '_')
                        
                        query = f"""
                        MATCH (a:`{source_type}` {{id: $source_id}})
                        MATCH (b:`{target_type}` {{id: $target_id}})
                        MERGE (a)-[r:`{rel_type}`]->(b)
                        SET r.created_at = datetime(),
                            r.source_paper = $source_paper
                        """
                        session.run(query, {
                            "source_id": rel.source.id,
                            "target_id": rel.target.id,
                            "source_paper": graph_doc.metadata.get("source_papers", ["unknown"])[0]
                        })
                        total_relationships += 1
                
                # Add paper metadata
                paper_sources = set()
                for graph_doc in graph_documents:
                    for paper in graph_doc.metadata.get("source_papers", []):
                        paper_sources.add(paper)
                
                for paper in paper_sources:
                    session.run("""
                    MERGE (p:Paper {name: $name})
                    SET p.upload_time = datetime(),
                        p.processed_at = datetime()
                    """, {"name": paper})
            
            print(f"✅ Pushed {total_nodes} nodes and {total_relationships} relationships to Neo4j")
            return True
            
        except Exception as e:
            print(f"❌ Error pushing data to Neo4j: {e}")
            return False
    
    def _clear_database(self):
        """Clear all data from current database"""
        try:
            with self.driver.session(database=self.current_database) as session:
                session.run("MATCH (n) DETACH DELETE n")
                print("✅ Database cleared successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not clear database: {e}")
    
    def get_available_databases(self) -> List[str]:
        """Get list of available databases"""
        try:
            with self.driver.session(database="system") as session:
                result = session.run("SHOW DATABASES")
                databases = []
                for record in result:
                    db_name = record["name"]
                    if db_name not in ["system", "neo4j"]:
                        databases.append(db_name)
                return databases
        except Exception as e:
            print(f"❌ Error getting databases: {e}")
            return []
    
    def test_query(self, query: str, params: Dict = None) -> List[Dict]:
        """Execute a test query"""
        try:
            with self.driver.session(database=self.current_database) as session:
                result = session.run(query, params or {})
                return [dict(record) for record in result]
        except Exception as e:
            print(f"❌ Query failed: {e}")
            return []
    
    def get_schema(self) -> Dict:
        """Get database schema information"""
        try:
            with self.driver.session(database=self.current_database) as session:
                # Get node labels
                node_labels = session.run("CALL db.labels() YIELD label RETURN collect(label) as labels")
                labels = node_labels.single()["labels"] if node_labels else []
                
                # Get relationship types
                rel_types = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types")
                types = rel_types.single()["types"] if rel_types else []
                
                # Get property keys
                properties = session.run("CALL db.propertyKeys() YIELD propertyKey RETURN collect(propertyKey) as keys")
                prop_keys = properties.single()["keys"] if properties else []
                
                return {
                    "node_labels": labels,
                    "relationship_types": types,
                    "property_keys": prop_keys
                }
        except Exception as e:
            print(f"❌ Error getting schema: {e}")
            return {}
    
    def close(self):
        """Close connection"""
        if self.driver:
            self.driver.close()
            print("✅ Neo4j connection closed")

# Validation function
def validate_neo4j_connection():
    """Test Neo4j connection functionality"""
    connector = Neo4jConnector()
    
    if connector.connect():
        try:
            # Test with a simple, valid database name
            test_db = "test_research"
            
            # Create test database
            if connector.create_database(test_db):
                print(f"✅ Successfully created database: {test_db}")
                
                # Switch to test database
                if connector.connect(test_db):
                    # Test a simple query
                    result = connector.test_query("RETURN 'Hello Neo4j' AS message")
                    if result and result[0].get("message") == "Hello Neo4j":
                        print("✅ Basic Neo4j operations working")
                    
                    # Test schema creation with academic data
                    schema_test = """
                    CREATE (p:Paper {name: 'Test_Paper', year: 2023})
                    CREATE (a:Author {name: 'Test_Author', affiliation: 'Test_University'})
                    CREATE (a)-[:AUTHORED]->(p)
                    RETURN p.name as paper, a.name as author
                    """
                    result = connector.test_query(schema_test)
                    if result:
                        print("✅ Academic schema creation test passed")
                    
                    # Test getting schema
                    schema = connector.get_schema()
                    if schema.get("node_labels"):
                        print(f"✅ Schema retrieval: {schema['node_labels']}")
            
            connector.close()
            return True
            
        except Exception as e:
            print(f"❌ Neo4j test operations failed: {e}")
            connector.close()
            return False
    else:
        print("❌ Failed to connect to Neo4j")
        return False

def test_database_naming():
    """Test various database naming scenarios"""
    connector = Neo4jConnector()
    
    test_names = [
        "research_papers",          # Good
        "Research-Papers-2024",     # Contains dashes (will be sanitized)
        "my research db",           # Contains spaces (will be sanitized)
        "123research",              # Starts with number (will be sanitized)
        "test@db#2024",             # Contains special chars (will be sanitized)
    ]
    
    print("\nTesting database naming rules...")
    for name in test_names:
        sanitized = connector._sanitize_database_name(name)
        print(f"  '{name}' -> '{sanitized}'")
    
    connector.close()

if __name__ == "__main__":
    print("Testing Neo4j connection...")
    validate_neo4j_connection()
    
    print("\nTesting database naming...")
    test_database_naming()