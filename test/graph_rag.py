from typing import Dict, Any, Optional
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from datetime import datetime
from config import Config
from utils import get_ollama_llm
from neo4j_connector import Neo4jConnector  # Import our connector

class GraphRAGSystem:
    def __init__(self):
        self.llm = get_ollama_llm()
        self.chain = None
        self.current_graph = None
        self.neo4j_connector = Neo4jConnector()
    
    def initialize_chain(self, database: str) -> bool:
        """Initialize the Graph RAG chain"""
        if self.llm is None:
            print("❌ LLM not initialized")
            return False
        
        # Connect to the specified database
        if not self.neo4j_connector.connect(database):
            print(f"❌ Failed to connect to database: {database}")
            return False
            
        try:
            # For GraphCypherQAChain, we need to use the old Neo4jGraph temporarily
            # or implement our own query mechanism
            try:
                from langchain_community.graphs import Neo4jGraph
                graph = Neo4jGraph(
                    url=Config.NEO4J_URI,
                    username=Config.NEO4J_USERNAME,
                    password=Config.NEO4J_PASSWORD,
                    database=database
                )
            except:
                # Fallback: Use our connector for simple queries
                print("⚠️ Using fallback query mode (no advanced Graph features)")
                graph = None
            
            # Custom prompt with audit trail requirements
            qa_prompt = PromptTemplate(
                template="""You are an expert research assistant analyzing academic papers. 
                Answer questions based ONLY on the provided context from research papers.

                CRITICAL: Always include specific references to source papers in your answers.
                Format references as: [Source: PaperName]

                Context: {context}
                Question: {question}

                Provide a comprehensive answer with proper citations:""",
                input_variables=["context", "question"]
            )
            
            if graph:
                # Use LangChain's GraphCypherQAChain if available
                self.chain = GraphCypherQAChain.from_llm(
                    llm=self.llm,
                    graph=graph,
                    verbose=True,
                    return_direct=False,
                    qa_prompt=qa_prompt
                )
            else:
                # Fallback: Use simple query approach
                self.chain = self._create_simple_chain()
            
            self.current_graph = database
            return True
            
        except Exception as e:
            print(f"❌ Error initializing Graph RAG chain: {e}")
            return False
    
    def _create_simple_chain(self):
        """Create a simple chain for basic queries"""
        # This is a simplified version for when advanced features aren't available
        def simple_query_chain(inputs):
            question = inputs.get("query", "")
            
            # Simple Cypher queries based on question type
            if "author" in question.lower():
                cypher_query = "MATCH (a:Author) RETURN a.name as author LIMIT 10"
            elif "paper" in question.lower():
                cypher_query = "MATCH (p:Paper) RETURN p.name as paper LIMIT 10"
            else:
                cypher_query = "MATCH (n) RETURN n.name as name, labels(n)[0] as type LIMIT 10"
            
            # Execute query
            results = self.neo4j_connector.test_query(cypher_query)
            
            # Format context
            context = "\n".join([str(result) for result in results])
            
            # Use LLM to generate answer
            prompt = f"""Based on the following graph data, answer the question:

            Data: {context}
            Question: {question}

            Answer:"""
            
            response = self.llm.invoke(prompt)
            return {"result": response}
        
        return simple_query_chain
    
    def query(self, question: str, add_audit_trail: bool = True) -> Dict[str, Any]:
        """Query the knowledge graph with audit trail"""
        if not self.chain:
            return {"error": "Chain not initialized"}
        
        try:
            if callable(self.chain):
                # Simple chain mode
                response = self.chain({"query": question})
            else:
                # LangChain chain mode
                response = self.chain.invoke({"query": question})
            
            result = response.get("result", "No response generated")
            
            if add_audit_trail:
                audit_info = f"\n\n---\n*Generated from knowledge graph '{self.current_graph}' at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} using {Config.OLLAMA_MODEL}*"
                result += audit_info
            
            return {
                "answer": result,
                "question": question,
                "timestamp": datetime.now().isoformat(),
                "model": Config.OLLAMA_MODEL,
                "graph": self.current_graph
            }
            
        except Exception as e:
            return {"error": f"Query failed: {e}"}
    
    def generate_summary(self, graph_name: str) -> Dict[str, Any]:
        """Generate comprehensive summary of the knowledge graph"""
        summary_prompt = """
        Provide a comprehensive summary of all research papers in the knowledge graph.
        Include:
        1. Main research themes and topics covered
        2. Key findings and conclusions across papers
        3. Relationships and connections between different research works
        4. Notable authors, institutions, and publications
        5. Research gaps and potential future directions
        6. Methodologies and techniques used across papers
        
        Always reference specific papers using: [Source: PaperName]
        """
        
        response = self.query(summary_prompt, add_audit_trail=True)
        
        if "answer" in response:
            response["summary_type"] = "comprehensive"
            response["graph_name"] = graph_name
        
        return response

# Validation function
def validate_graph_rag():
    """Test Graph RAG functionality"""
    rag = GraphRAGSystem()
    if rag.llm is None:
        print("❌ Graph RAG validation failed - LLM not available")
        return False
        
    # Test with a simple database
    if rag.initialize_chain("neo4j"):
        # Test query
        test_question = "What types of nodes are in the database?"
        response = rag.query(test_question)
        
        print("Graph RAG Test Results:")
        print(f"Question: {test_question}")
        print(f"Response: {response}")
        
        return True
    else:
        print("❌ Graph RAG validation failed - could not initialize")
        return False

if __name__ == "__main__":
    validate_graph_rag()