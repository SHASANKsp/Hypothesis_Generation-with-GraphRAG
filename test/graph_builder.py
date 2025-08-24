from typing import List, Any, Dict
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from datetime import datetime
import json
from config import Config
from utils import get_ollama_llm  # Updated import

class GraphBuilder:
    def __init__(self):
        self.llm = get_ollama_llm()
        if self.llm is None:
            raise ValueError("Failed to initialize Ollama LLM")
        
        # Create custom prompt template for academic extraction
        self.academic_prompt = PromptTemplate(
            template="""You are an expert academic research assistant. Extract entities and relationships from research papers.

Focus on extracting:
- Research concepts, methods, techniques, algorithms
- Authors, institutions, publications, conferences, journals
- Citations and references between papers
- Key findings, conclusions, contributions, hypotheses
- Technical terms, domain-specific entities, mathematical concepts
- Datasets, experiments, results, evaluations

For each entity, determine the most appropriate type from: 
[ResearchConcept, Method, Author, Institution, Publication, Conference, 
Journal, Dataset, Algorithm, Result, Conclusion, Reference]

For relationships, use types like: 
[AUTHORED_BY, PUBLISHED_IN, CITED_BY, USES_METHOD, HAS_RESULT, 
COMPARED_WITH, EXTENDS, CONTRIBUTES_TO]

Return structured nodes and relationships in valid JSON format.

Text to analyze: {input}
""",
            input_variables=["input"]
        )
        
        # Initialize transformer with custom prompt
        try:
            self.transformer = LLMGraphTransformer(
                llm=self.llm,
                prompt=self.academic_prompt
            )
        except ImportError as e:
            print(f"❌ Missing dependency: {e}")
            print("💡 Please install: pip install json-repair")
            raise
        except Exception as e:
            print(f"❌ Error initializing graph transformer: {e}")
            raise
    
    def build_graph_from_documents(self, documents: List[Document]) -> List[Any]:
        """Extract nodes and edges from documents using LLM with academic focus"""
        try:
            print(f"Building academic graph from {len(documents)} documents...")
            
            # Use the transformer with our custom academic prompt
            graph_documents = self.transformer.convert_to_graph_documents(documents)
            
            # Add comprehensive metadata for audit trail
            paper_sources = list(set([doc.metadata.get("paper_name", "unknown") for doc in documents]))
            
            for graph_doc in graph_documents:
                graph_doc.metadata = {
                    "source_papers": paper_sources,
                    "extraction_time": datetime.now().isoformat(),
                    "total_chunks_processed": len(documents),
                    "model_used": Config.OLLAMA_MODEL,
                    "prompt_type": "academic_research"
                }
            
            print(f"Generated {len(graph_documents)} graph documents")
            print(f"Total nodes: {sum(len(gd.nodes) for gd in graph_documents)}")
            print(f"Total relationships: {sum(len(gd.relationships) for gd in graph_documents)}")
            
            # Print sample of extracted entities
            if graph_documents:
                print("\nSample extracted entities:")
                for i, node in enumerate(graph_documents[0].nodes[:5]):  # First 5 nodes
                    print(f"  {i+1}. {node.id} ({node.type})")
                
                print("\nSample relationships:")
                for i, rel in enumerate(graph_documents[0].relationships[:3]):  # First 3 relationships
                    print(f"  {i+1}. {rel.source.id} --{rel.type}--> {rel.target.id}")
            
            return graph_documents
            
        except Exception as e:
            print(f"Error building academic graph: {e}")
            return []
    
    def build_graph_from_text(self, text: str, paper_name: str = "test_paper") -> List[Any]:
        """Build graph from raw text (for testing)"""
        test_doc = Document(
            page_content=text,
            metadata={"paper_name": paper_name, "chunk_id": 0}
        )
        return self.build_graph_from_documents([test_doc])
    
    def save_graph_schema(self, graph_documents: List[Any], output_file: str = "graph_schema.json"):
        """Save graph schema for analysis"""
        schema_data = {
            "extraction_time": datetime.now().isoformat(),
            "total_graph_documents": len(graph_documents),
            "node_types": set(),
            "relationship_types": set(),
            "papers_processed": set(),
            "statistics": {
                "total_nodes": 0,
                "total_relationships": 0,
                "nodes_by_type": {},
                "relationships_by_type": {}
            }
        }
        
        for gd in graph_documents:
            schema_data["total_nodes"] += len(gd.nodes)
            schema_data["total_relationships"] += len(gd.relationships)
            
            for node in gd.nodes:
                schema_data["node_types"].add(node.type)
                schema_data["statistics"]["nodes_by_type"][node.type] = \
                    schema_data["statistics"]["nodes_by_type"].get(node.type, 0) + 1
            
            for rel in gd.relationships:
                schema_data["relationship_types"].add(rel.type)
                schema_data["statistics"]["relationships_by_type"][rel.type] = \
                    schema_data["statistics"]["relationships_by_type"].get(rel.type, 0) + 1
            
            for paper in gd.metadata.get("source_papers", []):
                schema_data["papers_processed"].add(paper)
        
        # Convert sets to lists for JSON serialization
        schema_data["node_types"] = list(schema_data["node_types"])
        schema_data["relationship_types"] = list(schema_data["relationship_types"])
        schema_data["papers_processed"] = list(schema_data["papers_processed"])
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(schema_data, f, indent=2, ensure_ascii=False)
        
        print(f"Graph schema saved to {output_file}")
        return schema_data

# Enhanced validation function
def validate_graph_building():
    """Test graph building functionality with academic focus"""
    try:
        builder = GraphBuilder()
    except ValueError as e:
        print(f"❌ {e}")
        return False
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Please run: pip install json-repair")
        return False
    except Exception as e:
        print(f"❌ Error initializing GraphBuilder: {e}")
        return False
    
    # Test with academic research text
    test_text = """
    In their paper "Attention Is All You Need" published at NeurIPS 2017, 
    Vaswani et al. from Google Research introduced the Transformer architecture. 
    This approach eliminates recurrence and convolution, relying on attention mechanisms.
    
    The Transformer model achieved state-of-the-art results on translation tasks, 
    outperforming previous models. Key innovations include multi-head self-attention.
    
    This work has been highly influential, cited by many subsequent papers 
    including BERT by Devlin et al. and GPT series by OpenAI.
    """
    
    print("Testing academic graph extraction...")
    print("Input text:", test_text[:150] + "...")
    
    try:
        graph_docs = builder.build_graph_from_text(test_text, "Attention_Is_All_You_Need")
        
        if graph_docs and graph_docs[0].nodes:
            print("\n✅ Academic graph building successful!")
            
            # Show detailed results
            print(f"\n📊 Extraction Results:")
            print(f"   Documents: {len(graph_docs)}")
            print(f"   Total Nodes: {sum(len(gd.nodes) for gd in graph_docs)}")
            print(f"   Total Relationships: {sum(len(gd.relationships) for gd in graph_docs)}")
            
            # Show academic-specific entities
            academic_nodes = []
            for node in graph_docs[0].nodes:
                if any(keyword.lower() in node.type.lower() for keyword in [
                    'research', 'author', 'publication', 'method', 'algorithm', 
                    'conference', 'journal', 'dataset', 'result'
                ]):
                    academic_nodes.append(node)
            
            print(f"\n🎓 Academic entities found: {len(academic_nodes)}")
            for node in academic_nodes[:8]:  # Show first 8
                print(f"   - {node.id} ({node.type})")
            
            # Save schema for analysis
            schema = builder.save_graph_schema(graph_docs)
            return True
        else:
            print("❌ Academic graph building failed - no nodes extracted")
            return False
            
    except Exception as e:
        print(f"❌ Error during graph building: {e}")
        return False

def test_simple_extraction():
    """Test with simpler text to ensure basic functionality works"""
    simple_text = """
    John Smith published a paper about machine learning at ACM Conference 2023.
    The paper introduces a new algorithm called DeepNet for image classification.
    """
    
    print("\n" + "="*50)
    print("TESTING SIMPLE EXTRACTION")
    print("="*50)
    
    try:
        builder = GraphBuilder()
        graph_docs = builder.build_graph_from_text(simple_text, "Simple_Test")
        
        if graph_docs and graph_docs[0].nodes:
            print("✅ Simple extraction successful!")
            print(f"   Nodes: {[f'{n.id} ({n.type})' for n in graph_docs[0].nodes]}")
            print(f"   Relationships: {[f'{r.source.id}->{r.target.id} ({r.type})' for r in graph_docs[0].relationships]}")
            return True
        else:
            print("❌ Simple extraction failed")
            return False
            
    except Exception as e:
        print(f"❌ Error in simple extraction: {e}")
        return False

if __name__ == "__main__":
    # Install missing dependency if needed
    try:
        import json_repair
        print("✅ json-repair package is available")
    except ImportError:
        print("❌ json-repair package missing")
        print("💡 Please run: pip install json-repair")
        exit(1)
    
    # Run both validation tests
    success1 = validate_graph_building()
    success2 = test_simple_extraction()
    
    if success1 and success2:
        print("\n🎉 All graph building tests passed!")
    else:
        print("\n❌ Some tests failed")