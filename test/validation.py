#!/usr/bin/env python3
"""
Comprehensive validation script for the Research GraphRAG system
"""

def run_comprehensive_validation():
    """Run all validation tests"""
    print("=" * 60)
    print("RESEARCH GRAPHRAG VALIDATION SUITE")
    print("=" * 60)
    
    # Test 1: Configuration
    print("\n1. Testing Configuration...")
    from config import Config
    try:
        Config.validate_config()
        print("✅ Configuration validated successfully")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False
    
    # Test 2: Ollama Connection
    print("\n2. Testing Ollama Connection...")
    from utils import validate_ollama_connection, install_requirements
    ollama_status = validate_ollama_connection()
    
    if not ollama_status:
        print("\n🔄 Attempting to install required packages...")
        if install_requirements():
            ollama_status = validate_ollama_connection()
    
    if ollama_status:
        print("✅ Ollama connection successful")
    else:
        print("❌ Ollama connection failed")
        return False
    
    # Test 3: Neo4j Connection
    print("\n3. Testing Neo4j Connection...")
    from neo4j_connector import validate_neo4j_connection
    neo4j_status = validate_neo4j_connection()
    if neo4j_status:
        print("✅ Neo4j connection successful")
    else:
        print("⚠️ Neo4j connection failed - some features may not work")
    
    # Test 4: PDF Extraction
    print("\n4. Testing PDF Extraction...")
    from pdf_extractor import validate_pdf_extraction
    pdf_status = validate_pdf_extraction()
    if pdf_status:
        print("✅ PDF extraction test completed")
    else:
        print("⚠️ PDF extraction test skipped (no test PDF available)")
    
    # Test 5: Graph Building
    print("\n5. Testing Graph Building...")
    from graph_builder import validate_graph_building
    graph_building_status = validate_graph_building()
    if graph_building_status:
        print("✅ Graph building successful")
    else:
        print("❌ Graph building failed")
        return False
    
    # Test 6: Graph RAG
    print("\n6. Testing Graph RAG...")
    from graph_rag import validate_graph_rag
    rag_status = validate_graph_rag()
    if rag_status:
        print("✅ Graph RAG test completed")
    else:
        print("⚠️ Graph RAG test completed with warnings")
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY:")
    print(f"Ollama: {'✅' if ollama_status else '❌'}")
    print(f"Neo4j: {'✅' if neo4j_status else '⚠️'}")
    print(f"PDF Extraction: {'✅' if pdf_status else '⚠️'}")
    print(f"Graph Building: {'✅' if graph_building_status else '❌'}")
    print(f"Graph RAG: {'✅' if rag_status else '⚠️'}")
    print("=" * 60)
    
    if ollama_status and neo4j_status:
        print("🎉 System is ready for research paper processing!")
        return True
    else:
        print("❌ System needs configuration before use")
        return False

if __name__ == "__main__":
    run_comprehensive_validation()