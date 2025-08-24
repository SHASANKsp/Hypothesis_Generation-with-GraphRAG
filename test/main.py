import streamlit as st
import tempfile
import os
from datetime import datetime
from pdf_extractor import PDFExtractor
from graph_builder import GraphBuilder
from neo4j_connector import Neo4jConnector
from graph_rag import GraphRAGSystem
from utils import load_graph_configurations, save_graph_configuration, get_paper_list
from config import Config

def main():
    st.set_page_config(
        page_title="Research Paper GraphRAG",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Research Paper GraphRAG with Neo4j & Ollama")
    
    # Initialize session state
    if 'pdf_extractor' not in st.session_state:
        st.session_state.pdf_extractor = PDFExtractor()
    if 'graph_builder' not in st.session_state:
        st.session_state.graph_builder = GraphBuilder()
    if 'neo4j_connector' not in st.session_state:
        st.session_state.neo4j_connector = Neo4jConnector()
    if 'graph_rag' not in st.session_state:
        st.session_state.graph_rag = GraphRAGSystem()
    if 'processed_papers' not in st.session_state:
        st.session_state.processed_papers = get_paper_list()
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Graph database input
        graph_name = st.text_input("Graph Database Name", "research_papers")
        
        # Connect to Neo4j
        if st.button("Connect to Neo4j"):
            if st.session_state.neo4j_connector.connect(graph_name):
                st.success(f"Connected to: {graph_name}")
            else:
                st.error("Failed to connect to Neo4j")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Research Papers (PDF)",
            type="pdf",
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("Process Papers"):
            with st.spinner("Processing papers..."):
                for uploaded_file in uploaded_files:
                    # Save uploaded file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Extract content
                    documents = st.session_state.pdf_extractor.extract_from_pdf(
                        tmp_path, uploaded_file.name
                    )
                    
                    if documents:
                        # Build graph
                        graph_docs = st.session_state.graph_builder.build_graph_from_documents(documents)
                        
                        if graph_docs and st.session_state.neo4j_connector.graph:
                            # Push to Neo4j
                            if st.session_state.neo4j_connector.push_graph_data(graph_docs):
                                # Save configuration
                                save_graph_configuration(graph_name, [uploaded_file.name])
                                st.success(f"Processed: {uploaded_file.name}")
                    
                    # Clean up
                    os.unlink(tmp_path)
        
        # Available graphs
        st.header("Available Graphs")
        graph_configs = load_graph_configurations()
        if graph_configs:
            graph_names = [config["name"] for config in graph_configs]
            selected_graph = st.selectbox("Select Graph", graph_names)
            
            if st.button("Load Selected Graph"):
                if st.session_state.neo4j_connector.connect(selected_graph):
                    st.success(f"Loaded graph: {selected_graph}")
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("System Status")
        
        # Connection status
        if st.session_state.neo4j_connector.graph:
            st.success("✅ Connected to Neo4j")
        else:
            st.warning("⚠️ Not connected to Neo4j")
        
        # Processed papers
        st.subheader("Processed Papers")
        papers = get_paper_list()
        if papers:
            for paper in papers:
                st.write(f"• {paper}")
        else:
            st.info("No papers processed yet")
    
    with col2:
        st.header("Research Assistant")
        
        if st.session_state.neo4j_connector.graph:
            # Initialize RAG chain
            if st.session_state.graph_rag.initialize_chain(st.session_state.neo4j_connector.graph):
                # Chat interface
                if "messages" not in st.session_state:
                    st.session_state.messages = []
                
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                
                if prompt := st.chat_input("Ask about the research..."):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    with st.chat_message("assistant"):
                        with st.spinner("Researching..."):
                            response = st.session_state.graph_rag.query(prompt)
                            if "answer" in response:
                                st.markdown(response["answer"])
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": response["answer"]
                                })
                            else:
                                st.error("Error generating response")
                
                # Summary generation
                if st.button("Generate Comprehensive Summary"):
                    with st.spinner("Generating summary..."):
                        summary = st.session_state.graph_rag.generate_summary(
                            st.session_state.neo4j_connector.current_database
                        )
                        if "answer" in summary:
                            st.subheader("Research Summary")
                            st.markdown(summary["answer"])
            else:
                st.error("Failed to initialize research assistant")
        else:
            st.info("Please connect to a graph database to start chatting")

if __name__ == "__main__":
    main()