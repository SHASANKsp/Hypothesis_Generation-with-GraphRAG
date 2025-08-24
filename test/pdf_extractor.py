import os
import tempfile
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime
import json
from config import Config

class PDFExtractor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""])
    
    def extract_from_pdf(self, pdf_path: str, paper_name: str = None) -> List[Document]:
        """Extract structured content from PDF with academic paper focus"""
        try:
            if not paper_name:
                paper_name = os.path.basename(pdf_path)
            
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            split_docs = self.text_splitter.split_documents(documents)
            
            for i, doc in enumerate(split_docs):
                doc.metadata.update({
                    "chunk_id": i,
                    "paper_name": paper_name,
                    "total_chunks": len(split_docs),
                    "extraction_time": datetime.now().isoformat(),
                    "file_path": pdf_path})
            return split_docs
        except Exception as e:
            print(f"Error extracting from PDF {pdf_path}: {e}")
            return []
    
    def save_extracted_content(self, documents: List[Document], output_dir: str = None):
        """Save extracted content with proper formatting"""
        if output_dir is None:
            output_dir = Config.EXTRACTED_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        paper_groups = {}
        for doc in documents:
            paper_name = doc.metadata.get("paper_name", "unknown")
            if paper_name not in paper_groups:
                paper_groups[paper_name] = []
            paper_groups[paper_name].append(doc)
        
        for paper_name, docs in paper_groups.items():
            clean_name = "".join(c for c in paper_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            clean_name = clean_name.replace(' ', '_')[:50]
            
            output_data = {
                "paper_name": paper_name,
                "extraction_time": datetime.now().isoformat(),
                "total_chunks": len(docs),
                "chunks": []}
            
            for doc in docs:
                output_data["chunks"].append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "chunk_id": doc.metadata.get("chunk_id")})
            
            output_file = os.path.join(output_dir, f"{clean_name}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"Saved extracted content for {paper_name} to {output_file}")

# Validation function
def validate_pdf_extraction():
    """Test PDF extraction functionality"""
    extractor = PDFExtractor()
    
    # Create a test PDF (you can use any existing PDF)
    test_pdf_path = "FM_biology.pdf"  # Replace with actual test file
    
    if os.path.exists(test_pdf_path):
        documents = extractor.extract_from_pdf(test_pdf_path, "test_paper")
        
        print(f"Extracted {len(documents)} chunks")
        print(f"First chunk metadata: {documents[0].metadata if documents else 'No documents'}")
        
        extractor.save_extracted_content(documents)
        return True
    else:
        print(f"Test PDF not found at {test_pdf_path}")
        return False

if __name__ == "__main__":
    validate_pdf_extraction()