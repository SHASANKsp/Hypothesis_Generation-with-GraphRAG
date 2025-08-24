import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Neo4j Configuration
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "graphrag")
    
    # Ollama Configuration
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:latest")
    
    # Alternative models if primary not available
    FALLBACK_MODELS = ["llama3.1", "llama3", "mistral", "gemma:2b"]
    
    # File Paths
    UPLOAD_DIR = "uploads"
    EXTRACTED_DIR = "extracted_content"
    GRAPH_CONFIG_FILE = "graph_configs.json"
    
    # Processing Settings
    CHUNK_SIZE = 10000
    CHUNK_OVERLAP = 1000
    
    @classmethod
    def validate_config(cls):
        """Validate all required configurations are set"""
        required_vars = {
            "NEO4J_URI": cls.NEO4J_URI,
            "NEO4J_USERNAME": cls.NEO4J_USERNAME,
            "NEO4J_PASSWORD": cls.NEO4J_PASSWORD,
            "OLLAMA_BASE_URL": cls.OLLAMA_BASE_URL
        }
        
        missing = [var for var, value in required_vars.items() if not value]
        if missing:
            raise ValueError(f"Missing configuration: {missing}")
        
        # Create directories
        os.makedirs(cls.UPLOAD_DIR, exist_ok=True)
        os.makedirs(cls.EXTRACTED_DIR, exist_ok=True)
        
        return True
    
    @classmethod
    def get_available_models(cls):
        """Get list of available Ollama models"""
        try:
            import requests
            response = requests.get(f"{cls.OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                models = [model['name'] for model in response.json().get('models', [])]
                return models
        except:
            return []
        
        return []
    
    @classmethod
    def get_default_database(cls):
        """Get default database name"""
        return "research_db"