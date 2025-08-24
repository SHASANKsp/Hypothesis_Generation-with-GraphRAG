import os
import json
import requests
from typing import List, Dict, Any
from datetime import datetime
from config import Config

def load_graph_configurations() -> List[Dict]:
    """Load saved graph configurations"""
    if os.path.exists(Config.GRAPH_CONFIG_FILE):
        try:
            with open(Config.GRAPH_CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_graph_configuration(graph_name: str, papers: List[str]) -> bool:
    """Save graph configuration"""
    configs = load_graph_configurations()
    
    # Check if graph already exists
    for config in configs:
        if config["name"] == graph_name:
            # Update existing configuration
            config["papers"] = list(set(config["papers"] + papers))
            config["updated_at"] = datetime.now().isoformat()
            break
    else:
        # Add new configuration
        new_config = {
            "name": graph_name,
            "papers": papers,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        configs.append(new_config)
    
    try:
        with open(Config.GRAPH_CONFIG_FILE, 'w') as f:
            json.dump(configs, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving graph configuration: {e}")
        return False

def get_paper_list(extracted_dir: str = None) -> List[str]:
    """Get list of processed papers"""
    if extracted_dir is None:
        extracted_dir = Config.EXTRACTED_DIR
    
    papers = set()
    if os.path.exists(extracted_dir):
        for filename in os.listdir(extracted_dir):
            if filename.endswith('.json'):
                paper_name = filename.replace('.json', '')
                papers.add(paper_name)
    
    return sorted(list(papers))

def validate_ollama_connection():
    """Test Ollama connection and model availability"""
    try:
        # Check if Ollama is running
        response = requests.get(f"{Config.OLLAMA_BASE_URL}/api/tags")
        if response.status_code != 200:
            print("❌ Ollama server not running. Please start Ollama with: ollama serve")
            return False
        
        available_models = [model['name'] for model in response.json().get('models', [])]
        print(f"✅ Ollama server running at {Config.OLLAMA_BASE_URL}")
        print(f"📦 Available models: {available_models}")
        
        # Check if configured model is available
        if Config.OLLAMA_MODEL in available_models:
            print(f"✅ Model '{Config.OLLAMA_MODEL}' is available")
            return True
        else:
            print(f"❌ Model '{Config.OLLAMA_MODEL}' not found in available models")
            print(f"💡 Available models: {available_models}")
            
            # Try to find a fallback model
            for fallback in Config.FALLBACK_MODELS:
                if fallback in available_models:
                    print(f"🔄 Using fallback model: {fallback}")
                    # Update config to use fallback
                    Config.OLLAMA_MODEL = fallback
                    return True
            
            print("\n🚨 No suitable models found. Please pull a model:")
            print("   ollama pull llama3.1  # Recommended")
            print("   ollama pull mistral    # Alternative")
            return False
            
    except requests.ConnectionError:
        print("❌ Cannot connect to Ollama. Please:")
        print("   1. Install Ollama: https://ollama.com/")
        print("   2. Start Ollama: ollama serve")
        print("   3. Pull a model: ollama pull llama3.1")
        return False
    except Exception as e:
        print(f"❌ Ollama connection error: {e}")
        return False

def get_ollama_llm():
    """Get Ollama LLM instance with proper import"""
    try:
        # Try new langchain-ollama package first
        from langchain_ollama import OllamaLLM
        return OllamaLLM(
            base_url=Config.OLLAMA_BASE_URL,
            model=Config.OLLAMA_MODEL
        )
    except ImportError:
        try:
            # Fallback to old langchain community
            from langchain_community.llms import Ollama
            return Ollama(
                base_url=Config.OLLAMA_BASE_URL,
                model=Config.OLLAMA_MODEL
            )
        except ImportError:
            print("❌ No Ollama integration found. Please install:")
            print("   pip install langchain-ollama")
            return None

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        import subprocess
        result = subprocess.run([
            "pip", "install", "langchain-ollama", "requests"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Packages installed successfully")
            return True
        else:
            print(f"❌ Package installation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False