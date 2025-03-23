import os
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

class Config(BaseModel):
    # API Keys
    grok_api_key: str = os.getenv("GROK_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    
    # LangSmith Tracking
    langchain_tracing_v2: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    langchain_endpoint: str = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    langchain_api_key: str = os.getenv("LANGCHAIN_API_KEY", "")
    langchain_project: str = os.getenv("LANGCHAIN_PROJECT", "self-rag-project")
    
    # Other Configuration
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

# Create a global config instance
config = Config()

def get_config() -> Config:
    """Get the global configuration object."""
    return config 