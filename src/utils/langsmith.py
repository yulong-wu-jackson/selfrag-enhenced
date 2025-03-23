"""
LangSmith configuration for tracing.
"""
import os
import logging
from typing import Optional

from langsmith import Client

from src.config.config import get_config

logger = logging.getLogger(__name__)
config = get_config()

def setup_langsmith() -> Optional[Client]:
    """
    Configure LangSmith tracing.
    
    Returns:
        Optional LangSmith client if configured, None otherwise
    """
    if not config.langchain_api_key:
        logger.warning("LangSmith API key not set. Tracing disabled.")
        return None
        
    try:
        # Set environment variables for LangSmith
        os.environ["LANGCHAIN_TRACING_V2"] = "true" if config.langchain_tracing_v2 else "false"
        os.environ["LANGCHAIN_ENDPOINT"] = config.langchain_endpoint
        os.environ["LANGCHAIN_API_KEY"] = config.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = config.langchain_project
        
        # Create client
        client = Client(
            api_key=config.langchain_api_key,
            api_url=config.langchain_endpoint,
        )
        
        logger.info(f"LangSmith configured with project: {config.langchain_project}")
        return client
    except Exception as e:
        logger.error(f"Error setting up LangSmith: {e}")
        return None 