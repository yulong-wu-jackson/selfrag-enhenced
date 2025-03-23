"""
Utility functions for the Self-RAG system.
"""
import logging
from typing import Dict, List, Any

from src.config.config import get_config

# Configure logging
config = get_config()
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def format_documents(docs: List[Dict[str, Any]]) -> str:
    """Format retrieved documents for use in prompts."""
    formatted_docs = []
    for i, doc in enumerate(docs):
        content = doc.get("page_content", "")
        metadata = doc.get("metadata", {})
        source = metadata.get("source", f"Document {i+1}")
        formatted_docs.append(f"Document {i+1} (Source: {source}):\n{content}")
    
    return "\n\n".join(formatted_docs)

def convert_to_langsmith_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert data to LangSmith compatible metadata format.
    LangSmith has restrictions on metadata values that can be tracked.
    """
    processed = {}
    for key, value in data.items():
        # Convert complex objects to strings
        if isinstance(value, (dict, list)):
            processed[key] = str(value)
        else:
            processed[key] = value
    return processed 