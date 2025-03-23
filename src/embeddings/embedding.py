"""
Embeddings configuration using OpenAI.
"""
import logging
from typing import List

from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings

from src.config.config import get_config

logger = logging.getLogger(__name__)
config = get_config()

def get_embeddings() -> Embeddings:
    """Get an instance of OpenAI embeddings."""
    return OpenAIEmbeddings(
        api_key=config.openai_api_key,
        model="text-embedding-ada-002",
    ) 