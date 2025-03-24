"""
Traditional RAG Agent implementation for comparison with Self-RAG.
"""
import logging
from typing import Dict, List, Any, Optional, TypedDict

from langchain.schema import BaseRetriever
from langchain_core.runnables import RunnableConfig

from src.models.llm import get_llm
from src.utils.helpers import format_documents
from src.config.config import get_config

logger = logging.getLogger(__name__)
app_config = get_config()

# Simple prompt for traditional RAG response generation
TRADITIONAL_RAG_PROMPT = """You are a helpful assistant. Your task is to answer the user's query using the retrieved documents.

Query: {query}

Retrieved Documents:
{documents}

Generate a comprehensive response answering the user's query. Follow these guidelines:
1. Base your response on the information in the retrieved documents.
2. Cite specific documents using [Doc X] notation where X is the document number.
3. If the documents don't contain relevant information, respond based on your knowledge.
4. Be truthful and do not make up information.
5. If you don't know the answer, say so clearly.

Response:"""

class TraditionalRAGState(TypedDict):
    query: str
    documents: Optional[List[Dict[str, Any]]]
    response: Optional[str]
    metadata: Optional[Dict[str, Any]]

class TraditionalRAG:
    """
    Traditional RAG Agent class.
    
    Implements a standard RAG system with:
    1. Document retrieval
    2. Response generation
    
    No advanced features like query analysis, relevance assessment,
    or response evaluation that Self-RAG provides.
    """
    
    def __init__(self, retriever: BaseRetriever):
        """
        Initialize the Traditional RAG agent.
        
        Args:
            retriever: Document retriever to use
        """
        self.retriever = retriever
    
    def retrieve_documents(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve documents based on the query.
        
        Args:
            query: User query
        
        Returns:
            List of retrieved documents as dictionaries
        """
        # Retrieve documents
        docs = self.retriever.get_relevant_documents(query)
        
        # Convert to dicts for easier serialization
        docs_dicts = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in docs
        ]
        
        return docs_dicts
    
    def generate_response(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """
        Generate a response based on the query and retrieved documents.
        
        Args:
            query: User query
            documents: Retrieved documents
        
        Returns:
            Generated response
        """
        llm = get_llm()
        
        # Format documents for prompt
        formatted_docs = format_documents(documents) if documents else "No relevant documents found."
        
        # Generate response
        prompt = TRADITIONAL_RAG_PROMPT.format(
            query=query,
            documents=formatted_docs
        )
        
        response = llm.invoke(prompt).strip()
        return response
    
    def invoke(self, query: str, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """
        Run the Traditional RAG agent on a query.
        
        Args:
            query: User query
            config: Optional runnable config
        
        Returns:
            Dict with response and state information
        """
        # Set up metadata
        if config is None:
            run_config = {"metadata": {"query": query}}
            if app_config.langchain_tracing_v2:
                run_config["callbacks"] = []  # LangSmith callbacks will be added automatically
        else:
            run_config = config
        
        # Initialize state
        state: TraditionalRAGState = {
            "query": query,
            "documents": None,
            "response": None,
            "metadata": {"system": "Traditional RAG"}
        }
        
        # Step 1: Retrieve documents
        documents = self.retrieve_documents(query)
        state["documents"] = documents
        state["metadata"]["retrieval"] = f"Retrieved {len(documents)} documents"
        
        # Step 2: Generate response
        response = self.generate_response(query, documents)
        state["response"] = response
        state["metadata"]["response_generation"] = "Response generated"
        
        return state
    
    def get_response(self, query: str) -> str:
        """
        Get just the response text from the Traditional RAG agent.
        
        Args:
            query: User query
        
        Returns:
            Response text
        """
        result = self.invoke(query)
        return result.get("response", "No response generated.") 