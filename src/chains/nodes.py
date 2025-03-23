"""
Node implementations for Self-RAG.
"""
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

from langchain_core.documents import Document
from langchain.schema import BaseRetriever
import langgraph.graph
from langchain.schema.runnable import RunnableLambda, Runnable
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.utils import Output

from src.models.llm import get_llm
from src.chains.prompts import (
    QUERY_ANALYSIS_PROMPT,
    RELEVANCE_ASSESSMENT_PROMPT,
    RESPONSE_GENERATION_PROMPT,
)
from src.utils.helpers import format_documents, convert_to_langsmith_metadata

logger = logging.getLogger(__name__)

def analyze_query(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the query to determine if retrieval is needed.
    
    Args:
        state: Current state including the query
        
    Returns:
        Updated state with retrieval decision
    """
    query = state["query"]
    llm = get_llm()
    
    # Analyze query
    prompt = QUERY_ANALYSIS_PROMPT.format(query=query)
    response = llm.invoke(prompt).strip()
    
    # Update state
    state["retrieve_decision"] = response
    state["metadata"] = {
        "query_analysis": response
    }
    
    return state

def retrieve_documents(state: Dict[str, Any], retriever: BaseRetriever) -> Dict[str, Any]:
    """
    Retrieve documents based on the query.
    
    Args:
        state: Current state
        retriever: Document retriever
    
    Returns:
        State with retrieved documents
    """
    query = state["query"]
    
    # Only retrieve if the decision was to retrieve
    if state.get("retrieve_decision", "") == "RETRIEVE":
        docs = retriever.get_relevant_documents(query)
        # Convert to dicts for easier serialization
        docs_dicts = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in docs
        ]
        state["documents"] = docs_dicts
        state["metadata"]["retrieval"] = f"Retrieved {len(docs)} documents"
    else:
        state["documents"] = []
        state["metadata"]["retrieval"] = "No documents retrieved (based on query analysis)"
    
    return state

def assess_relevance(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assess the relevance of retrieved documents.
    
    Args:
        state: Current state
        
    Returns:
        State with relevant document indices
    """
    query = state["query"]
    documents = state.get("documents", [])
    
    # Skip if no documents
    if not documents:
        state["relevant_docs_indices"] = []
        state["metadata"]["relevance"] = "No documents to assess"
        return state
    
    llm = get_llm()
    
    # Format documents for prompt
    formatted_docs = format_documents(documents)
    
    # Assess relevance
    prompt = RELEVANCE_ASSESSMENT_PROMPT.format(
        query=query,
        documents=formatted_docs
    )
    
    response = llm.invoke(prompt).strip()
    
    # Process response
    if response == "NONE":
        relevant_indices = []
    else:
        # Parse comma-separated list of indices
        try:
            relevant_indices = [int(idx.strip()) for idx in response.split(",")]
        except Exception as e:
            logger.warning(f"Error parsing relevance indices: {e}. Response: {response}")
            relevant_indices = []
    
    # Update state
    state["relevant_docs_indices"] = relevant_indices
    state["metadata"]["relevance"] = f"Relevant documents: {relevant_indices}"
    
    return state

def generate_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate the final response.
    
    Args:
        state: Current state
        
    Returns:
        State with response
    """
    query = state["query"]
    documents = state.get("documents", [])
    relevant_indices = state.get("relevant_docs_indices", [])
    
    llm = get_llm()
    
    # Filter relevant documents if available
    if documents and relevant_indices:
        relevant_docs = [documents[i] for i in relevant_indices if i < len(documents)]
        formatted_docs = format_documents(relevant_docs)
        documents_info = f"Relevant Documents:\n{formatted_docs}"
    elif documents:
        documents_info = "Documents were retrieved but none were deemed relevant to your query."
    else:
        documents_info = "No documents were retrieved for this query."
    
    # Generate response
    prompt = RESPONSE_GENERATION_PROMPT.format(
        query=query,
        documents_info=documents_info
    )
    
    response = llm.invoke(prompt).strip()
    
    # Update state
    state["response"] = response
    state["metadata"]["response_generation"] = "Response generated"
    
    return state

def get_retriever_node(retriever: BaseRetriever) -> Callable:
    """Get a retriever node function configured with a specific retriever."""
    def retriever_node(state: Dict[str, Any]) -> Dict[str, Any]:
        return retrieve_documents(state, retriever)
    return retriever_node 