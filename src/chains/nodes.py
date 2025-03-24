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
    QUERY_REWRITING_PROMPT,
    RESPONSE_EVALUATION_PROMPT,
    ADDITIONAL_INFO_PROMPT,
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
    
    # Clean response to ensure it's only the expected value
    if "RETRIEVE" in response:
        clean_response = "RETRIEVE"
    else:
        clean_response = "NO_RETRIEVE"
    
    # Update state
    state["retrieve_decision"] = clean_response
    state["metadata"] = {
        "query_analysis": clean_response,
        "raw_response": response
    }
    
    return state

def rewrite_query(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rewrite the query to make it better for knowledge base searching and generate key sentences.
    Only runs if retrieval is needed.
    
    Args:
        state: Current state including the query and retrieve_decision
        
    Returns:
        Updated state with rewritten query and key sentences
    """
    # Skip if the decision was not to retrieve
    if state.get("retrieve_decision", "") != "RETRIEVE":
        return state
    
    query = state["query"]
    llm = get_llm()
    
    # Rewrite query
    prompt = QUERY_REWRITING_PROMPT.format(query=query)
    response = llm.invoke(prompt).strip()
    
    # Parse the response
    lines = response.split("\n")
    rewritten_query = ""
    key_sentences = []
    
    current_section = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if line.startswith("REWRITTEN QUERY:"):
            current_section = "query"
            continue
        elif line.startswith("KEY SENTENCES:"):
            current_section = "key_sentences"
            continue
        
        if current_section == "query":
            rewritten_query = line
            current_section = None
        elif current_section == "key_sentences" and line.startswith("-"):
            key_sentences.append(line[1:].strip())
    
    # Update state
    state["original_query"] = query
    state["rewritten_query"] = rewritten_query if rewritten_query else query
    state["search_key_sentences"] = key_sentences
    state["metadata"]["query_rewriting"] = {
        "rewritten_query": rewritten_query,
        "key_sentences": key_sentences
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
    # Only retrieve if the decision was to retrieve
    if state.get("retrieve_decision", "") == "RETRIEVE":
        # Get the rewritten query and key sentences
        query = state.get("rewritten_query", state["query"])
        key_sentences = state.get("search_key_sentences", [])
        
        all_docs = []
        
        # Search with the main rewritten query
        main_docs = retriever.get_relevant_documents(query)
        all_docs.extend(main_docs)
        
        # Search with each key sentence
        for sentence in key_sentences:
            sentence_docs = retriever.get_relevant_documents(sentence)
            all_docs.extend(sentence_docs)
        
        # Remove duplicates (based on page_content)
        unique_docs = []
        seen_contents = set()
        for doc in all_docs:
            if doc.page_content not in seen_contents:
                unique_docs.append(doc)
                seen_contents.add(doc.page_content)
        
        # Convert to dicts for easier serialization
        docs_dicts = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in unique_docs
        ]
        
        state["documents"] = docs_dicts
        state["metadata"]["retrieval"] = f"Retrieved {len(unique_docs)} unique documents from {len(all_docs)} total searches"
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
    
    # Process response with enhanced error handling
    if "NONE" in response:
        relevant_indices = []
    else:
        # Extract all digit sequences that might be indices
        import re
        # Find all numbers in the response, handling various formats
        index_matches = re.findall(r'\b\d+\b', response)
        try:
            relevant_indices = [int(idx) for idx in index_matches]
            # Filter out invalid indices
            relevant_indices = [idx for idx in relevant_indices if 0 <= idx < len(documents)]
        except Exception as e:
            logger.warning(f"Error parsing relevance indices: {e}. Response: {response}")
            relevant_indices = []
    
    # Update state
    state["relevant_docs_indices"] = relevant_indices
    state["metadata"]["relevance"] = {
        "relevant_indices": relevant_indices,
        "raw_response": response
    }
    
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

def evaluate_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate if the generated response fully answers the query and has no hallucinations.
    
    Args:
        state: Current state with generated response
        
    Returns:
        State with evaluation decision
    """
    original_query = state.get("original_query", state["query"])
    response = state.get("response", "")
    documents = state.get("documents", [])
    
    # If there's no response to evaluate, mark as needing improvement
    if not response:
        state["response_evaluation"] = "NEEDS_IMPROVEMENT"
        state["metadata"]["response_evaluation"] = "No response to evaluate"
        return state
    
    llm = get_llm()
    
    # Format documents for fact-checking
    formatted_docs = format_documents(documents) if documents else "No documents were retrieved."
    
    # Evaluate response
    prompt = RESPONSE_EVALUATION_PROMPT.format(
        original_query=original_query,
        response=response,
        documents=formatted_docs
    )
    
    raw_evaluation = llm.invoke(prompt).strip()
    
    # Clean response to ensure it's only the expected value
    if "SATISFACTORY" in raw_evaluation:
        evaluation = "SATISFACTORY"
    else:
        evaluation = "NEEDS_IMPROVEMENT"
    
    # Update state
    state["response_evaluation"] = evaluation
    state["metadata"]["response_evaluation"] = {
        "decision": evaluation,
        "raw_response": raw_evaluation
    }
    state["loop_count"] = state.get("loop_count", 0) + 1
    
    return state

def identify_additional_info(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Identify what additional information is needed to improve the response.
    
    Args:
        state: Current state with evaluated response
        
    Returns:
        State with new search terms
    """
    original_query = state.get("original_query", state["query"])
    response = state.get("response", "")
    documents = state.get("documents", [])
    
    llm = get_llm()
    
    # Format documents
    formatted_docs = format_documents(documents) if documents else "No documents were retrieved."
    
    # Identify gaps and generate new search terms
    prompt = ADDITIONAL_INFO_PROMPT.format(
        original_query=original_query,
        response=response,
        documents=formatted_docs
    )
    
    additional_info = llm.invoke(prompt).strip()
    
    # Parse the response to extract new key sentences
    new_key_sentences = []
    missing_info = ""
    hallucinations = ""
    
    current_section = None
    for line in additional_info.split("\n"):
        line = line.strip()
        if not line:
            continue
            
        if line.startswith("MISSING INFORMATION:"):
            current_section = "missing"
            continue
        elif line.startswith("POTENTIAL HALLUCINATIONS:"):
            current_section = "hallucinations"
            continue
        elif line.startswith("NEW KEY SENTENCES:"):
            current_section = "key_sentences"
            continue
            
        if current_section == "missing":
            missing_info += line + " "
        elif current_section == "hallucinations":
            hallucinations += line + " "
        elif current_section == "key_sentences" and line.startswith("-"):
            new_key_sentences.append(line[1:].strip())
    
    # Update the existing key sentences with the new ones
    current_key_sentences = state.get("search_key_sentences", [])
    state["search_key_sentences"] = current_key_sentences + new_key_sentences
    
    # Update state with analysis
    state["metadata"]["additional_info"] = {
        "missing_information": missing_info.strip(),
        "potential_hallucinations": hallucinations.strip(),
        "new_key_sentences": new_key_sentences
    }
    
    return state

def get_retriever_node(retriever: BaseRetriever) -> Callable:
    """Get a retriever node function configured with a specific retriever."""
    def retriever_node(state: Dict[str, Any]) -> Dict[str, Any]:
        return retrieve_documents(state, retriever)
    return retriever_node 