"""
Self-RAG Agent implementation using LangGraph.
"""
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple, TypedDict, Annotated

from langchain.schema import BaseRetriever
from langchain.schema.runnable import Runnable
from langchain_core.runnables import RunnableConfig
import langgraph.graph as lg
from langgraph.graph import StateGraph, END

from src.config.config import get_config
from src.chains.nodes import (
    analyze_query,
    get_retriever_node,
    assess_relevance,
    generate_response,
    rewrite_query,
    evaluate_response,
    identify_additional_info,
)
from src.utils.helpers import convert_to_langsmith_metadata

logger = logging.getLogger(__name__)
app_config = get_config()  # Renamed to avoid conflict with the config parameter

# Define the state type for the graph
class GraphState(TypedDict):
    query: str
    original_query: Optional[str]  # Original query before rewriting
    rewritten_query: Optional[str]  # Rewritten query for better search
    search_key_sentences: Optional[List[str]]  # Key sentences for multiple searches
    retrieve_decision: Optional[str]
    documents: Optional[List[Dict[str, Any]]]
    relevant_docs_indices: Optional[List[int]]
    response: Optional[str]
    response_evaluation: Optional[str]  # Evaluation of response quality
    loop_count: Optional[int]  # Counter to track number of retrieval loops
    metadata: Optional[Dict[str, Any]]

class SelfRAG:
    """
    Self-RAG Agent class.
    
    Implements a Self-RAG system using LangGraph, which:
    1. Analyzes if a query requires retrieval
    2. Conditionally retrieves documents
    3. Assesses document relevance 
    4. Generates a final response
    """
    
    def __init__(self, retriever: BaseRetriever):
        """
        Initialize the Self-RAG agent.
        
        Args:
            retriever: Document retriever to use
        """
        self.retriever = retriever
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph for Self-RAG.
        
        Returns:
            StateGraph: The constructed graph
        """
        # Define the graph with the GraphState type
        builder = StateGraph(state_schema=GraphState)
        
        # Add nodes
        builder.add_node("analyze_query", analyze_query)
        builder.add_node("rewrite_query", rewrite_query)
        builder.add_node("retrieve", get_retriever_node(self.retriever))
        builder.add_node("assess_relevance", assess_relevance)
        builder.add_node("generate_response", generate_response)
        builder.add_node("evaluate_response", evaluate_response)
        builder.add_node("identify_additional_info", identify_additional_info)
        
        # Define evaluation router for conditional branching
        def evaluation_router(state: GraphState) -> str:
            # Check if max iterations reached (prevent infinite loops)
            if state.get("loop_count", 0) >= 3:  # Cap at 3 iterations
                return "END"
                
            # Route based on response evaluation
            evaluation = state.get("response_evaluation", "")
            if evaluation == "SATISFACTORY":
                return "END"
            else:
                return "identify_additional_info"
        
        # Define edges
        builder.add_edge("analyze_query", "rewrite_query")
        builder.add_edge("rewrite_query", "retrieve")
        builder.add_edge("retrieve", "assess_relevance")
        builder.add_edge("assess_relevance", "generate_response")
        builder.add_edge("generate_response", "evaluate_response")
        
        # Add conditional routing after evaluation
        builder.add_conditional_edges(
            "evaluate_response",
            evaluation_router,
            {
                "END": END,
                "identify_additional_info": "identify_additional_info"
            }
        )
        
        # Complete the loop back to retrieve
        builder.add_edge("identify_additional_info", "retrieve")
        
        # Set entry point
        builder.set_entry_point("analyze_query")
        
        # Compile the graph
        return builder.compile()
    
    def invoke(self, query: str, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """
        Run the Self-RAG agent on a query.
        
        Args:
            query: User query
            config: Optional runnable config
        
        Returns:
            Dict with response and other state information
        """
        # Prepare input state
        state = {"query": query}
        
        # Set up config for LangSmith if tracing is enabled
        if config is None:
            run_config = {"metadata": {"query": query}}
            if app_config.langchain_tracing_v2:
                run_config["callbacks"] = []  # LangSmith callbacks will be added automatically
        else:
            run_config = config
        
        # Execute the graph
        result = self.graph.invoke(state, config=run_config)
        
        return result
    
    def get_response(self, query: str) -> str:
        """
        Get just the response text from the Self-RAG agent.
        
        Args:
            query: User query
        
        Returns:
            Response text
        """
        result = self.invoke(query)
        return result.get("response", "No response generated.") 