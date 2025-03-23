"""
Prompts for the Self-RAG system.
"""

from langchain.prompts import PromptTemplate

# Query analyzer prompt
QUERY_ANALYSIS_PROMPT = PromptTemplate.from_template(
    """You are a query analyzer for a retrieval system.
    Your task is to analyze the user query and determine if it requires information retrieval.
    
    User Query: {query}
    
    Based on the query, determine if it requires retrieving factual information from documents.
    
    Output should be either:
    "RETRIEVE" - If the query requires factual information from documents.
    "NO_RETRIEVE" - If the query is conversational, subjective, or doesn't need factual lookup.
    
    Decision:"""
)

# Document relevance assessment prompt
RELEVANCE_ASSESSMENT_PROMPT = PromptTemplate.from_template(
    """You are a document relevance assessor.
    Your task is to assess whether the retrieved documents are relevant to the query.
    
    Query: {query}
    
    Retrieved Documents:
    {documents}
    
    For each document, assess if it is relevant to answering the query.
    
    Output should be a comma-separated list of indices of relevant documents (starting from 0).
    If none are relevant, respond with "NONE".
    
    Relevant Documents:"""
)

# Self-RAG response generation prompt
RESPONSE_GENERATION_PROMPT = PromptTemplate.from_template(
    """You are a helpful assistant. Your task is to answer the user's query.
    
    Query: {query}
    
    {documents_info}
    
    Generate a comprehensive response answering the user's query. Follow these guidelines:
    1. If relevant documents were provided, use them as the basis of your response.
    2. Cite specific documents using [Doc X] notation where X is the document number.
    3. If no documents were provided or needed, rely on your own knowledge.
    4. Be truthful and do not make up information.
    5. If you don't know the answer, say so clearly.
    
    Response:"""
) 