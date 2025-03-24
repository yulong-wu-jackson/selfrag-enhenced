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
    
    You should only reply RETRIEVE or NO_RETRIEVE, no other explaination needed.
    
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

# Query rewriting prompt for enhanced search
QUERY_REWRITING_PROMPT = PromptTemplate.from_template(
    """You are a query optimization expert for knowledge retrieval systems.
    Your task is to analyze and rewrite a user query to make it more effective for searching a knowledge base,
    and to identify key sentences or search terms that will help retrieve relevant information.
    
    User Query: {query}
    
    Please provide:
    1. A rewritten version of the query that is clearer, more specific, and better formatted for knowledge base searching
    2. 1-5 key sentences or phrases that represent different aspects of the query or related search terms
    
    Format your response exactly as follows:
    
    REWRITTEN QUERY:
    <the rewritten query>
    
    KEY SENTENCES:
    - <key sentence 1>
    - <key sentence 2>
    - <key sentence 3>
    - <key sentence 4>
    - <key sentence 5>
    
    Make sure the rewritten query and key sentences are semantically different enough to capture various aspects of the original query.
    This will help retrieve a diverse set of relevant documents from the knowledge base.
    """
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