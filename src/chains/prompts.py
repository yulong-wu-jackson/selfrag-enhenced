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
    
    IMPORTANT: You must ONLY output EXACTLY "RETRIEVE" or "NO_RETRIEVE". No other text, explanations, or punctuation.
    
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
    
    IMPORTANT:
    - Output ONLY a comma-separated list of indices of relevant documents (starting from 0).
    - If none are relevant, respond with EXACTLY "NONE".
    - Do NOT include any explanations, additional text, or punctuation.
    - Examples of valid outputs: "0", "0,1,2", "NONE", "1,3"
    
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
    2. 1-5 key sentences or phrases that represent different aspects of the query or related search terms (you should decide on how many key sentence to generate if it is simple task then you only need 1 key sentence)
    
    IMPORTANT: Format your response EXACTLY as follows with NO additional text:
    
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

# Response evaluation prompt
RESPONSE_EVALUATION_PROMPT = PromptTemplate.from_template(
    """You are a response evaluator for a retrieval-augmented system.
    Your task is to evaluate whether the generated response fully answers the original query and contains no hallucinations.
    
    Original Query: {original_query}
    
    Generated Response: 
    {response}
    
    Retrieved Documents (for fact checking):
    {documents}
    
    Evaluate the response on two criteria:
    1. Completeness: Does it fully answer all aspects of the query?
    2. Factual accuracy: Is all information grounded in the provided documents or common knowledge without hallucinations?
    
    IMPORTANT:
    - You MUST output ONLY "SATISFACTORY" or "NEEDS_IMPROVEMENT" with NO additional text
    - "SATISFACTORY" - If the response completely answers the query and contains no hallucinations
    - "NEEDS_IMPROVEMENT" - If the response is incomplete or contains potential hallucinations
    - Do NOT include any explanations, reasoning, or punctuation
    
    Decision:"""
)

# Additional information prompt
ADDITIONAL_INFO_PROMPT = PromptTemplate.from_template(
    """You are an information gap analyzer for a retrieval system.
    The current response to a query is incomplete or potentially contains hallucinations.
    Your task is to identify what additional information is needed to improve the response.
    
    Original Query: {original_query}
    Current Response: {response}
    Retrieved Documents So Far: 
    {documents}
    
    Based on the above, determine:
    1. What specific information is missing from the current response
    2. What potential hallucinations exist in the current response
    3. What additional search queries would retrieve the information needed to fix these issues
    
    IMPORTANT: Format your response EXACTLY as follows with NO additional explanation text:
    
    MISSING INFORMATION:
    <brief description of what's missing>
    
    POTENTIAL HALLUCINATIONS:
    <brief description of potential hallucinations, if any>
    
    NEW KEY SENTENCES:
    - <specific search query 1>
    - <specific search query 2>
    - <specific search query 3>
    
    Make the new key sentences specific, targeted, and different from previous searches to fill in the information gaps.
    """
) 