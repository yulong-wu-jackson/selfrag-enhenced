"""
Main module to demonstrate the Self-RAG system.
"""
import os
import logging
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain.vectorstores.base import VectorStore
from langchain.schema import BaseRetriever

from src.config.config import get_config
from src.embeddings.embedding import get_embeddings
from src.agents.self_rag import SelfRAG
from src.utils.langsmith import setup_langsmith

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_vectorstore() -> VectorStore:
    """
    Create a sample vector store with some documents using Chroma DB.
    In a real application, this would load and index your actual documents.
    """
    # Sample documents
    documents = [
        Document(
            page_content="Python is a high-level, interpreted programming language known for its readability and versatility.",
            metadata={"source": "programming_languages.txt", "topic": "Python"}
        ),
        Document(
            page_content="JavaScript is a scripting language that enables interactive web pages and is an essential part of web applications.",
            metadata={"source": "programming_languages.txt", "topic": "JavaScript"}
        ),
        Document(
            page_content="Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.",
            metadata={"source": "ai_concepts.txt", "topic": "Machine Learning"}
        ),
        Document(
            page_content="Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.",
            metadata={"source": "ai_concepts.txt", "topic": "NLP"}
        ),
        Document(
            page_content="Self-RAG (Retrieval Augmented Generation) is an approach where the model decides when to retrieve information and which retrieved information to use.",
            metadata={"source": "llm_techniques.txt", "topic": "Self-RAG"}
        ),
    ]
    
    # Create vector store with Chroma
    embeddings = get_embeddings()
    # Set a directory for persistence
    persist_directory = "./chroma_db"
    
    # Create or load the Chroma vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="self-rag-collection",
    )
    
    # Persist to disk
    vectorstore.persist()
    logger.info(f"Vector store created with {len(documents)} documents and persisted to {persist_directory}")
    
    return vectorstore

def main():
    """Main function to run the Self-RAG demo."""
    config = get_config()
    
    # Setup LangSmith
    langsmith_client = setup_langsmith()
    if langsmith_client:
        logger.info("LangSmith tracing enabled")
    
    # Create a sample vector store and retriever
    vectorstore = create_sample_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Create the Self-RAG agent
    self_rag = SelfRAG(retriever)
    
    # Example queries
    queries = [
        "What is Python programming language?",
        "How are you doing today?",
        "Explain the concept of Self-RAG",
        "Tell me a joke",
    ]
    
    # Process each query
    for query in queries:
        logger.info(f"\n\n========== QUERY: {query} ==========")
        
        # Get response
        response = self_rag.get_response(query)
        
        # Print response
        logger.info(f"RESPONSE: {response}")
        print(f"\nQUERY: {query}")
        print(f"RESPONSE: {response}\n")

if __name__ == "__main__":
    main() 