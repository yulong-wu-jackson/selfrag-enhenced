"""
Simple test to demonstrate the Self-RAG system.
"""
import os
import logging
import shutil
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
    # Set a directory for persistence
    persist_directory = "./chroma_db"
    
    # Check if vector store already exists and clear it if it does
    if os.path.exists(persist_directory):
        logger.info(f"Existing vector store found at {persist_directory}. Clearing it...")
        try:
            shutil.rmtree(persist_directory)
            logger.info(f"Successfully cleared existing vector store at {persist_directory}")
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
    
    # Sample documents
    documents = [
        Document(
            page_content="Jackson is a 3rd year uoft student",
            metadata={"source": "personal_data.txt", "topic": "profile"}
        ),
    ]
    
    # Create vector store with Chroma
    embeddings = get_embeddings()
    
    # Create or load the Chroma vector store
    # Note: Chroma automatically persists docs since v0.4.x
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="self-rag-collection",
    )
    
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
        "Tell me a joke",
        "Who is Jackson?"
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