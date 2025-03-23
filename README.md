# Self-RAG Enhanced

A Self-RAG (Retrieval Augmented Generation) implementation using LangGraph, designed to intelligently decide when to retrieve information and which retrieved information to use.

## Features

- Query analysis to determine if retrieval is needed
- Document retrieval based on query analysis
- Document relevance assessment
- Intelligent response generation
- LangSmith integration for tracing and monitoring
- xAI's Grok-2 API for LLM and OpenAI for embeddings
- Chroma DB for vector storage and retrieval

## Project Structure

```
selfrag-enhanced/
├── .env                    # Environment variables
├── README.md               # Project documentation
├── requirements.txt        # Project dependencies
├── chroma_db/              # Chroma DB persistence directory
└── src/                    # Source code
    ├── agents/             # Agent implementations
    │   └── self_rag.py     # Self-RAG agent
    ├── chains/             # LangChain and LangGraph components
    │   ├── nodes.py        # LangGraph nodes
    │   └── prompts.py      # Prompts
    ├── config/             # Configuration
    │   └── config.py       # Config loader
    ├── embeddings/         # Embedding models
    │   └── embedding.py    # OpenAI embeddings
    ├── models/             # LLM models
    │   └── llm.py          # Grok-2 LLM
    ├── utils/              # Utility functions
    │   ├── helpers.py      # Helper functions
    │   └── langsmith.py    # LangSmith setup
    └── main.py             # Main application entry point
```

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/selfrag-enhanced.git
   cd selfrag-enhanced
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   - Copy `.env.example` to `.env`
   - Add your API keys (xAI's Grok-2 and OpenAI)
   - Configure LangSmith credentials if needed

## Usage

Run the demo application:

```
python -m src.main
```

## Integrations

- **LangGraph**: For implementing the Self-RAG workflow
- **LangSmith**: For tracing and monitoring agent behavior
- **xAI's Grok-2 API**: For the LLM backend
- **OpenAI Embeddings**: For document embeddings
- **Chroma DB**: For vector storage and retrieval
