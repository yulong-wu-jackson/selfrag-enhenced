"""
LLM model configuration for xAI's Grok-2.
"""
import requests
from typing import Dict, List, Any, Optional, Union
import json
import logging

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from pydantic import Field

from src.config.config import get_config

logger = logging.getLogger(__name__)
config = get_config()

class GrokLLM(LLM):
    """Custom LLM for xAI's Grok-2 API."""
    
    api_key: str = Field(default_factory=lambda: config.grok_api_key)
    api_base: str = "https://api.x.ai/v1"  # xAI's API base URL
    model_name: str = "grok-2-latest"
    temperature: float = 0.7
    max_tokens: int = 4096
    
    @property
    def _llm_type(self) -> str:
        return "grok-llm"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the xAI Grok-2 API."""
        # Check if we're running with a real API key
        if not self.api_key or self.api_key == "your_grok_api_key":
            logger.warning("Using mock response as no valid API key was provided")
            # Return a mock response for testing purposes
            return self._get_mock_response(prompt)
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        if stop:
            data["stop"] = stop
            
        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=data,
            )
            response.raise_for_status()
            result = response.json()
            
            if "choices" not in result or not result["choices"]:
                raise ValueError(f"Unexpected API response structure: {result}")
                
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Error calling xAI Grok-2 API: {e}")
            # Return mock response if API call fails
            return self._get_mock_response(prompt)
            
    def _get_mock_response(self, prompt: str) -> str:
        """
        Generate a mock response for testing purposes.
        
        Args:
            prompt: The user prompt
            
        Returns:
            A static mock response
        """
        # Simple keyword-based mock responses
        if "python" in prompt.lower():
            return "Python is a high-level, interpreted programming language known for its readability, simplicity, and versatility. It supports multiple programming paradigms including procedural, object-oriented, and functional programming."
        
        elif "how are you" in prompt.lower():
            return "I'm an AI assistant, so I don't have feelings, but I'm functioning properly and ready to help you with any questions or tasks!"
        
        elif "self-rag" in prompt.lower():
            return "Self-RAG (Retrieval Augmented Generation) is an approach where the LLM itself decides when to retrieve information and evaluates which retrieved information to use. Unlike traditional RAG where retrieval is always performed, Self-RAG adds self-reflection that allows the model to determine when retrieval is necessary and assess the relevance of retrieved documents."
        
        elif "joke" in prompt.lower():
            return "Why don't scientists trust atoms? Because they make up everything!"
        
        else:
            return "I'm a mock response from the Grok-2 API simulation. To get real responses, please configure your API key in the .env file."
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

def get_llm() -> GrokLLM:
    """Get an instance of the Grok LLM."""
    return GrokLLM() 