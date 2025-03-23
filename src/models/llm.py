"""
LLM model configuration for xAI's Grok-2.
"""
import requests
from typing import Dict, List, Any, Optional, Union
import json
import logging

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.pydantic_v1 import Field

from src.config.config import get_config

logger = logging.getLogger(__name__)
config = get_config()

class GrokLLM(LLM):
    """Custom LLM for xAI's Grok-2 API."""
    
    api_key: str = Field(default_factory=lambda: config.grok_api_key)
    api_base: str = "https://api.xai.com/v1"  # xAI's API base URL
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
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Error calling xAI Grok-2 API: {e}")
            return f"Error: {str(e)}"
    
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