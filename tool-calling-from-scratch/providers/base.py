"""
ModelProvider interface and implementations for different LLM providers.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.models import Message


class ModelProvider(ABC):
    """
    Abstract base class for LLM model providers.
    All model providers must implement the generate_response method.
    """
    
    @abstractmethod
    def generate_response(
        self,
        messages: List["Message"],
        system_prompt: Optional[str] = None,
        tools_description: Optional[str] = None
    ) -> str:
        """
        Generate a response from the LLM based on conversation history.
        
        Args:
            messages: List of Message objects representing the conversation history.
            system_prompt: Optional system prompt to include.
            tools_description: Optional description of available tools to include in the prompt.
            
        Returns:
            Raw text response from the LLM that should be parsed deterministically.
        """
        pass
