"""
Factory function for creating ModelProvider instances.
"""
from .base import ModelProvider
from .gemini import GeminiModelProvider


def create_model_provider(client: str, **kwargs) -> ModelProvider:
    """
    Factory function to create a ModelProvider instance based on the client name.
    
    Args:
        client: Name of the client to use (e.g., 'gemini')
        **kwargs: Additional provider-specific arguments (e.g., model_name for Gemini)
        
    Returns:
        ModelProvider instance
        
    Raises:
        ValueError: If the client name is not supported
        
    Note:
        Each provider will automatically fetch its API key from the appropriate
        environment variable (e.g., GEMINI_API_KEY for Gemini).
    """
    if client.lower() == 'gemini':
        model_name = kwargs.get('model_name', 'gemini-2.5-flash')
        return GeminiModelProvider(model_name=model_name)
    else:
        raise ValueError(f"Unsupported client: {client}. Supported clients: 'gemini'")
