"""
Gemini API client and model provider implementation.
"""
import os
from typing import List, Optional, TYPE_CHECKING
import google.generativeai as genai
from PIL import Image

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, will use environment variables only

if TYPE_CHECKING:
    from core.models import Message, Role

from .base import ModelProvider


class GeminiClient:
    """
    Client for interacting with Google's Gemini API.
    Handles API initialization and message generation.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-3-flash-preview"):
        """
        Initialize the Gemini client.
        
        Args:
            api_key: Gemini API key. If not provided, will try to get from GEMINI_API_KEY env var.
            model_name: Name of the Gemini model to use.
        """
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be provided either as parameter or environment variable")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    
    def generate_response(
        self, 
        messages: List["Message"], 
        system_prompt: Optional[str] = None,
        tools_description: Optional[str] = None
    ) -> str:
        """
        Generate a response from Gemini based on conversation history.
        
        Args:
            messages: List of Message objects representing the conversation history.
                Messages can optionally include images via the image_path attribute.
            system_prompt: Optional system prompt to include.
            tools_description: Optional description of available tools to include in the prompt.
            
        Returns:
            Raw text response from Gemini that should be parsed deterministically.
        """
        # Import here to avoid circular dependency
        from core.models import Role
        
        # Convert conversation history to Gemini format
        chat_history = []
        system_content = system_prompt
        
        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_content = msg.content
            elif msg.role == Role.USER:
                parts = [msg.content]
                # Add image if provided
                if hasattr(msg, 'image_path') and msg.image_path:
                    try:
                        image = Image.open(msg.image_path)
                        parts.append(image)
                    except Exception as e:
                        # If image loading fails, continue without image
                        print("Image failed to load: ", e)
                        pass
                chat_history.append({"role": "user", "parts": parts})
            elif msg.role == Role.ASSISTANT:
                chat_history.append({"role": "model", "parts": [msg.content]})
        
        # Start a chat session with history
        chat = self.model.start_chat(history=chat_history)
        
        # Determine what to send as the current prompt
        # If the last message is an assistant message (tool results), we want to continue from there
        # Otherwise, use the last user message
        last_message = messages[-1] if messages else None
        if not last_message:
            raise ValueError("No messages provided")
        
        # If the last message is an assistant message (tool results), use it as the prompt
        # This enables automatic tool chaining
        if last_message.role == Role.ASSISTANT:
            prompt_content = last_message.content
            # For assistant messages (tool results), we don't have an image_path
            message_image = None
        else:
            # Use the last user message
            user_messages = [msg for msg in messages if msg.role == Role.USER]
            if not user_messages:
                raise ValueError("No user messages provided")
            
            last_user_message = user_messages[-1]
            prompt_content = last_user_message.content
            # Check for image in the last user message
            message_image = None
            if hasattr(last_user_message, 'image_path') and last_user_message.image_path:
                try:
                    message_image = Image.open(last_user_message.image_path)
                except Exception as e:
                    # If image loading fails, continue without image
                    print("Image failed to load: ", e)
                    pass
        
        # Build the prompt with system prompt and tools description if provided
        prompt_parts = []
        
        if system_content:
            prompt_parts.append(system_content)
        
        if tools_description:
            prompt_parts.append(f"\n\nAvailable tools:\n{tools_description}")
        
        # Prepare message parts (text and optionally image)
        message_parts = []
        
        # If we have system prompt or tools, include them in the message
        # Otherwise just send the prompt content
        if prompt_parts:
            full_prompt = "\n".join(prompt_parts) + f"\n\nUser: {prompt_content}"
            message_parts.append(full_prompt)
        else:
            message_parts.append(prompt_content)
        
        # Add image if available
        if message_image:
            message_parts.append(message_image)
        
        response = chat.send_message(message_parts)
        
        return response.text


class GeminiModelProvider(ModelProvider):
    """
    ModelProvider implementation for Google's Gemini API.
    Wraps the GeminiClient class.
    """
    
    def __init__(self, model_name: str = "gemini-3-flash-preview"):
        """
        Initialize the Gemini model provider.
        
        Args:
            model_name: Name of the Gemini model to use.
            
        Note:
            API key will be automatically fetched from GEMINI_API_KEY environment variable
            by the underlying GeminiClient.
        """
        # Pass None for api_key to let GeminiClient fetch from environment
        self.client = GeminiClient(api_key=None, model_name=model_name)
    
    def generate_response(
        self,
        messages: List["Message"],
        system_prompt: Optional[str] = None,
        tools_description: Optional[str] = None
    ) -> str:
        """
        Generate a response using Gemini.
        
        Args:
            messages: List of Message objects representing the conversation history.
            system_prompt: Optional system prompt to include.
            tools_description: Optional description of available tools to include in the prompt.
            
        Returns:
            Raw text response from Gemini.
        """
        return self.client.generate_response(
            messages=messages,
            system_prompt=system_prompt,
            tools_description=tools_description
        )
