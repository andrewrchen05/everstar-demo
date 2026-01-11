"""
Gemini API client for making LLM calls.
"""
import os
from typing import List, Optional, TYPE_CHECKING
import google.generativeai as genai

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, will use environment variables only

if TYPE_CHECKING:
    from agent import Message, Role


class GeminiClient:
    """
    Client for interacting with Google's Gemini API.
    Handles API initialization and message generation.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
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
            system_prompt: Optional system prompt to include.
            tools_description: Optional description of available tools to include in the prompt.
            
        Returns:
            Raw text response from Gemini that should be parsed deterministically.
        """
        # Import here to avoid circular dependency
        from agent import Role
        
        # Convert conversation history to Gemini format
        chat_history = []
        system_content = system_prompt
        
        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_content = msg.content
            elif msg.role == Role.USER:
                chat_history.append({"role": "user", "parts": [msg.content]})
            elif msg.role == Role.ASSISTANT:
                chat_history.append({"role": "model", "parts": [msg.content]})
        
        # Start a chat session with history
        chat = self.model.start_chat(history=chat_history)
        
        # Get the last user message for the current prompt
        user_messages = [msg for msg in messages if msg.role == Role.USER]
        if not user_messages:
            raise ValueError("No user messages provided")
        
        last_user_message = user_messages[-1].content
        
        # Build the prompt with system prompt and tools description if provided
        prompt_parts = []
        
        if system_content:
            prompt_parts.append(system_content)
        
        if tools_description:
            prompt_parts.append(f"\n\nAvailable tools:\n{tools_description}")
        
        # If we have system prompt or tools, include them in the message
        # Otherwise just send the user message
        if prompt_parts:
            full_prompt = "\n".join(prompt_parts) + f"\n\nUser: {last_user_message}"
            response = chat.send_message(full_prompt)
        else:
            response = chat.send_message(last_user_message)
        
        return response.text
    
