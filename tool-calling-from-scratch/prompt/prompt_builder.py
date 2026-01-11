from typing import List, Any
from .system_prompt import SYSTEM_PROMPT


class PromptBuilder:
    """Class for building system prompts with tool descriptions."""
    
    def __init__(self, base_prompt: str = SYSTEM_PROMPT):
        """
        Initialize the PromptBuilder.
        
        Args:
            base_prompt: The base system prompt to use
        """
        self.base_prompt = base_prompt
    
    def build_prompt(self, tools: List[Any]) -> str:
        """
        Build a system prompt by inserting tool descriptions into the base prompt.
        
        Args:
            tools: List of tool objects that have a get_prompt() method
            
        Returns:
            A complete system prompt with tool descriptions included
        """
        if not tools:
            return self.base_prompt
        
        # Collect tool prompts
        tool_descriptions = []
        for tool in tools:
            if hasattr(tool, 'get_prompt'):
                tool_descriptions.append(tool.get_prompt())
        
        if not tool_descriptions:
            return self.base_prompt
        
        # Combine base prompt with tool descriptions
        tools_section = "\n\nAvailable Tools:\n" + "\n\n".join(tool_descriptions)
        return self.base_prompt + tools_section
