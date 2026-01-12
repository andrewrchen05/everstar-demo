from typing import Callable, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ToolUse:
    """Represents a tool use request from the agent."""
    type: str = "tool_use"
    name: str = ""
    params: Dict[str, str] = None
    partial: bool = False
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}

class Tool:
    def __init__(self, name: str, description: str, function: Callable, parameters: Optional[Dict[str, Any]] = None):
        self.name = name
        self.description = description
        self.function = function
        self.parameters = parameters or {}
    
    def __repr__(self):
        return f"Tool(name={self.name}, description={self.description})"
    
    def execute(self, tool_use: ToolUse) -> Any:
        """
        Execute the tool with the given ToolUse object.
        
        Args:
            tool_use: ToolUse object containing the tool name and parameters
            
        Returns:
            The result of executing the tool function
            
        Raises:
            ValueError: If the tool name doesn't match or required parameters are missing
        """
        if tool_use.name != self.name:
            raise ValueError(f"Tool name mismatch: expected {self.name}, got {tool_use.name}")
        
        # Convert string params to appropriate types if needed
        # For now, we'll pass them as-is and let the function handle type conversion
        # If partial is True, we might want to handle missing parameters differently
        if tool_use.partial:
            # For partial tool uses, we might want to return a partial result or handle differently
            # This depends on the specific use case
            pass
        
        # Execute the function with the provided parameters
        return self.function(**tool_use.params)
