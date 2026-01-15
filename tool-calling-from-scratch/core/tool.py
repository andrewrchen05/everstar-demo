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
    def __init__(self, name: str, description: str, function: Optional[Callable] = None, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize a Tool.
        
        Args:
            name: Name of the tool
            description: Description of what the tool does
            function: Optional callable function. If provided, execute() will call this function.
                     If None, subclasses should override execute() to provide custom behavior.
            parameters: Optional dictionary of parameter definitions
        """
        self.name = name
        self.description = description
        self.function = function
        self.parameters = parameters or {}
    
    def __repr__(self):
        return f"Tool(name={self.name}, description={self.description})"
    
    def execute(self, tool_use: ToolUse) -> Any:
        """
        Execute the tool with the given ToolUse object.
        
        Subclasses can override this method to provide custom execution logic.
        If a function was provided in __init__, this will call that function.
        Otherwise, subclasses must override this method.
        
        Args:
            tool_use: ToolUse object containing the tool name and parameters
            
        Returns:
            The result of executing the tool
            
        Raises:
            ValueError: If the tool name doesn't match, required parameters are missing,
                       or no function is provided and execute() is not overridden
        """
        if tool_use.name != self.name:
            raise ValueError(f"Tool name mismatch: expected {self.name}, got {tool_use.name}")
        
        if self.function is None:
            raise NotImplementedError(
                f"Tool '{self.name}' has no function provided and execute() was not overridden. "
                "Either provide a function in __init__ or override execute() in a subclass."
            )
        
        # Convert string params to appropriate types if needed
        # For now, we'll pass them as-is and let the function handle type conversion
        # If partial is True, we might want to handle missing parameters differently
        if tool_use.partial:
            # For partial tool uses, we might want to return a partial result or handle differently
            # This depends on the specific use case
            pass
        
        # Execute the function with the provided parameters
        return self.function(**tool_use.params)
