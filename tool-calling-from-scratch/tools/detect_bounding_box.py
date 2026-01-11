from typing import Any
from tool import ToolUse

class DetectBoundingBox:
    """Tool for detecting bounding boxes around items in images."""
    
    def __init__(self):
        self.name = "detect_bounding_box"
        self.description = "Detects and returns bounding boxes around specific items in images"
        self.parameters = {}
    
    def get_prompt(self) -> str:
        """
        Returns the prompt description for this tool to be added to the system prompt.
        
        Returns:
            A string describing the tool's purpose and usage
        """
        return f"""Tool: {self.name}
Description: {self.description}
This tool can be used to detect and locate specific items in images by returning bounding box coordinates."""
    
    def execute(self, tool_use: ToolUse) -> Any:
        """
        Execute the detect_bounding_box tool.
        
        Args:
            tool_use: ToolUse object containing the tool name and parameters
            
        Returns:
            The result of the bounding box detection
        """
        if tool_use.name != self.name:
            raise ValueError(f"Tool name mismatch: expected {self.name}, got {tool_use.name}")
        
        print("generating bounding box")
        return "Bounding box detection completed (dummy implementation)"
