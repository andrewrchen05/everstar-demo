"""
Core domain models for the agent system.
"""
from typing import List, Optional, Union
from enum import Enum
from .tool import ToolUse


class Role(Enum):
    """Message role types."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ResponseType(Enum):
    """Response type enumeration."""
    TEXT = "text"
    TOOL_USE = "tool_use"


class Message:
    """Represents a message in the conversation."""
    
    def __init__(self, role: Role, content: str, image_path: Optional[str] = None):
        """
        Initialize a Message.
        
        Args:
            role: The role of the message sender (USER, ASSISTANT, or SYSTEM)
            content: The text content of the message
            image_path: Optional path to an image file to include with the message
        """
        self.role = role
        self.content = content
        self.image_path = image_path
    
    def __repr__(self):
        image_info = f", image={self.image_path}" if self.image_path else ""
        return f"Message(role={self.role.value}, content={self.content[:50]}...{image_info})"


class AssistantResponse:
    """
    Represents an assistant's response, which can be either a text response
    or one or more tool uses.
    """
    
    def __init__(
        self, 
        response_type: ResponseType,
        text: Optional[str] = None,
        tool_uses: Optional[List[ToolUse]] = None
    ):
        """
        Initialize an AssistantResponse.
        
        Args:
            response_type: The type of response (TEXT or TOOL_USE)
            text: Text content (required for TEXT type)
            tool_uses: List of ToolUse objects (required for TOOL_USE type)
        """
        self.response_type = response_type
        
        if response_type == ResponseType.TEXT:
            if text is None:
                raise ValueError("text is required for TEXT response type")
            self.text = text
            self.tool_uses = None
        elif response_type == ResponseType.TOOL_USE:
            if tool_uses is None or len(tool_uses) == 0:
                raise ValueError("tool_uses is required for TOOL_USE response type")
            self.tool_uses = tool_uses
            self.text = None
        else:
            raise ValueError(f"Unknown response type: {response_type}")
    
    @classmethod
    def text_response(cls, text: str) -> "AssistantResponse":
        """Create a text response."""
        return cls(response_type=ResponseType.TEXT, text=text)
    
    @classmethod
    def tool_use_response(cls, tool_uses: Union[ToolUse, List[ToolUse]]) -> "AssistantResponse":
        """Create a tool use response."""
        if isinstance(tool_uses, ToolUse):
            tool_uses = [tool_uses]
        return cls(response_type=ResponseType.TOOL_USE, tool_uses=tool_uses)
    
    def is_text(self) -> bool:
        """Check if this is a text response."""
        return self.response_type == ResponseType.TEXT
    
    def is_tool_use(self) -> bool:
        """Check if this is a tool use response."""
        return self.response_type == ResponseType.TOOL_USE
    
    def __repr__(self):
        if self.is_text():
            return f"AssistantResponse(type=TEXT, text={self.text[:50]}...)"
        else:
            return f"AssistantResponse(type=TOOL_USE, tool_uses={len(self.tool_uses)} tools)"
