from typing import List, Optional, Any, Union
from enum import Enum
from tool import Tool, ToolUse
from gemini_client import GeminiClient

class Role(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class ResponseType(Enum):
    TEXT = "text"
    TOOL_USE = "tool_use"

class Message:
    def __init__(self, role: Role, content: str):
        self.role = role
        self.content = content
    
    def __repr__(self):
        return f"Message(role={self.role.value}, content={self.content[:50]}...)"


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


class Agent:
    def __init__(self, tools: List[Tool] = None, system_prompt: Optional[str] = None, api_key: Optional[str] = None):
        self.tools = tools or []
        self.system_prompt = system_prompt
        self.conversation_history: List[Message] = []
        
        # Initialize Gemini client
        self.llm_client = GeminiClient(api_key=api_key)
        
        # Add system message if provided
        if system_prompt:
            self.conversation_history.append(Message(role=Role.SYSTEM, content=system_prompt))
    
    def run(self, messages: List[Message]) -> Message:
        """
        Process messages and generate a response.
        
        Args:
            messages: List of messages in the conversation
            
        Returns:
            Message: Assistant's response message
        """
        # Update conversation history with new messages
        self.conversation_history.extend(messages)
        
        # Get the last user message
        user_messages = [msg for msg in messages if msg.role == Role.USER]
        if not user_messages:
            return Message(role=Role.ASSISTANT, content="I didn't receive any user messages.")
        
        last_user_message = user_messages[-1]
        
        # Generate response (can be text or tool use)
        assistant_response = self._generate_response(last_user_message.content)

        # Check if the response contains a tool use or is a text response
        if assistant_response.is_text():
            # Convert text response to Message
            response = Message(role=Role.ASSISTANT, content=assistant_response.text)
        elif assistant_response.is_tool_use():
            # Execute all tools and collect results
            tool_results = []
            for tool_use in assistant_response.tool_uses:
                try:
                    result = self.execute_tool(tool_use)
                    tool_results.append(f"{tool_use.name}: {result}")
                except Exception as e:
                    tool_results.append(f"{tool_use.name}: Error - {str(e)}")
            
            # Format tool execution results into a response
            response_content = "Tool execution results:\n" + "\n".join(tool_results)
            response = Message(role=Role.ASSISTANT, content=response_content)
        else:
            # Fallback
            response = Message(role=Role.ASSISTANT, content="Unknown response type.")
        
        self.conversation_history.append(response)
        
        return response
    
    def _generate_response(self, user_input: str) -> AssistantResponse:
        """
        Generate a response to user input using Gemini API.
        Returns an AssistantResponse which can be either text or tool use.
        The response from Gemini is parsed deterministically to extract tool calls or text.
        """
        # Build tools description for the prompt
        tools_description = None
        if self.tools:
            tools_list = []
            for tool in self.tools:
                tool_desc = f"- {tool.name}: {tool.description}"
                if tool.parameters:
                    params_desc = ", ".join([f"{name}" for name in tool.parameters.keys()])
                    tool_desc += f" (parameters: {params_desc})"
                tools_list.append(tool_desc)
            tools_description = "\n".join(tools_list)
        
        # Build instruction for structured response format
        response_format_instruction = """
You must respond in one of two formats:

1. For a text response, respond with:
   {"type": "text", "content": "your response text here"}

2. For tool use, respond with:
   {"type": "tool_use", "tools": [{"name": "tool_name", "params": {"param1": "value1", "param2": "value2"}}]}

Always respond with valid JSON only, no additional text before or after.
"""
        
        # Combine tools description with format instruction
        if tools_description:
            tools_description = f"{tools_description}\n\n{response_format_instruction}"
        else:
            tools_description = response_format_instruction
        
        # Call Gemini client
        try:
            response_text = self.llm_client.generate_response(
                messages=self.conversation_history,
                system_prompt=self.system_prompt,
                tools_description=tools_description
            )
        except Exception as e:
            return AssistantResponse.text_response(f"Error calling Gemini API: {str(e)}")
        
        # Parse the response deterministically
        return self._parse_response(response_text)
    
    def _parse_response(self, response_text: str) -> AssistantResponse:
        """
        Parse the response from Gemini deterministically.
        Expects JSON format with either "text" or "tool_use" type.
        """
        import json
        import re
        
        # Try to extract JSON from the response (in case there's extra text)
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_match:
            # If no JSON found, treat as plain text response
            return AssistantResponse.text_response(response_text.strip())
        
        try:
            response_data = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            # If JSON parsing fails, treat as plain text
            return AssistantResponse.text_response(response_text.strip())
        
        # Parse based on response type
        response_type = response_data.get("type", "").lower()
        
        if response_type == "text":
            content = response_data.get("content", response_text)
            return AssistantResponse.text_response(content)
        elif response_type == "tool_use":
            tools_data = response_data.get("tools", [])
            if not tools_data:
                # Fallback to text if no tools specified
                return AssistantResponse.text_response(response_text.strip())
            
            tool_uses = []
            for tool_data in tools_data:
                tool_name = tool_data.get("name", "")
                tool_params = tool_data.get("params", {})
                
                if tool_name:
                    tool_use = ToolUse(name=tool_name, params=tool_params)
                    tool_uses.append(tool_use)
            
            if tool_uses:
                return AssistantResponse.tool_use_response(tool_uses)
            else:
                # Fallback to text if no valid tools
                return AssistantResponse.text_response(response_text.strip())
        else:
            # Unknown type, treat as text
            return AssistantResponse.text_response(response_text.strip())
    
    def reset(self):
        """Reset the conversation history."""
        self.conversation_history = []
        if self.system_prompt:
            self.conversation_history.append(Message(role=Role.SYSTEM, content=self.system_prompt))
    
    def execute_tool(self, tool_use: ToolUse) -> Any:
        """
        Execute a tool using a ToolUse object.
        
        Args:
            tool_use: ToolUse object containing the tool name and parameters
            
        Returns:
            The result of executing the tool
            
        Raises:
            ValueError: If the tool is not found in the agent's tool list
        """
        # Find the tool by name
        tool = next((t for t in self.tools if t.name == tool_use.name), None)
        if tool is None:
            raise ValueError(f"Tool '{tool_use.name}' not found in agent's tool list")
        
        # Execute the tool
        return tool.execute(tool_use)

