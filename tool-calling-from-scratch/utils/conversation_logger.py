"""
Conversation history logger for debugging agent interactions.
"""
import json
import os
from datetime import datetime
from typing import List, Optional, Any, Dict
from uuid import uuid4


class ConversationLogger:
    """Logs agent conversations to files for debugging."""
    
    def __init__(self, output_dir: str = "conversation_history"):
        """
        Initialize the conversation logger.
        
        Args:
            output_dir: Directory to save conversation history files
        """
        self.output_dir = output_dir
        self.current_conversation_id: Optional[str] = None
        self.conversation_data: Dict[str, Any] = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def start_conversation(self) -> str:
        """
        Start a new conversation and return its unique ID.
        
        Returns:
            Unique conversation ID
        """
        # Generate unique conversation ID: timestamp + UUID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid4())[:8]
        self.current_conversation_id = f"{timestamp}_{unique_id}"
        
        # Initialize conversation data
        self.conversation_data = {
            "conversation_id": self.current_conversation_id,
            "started_at": datetime.now().isoformat(),
            "messages": [],
            "tool_executions": [],
            "responses": []
        }
        
        return self.current_conversation_id
    
    def log_message(self, message):
        """
        Log a message to the conversation history.
        
        Args:
            message: Message to log
        """
        if not self.current_conversation_id:
            self.start_conversation()
        
        message_data = {
            "role": message.role.value,
            "content": message.content,
            "timestamp": datetime.now().isoformat()
        }
        
        if message.image_path:
            message_data["image_path"] = message.image_path
        
        self.conversation_data["messages"].append(message_data)
    
    def log_tool_execution(self, tool_use, result: Any, error: Optional[str] = None):
        """
        Log a tool execution to the conversation history.
        
        Args:
            tool_use: ToolUse object that was executed
            result: Result of the tool execution (or None if error)
            error: Error message if execution failed
        """
        if not self.current_conversation_id:
            self.start_conversation()
        
        execution_data = {
            "timestamp": datetime.now().isoformat(),
            "tool_name": tool_use.name,
            "parameters": tool_use.params,
            "success": error is None
        }
        
        if error:
            execution_data["error"] = str(error)
        else:
            # Convert result to dict/JSON-serializable format
            try:
                if isinstance(result, (dict, list)):
                    execution_data["result"] = result
                elif hasattr(result, 'to_dict'):
                    # Handle dataclass objects with to_dict() method
                    execution_data["result"] = result.to_dict()
                else:
                    execution_data["result"] = str(result)
            except Exception as e:
                execution_data["result"] = f"<Unable to serialize result: {str(e)}>"
        
        self.conversation_data["tool_executions"].append(execution_data)
    
    def log_response(self, response):
        """
        Log an assistant response to the conversation history.
        
        Args:
            response: AssistantResponse to log
        """
        if not self.current_conversation_id:
            self.start_conversation()
        
        response_data = {
            "timestamp": datetime.now().isoformat(),
            "type": response.response_type.value
        }
        
        if response.is_text():
            response_data["text"] = response.text
        elif response.is_tool_use():
            response_data["tool_uses"] = [
                {
                    "name": tool_use.name,
                    "params": tool_use.params
                }
                for tool_use in response.tool_uses
            ]
        
        self.conversation_data["responses"].append(response_data)
    
    def save_conversation(self):
        """Save the current conversation to a file."""
        if not self.current_conversation_id:
            return
        
        # Add end timestamp
        self.conversation_data["ended_at"] = datetime.now().isoformat()
        
        # Save to JSON file
        filename = f"{self.current_conversation_id}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to save conversation history: {e}")
    
    def reset(self):
        """Reset the logger for a new conversation."""
        if self.current_conversation_id:
            self.save_conversation()
        
        self.current_conversation_id = None
        self.conversation_data = {}
