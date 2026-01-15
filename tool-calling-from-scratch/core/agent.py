from typing import List, Optional, Any
import json
from .tool import Tool, ToolUse
from .models import Message, Role, AssistantResponse, ResponseType
from providers.base import ModelProvider
from providers.factory import create_model_provider
from utils.conversation_logger import ConversationLogger


class Agent:
    def __init__(self, tools: List[Tool] = None, system_prompt: Optional[str] = None, client: str = 'gemini'):
        """
        Initialize the Agent.
        
        Args:
            tools: List of Tool objects available to the agent
            system_prompt: Optional system prompt for the agent
            client: Name of the model provider client to use (default: 'gemini')
            
        Note:
            Each model provider will automatically fetch its API key from the appropriate
            environment variable (e.g., GEMINI_API_KEY for Gemini).
        """
        self.tools = tools or []
        self.system_prompt = system_prompt
        self.conversation_history: List[Message] = []
        
        # Initialize model provider (each provider handles its own API key from env vars)
        self.llm_client: ModelProvider = create_model_provider(client=client)
        
        # Initialize conversation logger
        self.logger = ConversationLogger()
        
        # Add system message if provided
        if system_prompt:
            system_message = Message(role=Role.SYSTEM, content=system_prompt)
            self.conversation_history.append(system_message)
    
    def run(self, messages: List[Message], max_iterations: int = 10) -> Message:
        """
        Process messages and generate a response.
        Automatically chains tool calls until a final text response is generated.
        
        Args:
            messages: List of messages in the conversation
            max_iterations: Maximum number of tool execution iterations to prevent infinite loops (default: 10)
            
        Returns:
            Message: Assistant's final response message
        """
        # Start a new conversation if this is the first run
        if not self.logger.current_conversation_id:
            self.logger.start_conversation()
            # Log existing conversation history (including system message)
            for msg in self.conversation_history:
                self.logger.log_message(msg)
        
        # Log new incoming messages
        for msg in messages:
            self.logger.log_message(msg)
        
        # Update conversation history with new messages
        self.conversation_history.extend(messages)
        
        # Get the last user message
        user_messages = [msg for msg in messages if msg.role == Role.USER]
        if not user_messages:
            return Message(role=Role.ASSISTANT, content="I didn't receive any user messages.")
        
        # Loop until we get a text response (automatic tool chaining)
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            
            # Generate response based on current conversation history
            assistant_response = self._generate_response_from_history()
            
            # Log the assistant response
            self.logger.log_response(assistant_response)

            # Check if the response contains a tool use or is a text response
            if assistant_response.is_text():
                # Convert text response to Message - this is our final response
                response = Message(role=Role.ASSISTANT, content=assistant_response.text)
                self.conversation_history.append(response)
                self.logger.log_message(response)
                return response
                
            elif assistant_response.is_tool_use():
                # Execute all tools and collect results
                tool_results = []
                for tool_use in assistant_response.tool_uses:
                    try:
                        result = self.execute_tool(tool_use)
                        # Format result in a way that's easy for LLM to parse and reuse
                        formatted_result = self._format_tool_result(result)
                        tool_results.append(f"{tool_use.name}: {formatted_result}")
                        # Log successful tool execution
                        self.logger.log_tool_execution(tool_use, result)
                    except Exception as e:
                        tool_results.append(f"{tool_use.name}: Error - {str(e)}")
                        # Log failed tool execution
                        self.logger.log_tool_execution(tool_use, None, error=str(e))
                
                # Format tool execution results into a message and add to conversation history
                # This allows the LLM to see the results and potentially make more tool calls
                response_content = "Tool execution results:\n" + "\n".join(tool_results)
                tool_result_message = Message(role=Role.ASSISTANT, content=response_content)
                self.conversation_history.append(tool_result_message)
                self.logger.log_message(tool_result_message)
                
                # Continue the loop to generate the next response based on tool results
                continue
            else:
                # Fallback - treat as text response
                response = Message(role=Role.ASSISTANT, content="Unknown response type.")
                self.conversation_history.append(response)
                self.logger.log_message(response)
                return response
        
        # If we've exceeded max iterations, return an error message
        error_msg = Message(
            role=Role.ASSISTANT, 
            content=f"Maximum tool execution iterations ({max_iterations}) reached. The agent may be stuck in a loop."
        )
        self.conversation_history.append(error_msg)
        self.logger.log_message(error_msg)
        return error_msg
    
    def _generate_response_from_history(self) -> AssistantResponse:
        """
        Generate a response based on the current conversation history.
        Returns an AssistantResponse which can be either text or tool use.
        The response from Gemini is parsed deterministically to extract tool calls or text.
        """
        # Build tools description for the prompt using full tool prompts
        tools_description = None
        if self.tools:
            tool_descriptions = []
            for tool in self.tools:
                # Use get_prompt() if available for detailed format examples, otherwise fall back to basic description
                if hasattr(tool, 'get_prompt'):
                    tool_descriptions.append(tool.get_prompt())
                else:
                    # Fallback to basic description if get_prompt() is not available
                    tool_desc = f"- {tool.name}: {tool.description}"
                    if tool.parameters:
                        params_desc = ", ".join([f"{name}" for name in tool.parameters.keys()])
                        tool_desc += f" (parameters: {params_desc})"
                    tool_descriptions.append(tool_desc)
            tools_description = "\n\n".join(tool_descriptions)
        
        # Get the last message content for the LLM client
        # If the last message is an assistant message (tool results), we want to continue from there
        last_message = self.conversation_history[-1] if self.conversation_history else None
        if not last_message:
            return AssistantResponse.text_response("No messages in conversation history.")
        
        # Use the last message content as the prompt
        # The LLM client will use the full conversation history for context
        user_input = last_message.content
        
        # Call LLM client with full conversation history
        try:
            response_text = self.llm_client.generate_response(
                messages=self.conversation_history,
                system_prompt=self.system_prompt,
                tools_description=tools_description
            )
        except Exception as e:
            return AssistantResponse.text_response(f"Error calling LLM API: {str(e)}")
        
        # Parse the response deterministically
        return self._parse_response(response_text)
    
    def _generate_response(self, user_input: str) -> AssistantResponse:
        """
        Generate a response to user input using Gemini API.
        DEPRECATED: Use _generate_response_from_history() instead for automatic tool chaining.
        This method is kept for backward compatibility but may not work correctly with tool chaining.
        
        Returns an AssistantResponse which can be either text or tool use.
        The response from Gemini is parsed deterministically to extract tool calls or text.
        """
        # Build tools description for the prompt using full tool prompts
        tools_description = None
        if self.tools:
            tool_descriptions = []
            for tool in self.tools:
                # Use get_prompt() if available for detailed format examples, otherwise fall back to basic description
                if hasattr(tool, 'get_prompt'):
                    tool_descriptions.append(tool.get_prompt())
                else:
                    # Fallback to basic description if get_prompt() is not available
                    tool_desc = f"- {tool.name}: {tool.description}"
                    if tool.parameters:
                        params_desc = ", ".join([f"{name}" for name in tool.parameters.keys()])
                        tool_desc += f" (parameters: {params_desc})"
                    tool_descriptions.append(tool_desc)
            tools_description = "\n\n".join(tool_descriptions)
        
        # Call LLM client
        try:
            response_text = self.llm_client.generate_response(
                messages=self.conversation_history,
                system_prompt=self.system_prompt,
                tools_description=tools_description
            )
        except Exception as e:
            return AssistantResponse.text_response(f"Error calling LLM API: {str(e)}")
        
        # Parse the response deterministically
        return self._parse_response(response_text)
    
    def _parse_response(self, response_text: str) -> AssistantResponse:
        """
        Parse the response from the LLM deterministically.
        Expects JSON in one of the following shapes:
          - {"type": "text", "text": "..."}
          - {"type": "tool_use", "tool_uses": [{"name": "...", "params": {...}, "partial": false}]}
        Falls back to returning the raw text if parsing fails.
        """
        import json
        import re

        def try_load_json(text: str):
            try:
                return json.loads(text)
            except Exception:
                return None

        # First attempt: direct JSON parsing of the whole response
        data = try_load_json(response_text.strip())

        # Second attempt: extract the first JSON object substring
        if data is None:
            match = re.search(r'\{[\s\S]*\}', response_text)
            if match:
                data = try_load_json(match.group(0))

        # If still no JSON, treat as plain text
        if data is None or not isinstance(data, dict):
            return AssistantResponse.text_response(response_text.strip())

        response_type = data.get("type") or data.get("response_type")

        # Handle text response
        if response_type == "text":
            text_value = data.get("text", "")
            return AssistantResponse.text_response(str(text_value).strip())

        # Handle tool use response
        if response_type in ["tool_use", "tool"]:
            tool_entries = data.get("tool_uses") or data.get("tool_calls") or []
            tool_uses = []
            for entry in tool_entries:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name")
                params = entry.get("params") or entry.get("arguments") or {}
                partial = bool(entry.get("partial", False))
                if name:
                    tool_uses.append(ToolUse(name=name, params=params, partial=partial))

            if tool_uses:
                return AssistantResponse.tool_use_response(tool_uses)

        # Fallback to text if the JSON structure is unexpected
        return AssistantResponse.text_response(response_text.strip())
    
    def _parse_tool_use_xml(self, xml_content: str) -> List[ToolUse]:
        """
        Parse tool use from XML content. Handles both wrapped and unwrapped tool elements.
        """
        import xml.etree.ElementTree as ET
        import re
        
        tool_uses = []
        
        # Try to parse as XML
        try:
            # Try wrapping in a root element if needed
            try:
                root = ET.fromstring(xml_content)
            except ET.ParseError:
                # Try wrapping in a container
                root = ET.fromstring(f"<container>{xml_content}</container>")
            
            # If root is a tool_use container, iterate children
            if root.tag == "tool_use" or root.tag == "container":
                elements_to_process = root
            else:
                # Root itself might be a tool
                elements_to_process = [root]
            
            for elem in elements_to_process:
                # Skip if it's a known container tag
                if elem.tag in ["tool_use", "container"]:
                    for child in elem:
                        tool_name = child.tag
                        tool_params = self._extract_params_from_element(child)
                        if tool_name:
                            tool_uses.append(ToolUse(name=tool_name, params=tool_params))
                else:
                    # Element is a tool itself
                    tool_name = elem.tag
                    tool_params = self._extract_params_from_element(elem)
                    if tool_name:
                        tool_uses.append(ToolUse(name=tool_name, params=tool_params))
        except Exception:
            # If XML parsing fails, try regex-based parsing
            # Look for tool-like patterns: <tool_name>...</tool_name>
            tool_pattern = r'<([a-zA-Z_][a-zA-Z0-9_]*)>(.*?)</\1>'
            matches = re.finditer(tool_pattern, xml_content, re.DOTALL)
            for match in matches:
                tool_name = match.group(1)
                tool_content = match.group(2)
                
                # Extract parameters from the tool content
                tool_params = {}
                param_pattern = r'<([a-zA-Z_][a-zA-Z0-9_]*)>(.*?)</\1>'
                param_matches = re.finditer(param_pattern, tool_content, re.DOTALL)
                for param_match in param_matches:
                    param_name = param_match.group(1)
                    param_value = param_match.group(2).strip()
                    tool_params[param_name] = param_value
                
                if tool_name and tool_name not in ["text", "tool_use"]:
                    tool_uses.append(ToolUse(name=tool_name, params=tool_params))
        
        return tool_uses
    
    def _extract_params_from_element(self, elem) -> dict:
        """Extract parameters from an XML element."""
        import xml.etree.ElementTree as ET
        
        params = {}
        for child in elem:
            param_name = child.tag
            param_value = child.text or ""
            # Handle nested elements
            if len(child) > 0:
                param_value += "".join([ET.tostring(sub_elem, encoding='unicode') for sub_elem in child])
            params[param_name] = param_value.strip()
        return params
    
    def save_conversation(self):
        """Manually save the current conversation to disk."""
        self.logger.save_conversation()
    
    def reset(self):
        """Reset the conversation history."""
        # Save current conversation before resetting
        self.logger.save_conversation()
        self.logger.reset()
        
        self.conversation_history = []
        if self.system_prompt:
            system_message = Message(role=Role.SYSTEM, content=self.system_prompt)
            self.conversation_history.append(system_message)
    
    def _format_tool_result(self, result: Any) -> str:
        """
        Format a tool result in a way that's easy for the LLM to parse and reuse.
        For structured objects with to_dict(), formats as JSON.
        For other types, uses string representation.
        
        Args:
            result: The result from tool execution
            
        Returns:
            Formatted string representation of the result
        """
        # If the result has a to_dict() method, format as JSON for easy parsing
        if hasattr(result, 'to_dict'):
            try:
                result_dict = result.to_dict()
                return json.dumps(result_dict, indent=2)
            except Exception:
                # Fall back to string representation if to_dict() fails
                return str(result)
        
        # For dicts and lists, format as JSON
        if isinstance(result, (dict, list)):
            try:
                return json.dumps(result, indent=2)
            except Exception:
                return str(result)
        
        # For other types, use string representation
        return str(result)
    
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
