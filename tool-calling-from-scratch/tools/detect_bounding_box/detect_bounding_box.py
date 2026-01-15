from typing import Any, Dict, Optional
import json
import re
from PIL import Image
from core.tool import Tool, ToolUse
from core.models import Message, Role
from providers.factory import create_model_provider
from providers.base import ModelProvider
from prompt.bounding_box_prompt import BOUNDING_BOX_PROMPT
from .bounding_box_input import BoundingBoxInput
from .bounding_box import BoundingBox
from .bounding_box_output import BoundingBoxOutput


class DetectBoundingBox(Tool):
    """Tool for detecting bounding boxes around items in images."""
    
    def __init__(self, model_provider: Optional[ModelProvider] = None, model_name: str = "gemini-3-flash-preview"):
        """
        Initialize the DetectBoundingBox tool.
        
        Args:
            model_provider: Optional ModelProvider instance. If not provided, will create a Gemini provider.
            model_name: Name of the model to use (default: "gemini-3-flash-preview")
        """
        parameters = {
            "image_path": {
                "type": "string",
                "description": "Local file path to the image to process (e.g., /path/to/image.jpg or ./assets/image.png)"
            },
            "label": {
                "type": "string",
                "description": "The label/class of the item to detect (e.g., 'button', 'text', 'icon')"
            }
        }
        # Initialize parent Tool class (no function provided since we override execute())
        super().__init__(
            name="detect_bounding_box",
            description="Detects and returns bounding boxes around specific items in images",
            function=None,  # We override execute() instead
            parameters=parameters
        )
        # Initialize model provider if not provided
        self.model_provider = model_provider or create_model_provider("gemini", model_name=model_name)
    
    def get_prompt(self) -> str:
        """
        Returns the prompt description for this tool to be added to the system prompt.
        
        Returns:
            A string describing the tool's purpose and usage
        """
        return f"""Tool: {self.name}
Description: {self.description}
Parameters:
  - image_path (string): Local file path to the image to process (e.g., /path/to/image.jpg or ./assets/image.png)
  - label (string): The label/class of the item to detect (e.g., 'button', 'text', 'icon')

Input format:
  {{
    "image_path": "/path/to/image.jpg",
    "label": "button"
  }}

Output format:
  {{
    "boxes": [
      {{
        "confidence": 0.92,
        "xyxy": [0.219, 0.287, 0.320, 0.338]
      }}
    ]
  }}

The output contains:
  - boxes: Array of detected bounding boxes, each with:
    - confidence: Detection confidence score (0.0 to 1.0)
    - xyxy: Bounding box coordinates as [x1, y1, x2, y2] in normalized format (0.0 to 1.0), where (x1,y1) is top-left and (x2,y2) is bottom-right"""
    
    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """
        Extract JSON from the LLM response, handling markdown code blocks if present.
        
        Args:
            response_text: Raw response text from the LLM
            
        Returns:
            Parsed JSON dictionary
        """
        # Try to find JSON in markdown code blocks first
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object directly
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # If no JSON found, try parsing the whole response
                json_str = response_text.strip()
        
        # Parse JSON
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from response: {e}\nResponse was: {response_text[:500]}")
    
    def execute(self, tool_use: ToolUse) -> BoundingBoxOutput:
        """
        Execute the detect_bounding_box tool using Gemini.
        
        Args:
            tool_use: ToolUse object containing the tool name and parameters
                Expected params:
                - image_path: Local file path to the image to process
                - label: The label/class to detect
            
        Returns:
            BoundingBoxOutput containing the detection results
        """
        # Validate tool name (parent class would do this, but we override execute so we do it here)
        if tool_use.name != self.name:
            raise ValueError(f"Tool name mismatch: expected {self.name}, got {tool_use.name}")
        
        # Parse input parameters into structured class
        input_data = BoundingBoxInput.from_dict(tool_use.params)
        
        if not input_data.image_path:
            raise ValueError("image_path parameter is required")
        if not input_data.label:
            raise ValueError("label parameter is required")
        
        # Load image to get dimensions and verify it exists
        try:
            image = Image.open(input_data.image_path)
            image_width, image_height = image.size
        except Exception as e:
            raise ValueError(f"Failed to load image from {input_data.image_path}: {e}")
        
        print(f"Detecting bounding boxes for label '{input_data.label}' in image '{input_data.image_path}'")
        
        # Create user message with image and label request
        user_message_content = f"Detect all instances of '{input_data.label}' in this image and return bounding boxes."
        user_message = Message(
            role=Role.USER,
            content=user_message_content,
            image_path=input_data.image_path
        )
        
        # Call Gemini with bounding box detection prompt
        try:
            response_text = self.model_provider.generate_response(
                messages=[user_message],
                system_prompt=BOUNDING_BOX_PROMPT,
                tools_description=None
            )
        except Exception as e:
            raise RuntimeError(f"Failed to call Gemini API: {e}")
        
        # Parse JSON response
        try:
            response_dict = self._extract_json_from_response(response_text)
        except ValueError as e:
            raise ValueError(f"Failed to parse bounding box response: {e}")
        
        # Validate response format
        if "boxes" not in response_dict:
            raise ValueError(f"Invalid response format. Expected 'boxes' key. Got: {list(response_dict.keys())}")
        
        # Convert boxes to BoundingBox objects
        boxes = []
        for box_dict in response_dict.get("boxes", []):
            if "confidence" not in box_dict or "xyxy" not in box_dict:
                raise ValueError(f"Invalid box format. Expected 'confidence' and 'xyxy' keys. Got: {list(box_dict.keys())}")
            
            boxes.append(BoundingBox(
                confidence=float(box_dict["confidence"]),
                xyxy=[float(x) for x in box_dict["xyxy"]]
            ))
        
        return BoundingBoxOutput(
            width=image_width,
            height=image_height,
            boxes=boxes
        )
