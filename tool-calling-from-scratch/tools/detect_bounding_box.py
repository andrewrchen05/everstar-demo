from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import json
import re
from PIL import Image
from core.tool import ToolUse
from core.models import Message, Role
from providers.factory import create_model_provider
from providers.base import ModelProvider
from prompt.bounding_box_prompt import BOUNDING_BOX_PROMPT


@dataclass
class BoundingBoxInput:
    """Input parameters for bounding box detection."""
    image_path: str
    label: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BoundingBoxInput":
        """Create BoundingBoxInput from a dictionary."""
        return cls(
            image_path=data.get("image_path", ""),
            label=data.get("label", "")
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "image_path": self.image_path,
            "label": self.label
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"BoundingBoxInput(image_path={self.image_path}, label={self.label})"


@dataclass
class BoundingBox:
    """Represents a single detected bounding box."""
    confidence: float
    xyxy: List[int]  # [x1, y1, x2, y2]
    
    def __post_init__(self):
        """Validate bounding box data."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        if len(self.xyxy) != 4:
            raise ValueError(f"xyxy must contain exactly 4 coordinates, got {len(self.xyxy)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "confidence": self.confidence,
            "xyxy": self.xyxy
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"BoundingBox(confidence={self.confidence:.2f}, xyxy={self.xyxy})"


@dataclass
class BoundingBoxOutput:
    """Output result from bounding box detection."""
    width: int
    height: int
    boxes: List[BoundingBox]
    
    def __post_init__(self):
        """Validate output data."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Width and height must be positive, got {self.width}x{self.height}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "width": self.width,
            "height": self.height,
            "boxes": [box.to_dict() for box in self.boxes]
        }
    
    def __str__(self) -> str:
        """String representation."""
        boxes_str = ", ".join([str(box) for box in self.boxes])
        return f"BoundingBoxOutput(width={self.width}, height={self.height}, boxes=[{boxes_str}])"


class DetectBoundingBox:
    """Tool for detecting bounding boxes around items in images."""
    
    def __init__(self, model_provider: Optional[ModelProvider] = None, model_name: str = "gemini-2.5-flash"):
        """
        Initialize the DetectBoundingBox tool.
        
        Args:
            model_provider: Optional ModelProvider instance. If not provided, will create a Gemini provider.
            model_name: Name of the model to use (default: "gemini-2.5-flash")
        """
        self.name = "detect_bounding_box"
        self.description = "Detects and returns bounding boxes around specific items in images"
        self.parameters = {
            "image_path": {
                "type": "string",
                "description": "Local file path to the image to process (e.g., /path/to/image.jpg or ./assets/image.png)"
            },
            "label": {
                "type": "string",
                "description": "The label/class of the item to detect (e.g., 'button', 'text', 'icon')"
            }
        }
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
    "width": 1920,
    "height": 1080,
    "boxes": [
      {{
        "confidence": 0.92,
        "xyxy": [420, 310, 615, 365]
      }}
    ]
  }}

The output contains:
  - width: Image width in pixels
  - height: Image height in pixels
  - boxes: Array of detected bounding boxes, each with:
    - confidence: Detection confidence score (0.0 to 1.0)
    - xyxy: Bounding box coordinates as [x1, y1, x2, y2] where (x1,y1) is top-left and (x2,y2) is bottom-right"""
    
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
        
        # Use actual image dimensions (more reliable than trusting LLM response)
        # But validate that LLM returned dimensions match if provided
        if "width" in response_dict and "height" in response_dict:
            llm_width = int(response_dict["width"])
            llm_height = int(response_dict["height"])
            if llm_width != image_width or llm_height != image_height:
                print(f"Warning: LLM returned dimensions {llm_width}x{llm_height}, but actual image is {image_width}x{image_height}. Using actual dimensions.")
        
        # Convert boxes to BoundingBox objects
        boxes = []
        for box_dict in response_dict.get("boxes", []):
            if "confidence" not in box_dict or "xyxy" not in box_dict:
                raise ValueError(f"Invalid box format. Expected 'confidence' and 'xyxy' keys. Got: {list(box_dict.keys())}")
            
            boxes.append(BoundingBox(
                confidence=float(box_dict["confidence"]),
                xyxy=[int(x) for x in box_dict["xyxy"]]
            ))
        
        return BoundingBoxOutput(
            width=image_width,
            height=image_height,
            boxes=boxes
        )
