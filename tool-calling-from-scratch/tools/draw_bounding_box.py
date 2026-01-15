from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
from core.tool import Tool, ToolUse
from tools.detect_bounding_box import BoundingBoxOutput, BoundingBox


@dataclass
class DrawBoundingBoxInput:
    """Input parameters for drawing bounding boxes on an image."""
    image_path: str
    boxes: Union[List[Dict[str, Any]], Dict[str, Any]]  # Can be list of boxes or BoundingBoxOutput dict
    output_path: Optional[str] = None
    color: str = "red"
    line_width: int = 3
    draw_labels: bool = True
    label_text: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DrawBoundingBoxInput":
        """Create DrawBoundingBoxInput from a dictionary."""
        return cls(
            image_path=data.get("image_path", ""),
            boxes=data.get("boxes", []),
            output_path=data.get("output_path"),
            color=data.get("color", "red"),
            line_width=int(data.get("line_width", 3)),
            draw_labels=bool(data.get("draw_labels", True)),
            label_text=data.get("label_text")
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "image_path": self.image_path,
            "boxes": self.boxes,
            "output_path": self.output_path,
            "color": self.color,
            "line_width": self.line_width,
            "draw_labels": self.draw_labels,
            "label_text": self.label_text
        }


@dataclass
class DrawBoundingBoxOutput:
    """Output result from drawing bounding boxes."""
    output_path: str
    boxes_drawn: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "output_path": self.output_path,
            "boxes_drawn": self.boxes_drawn
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"DrawBoundingBoxOutput(output_path={self.output_path}, boxes_drawn={self.boxes_drawn})"


class DrawBoundingBox(Tool):
    """Tool for drawing bounding boxes on images."""
    
    def __init__(self):
        """
        Initialize the DrawBoundingBox tool.
        """
        parameters = {
            "image_path": {
                "type": "string",
                "description": "Local file path to the image to draw on (e.g., /path/to/image.jpg or ./assets/image.png)"
            },
            "boxes": {
                "type": "object",
                "description": "Bounding boxes to draw. Can be either: 1) A BoundingBoxOutput dict with 'width', 'height', and 'boxes' keys, or 2) A list of box dicts, each with 'xyxy' coordinates [x1, y1, x2, y2] and optionally 'confidence' and 'label'"
            },
            "output_path": {
                "type": "string",
                "description": "Optional output file path. If not provided, will save to the same directory as input with '_annotated' suffix"
            },
            "color": {
                "type": "string",
                "description": "Color for the bounding box lines (default: 'red'). Can be color name or hex code (e.g., 'red', '#FF0000')"
            },
            "line_width": {
                "type": "integer",
                "description": "Width of the bounding box lines in pixels (default: 3)"
            },
            "draw_labels": {
                "type": "boolean",
                "description": "Whether to draw labels on the boxes (default: false)"
            },
            "label_text": {
                "type": "string",
                "description": "Optional label text to display on all boxes. If not provided and draw_labels is true, will use confidence scores or box indices"
            }
        }
        super().__init__(
            name="draw_bounding_box",
            description="Draws bounding boxes on an image using the provided coordinates",
            function=None,  # We override execute() instead
            parameters=parameters
        )
    
    def get_prompt(self) -> str:
        """
        Returns the prompt description for this tool to be added to the system prompt.
        
        Returns:
            A string describing the tool's purpose and usage
        """
        return f"""Tool: {self.name}
Description: {self.description}
Parameters:
  - image_path (string): Local file path to the image to draw on (e.g., /path/to/image.jpg or ./assets/image.png)
  - boxes (object): Bounding boxes to draw. Can be either:
      a) A list of box dicts, each with 'xyxy' coordinates [x1, y1, x2, y2] and optionally 'confidence' and 'label' (recommended)
      b) A BoundingBoxOutput dict (flat structure with 'width', 'height', and 'boxes' keys) - when passed as the 'boxes' parameter, this creates a nested structure where the parameter 'boxes' contains a dict with its own 'boxes' key
  - output_path (string, optional): Output file path. If not provided, saves to same directory with '_annotated' suffix
  - color (string, optional): Color for bounding box lines (default: 'red')
  - line_width (integer, optional): Width of bounding box lines in pixels (default: 3)
  - draw_labels (boolean, optional): Whether to draw labels on boxes (default: false)
  - label_text (string, optional): Label text to display on all boxes

Input format (simple list of boxes - recommended):
  {{
    "image_path": "/path/to/image.jpg",
    "boxes": [
      {{
        "xyxy": [420, 310, 615, 365],
        "confidence": 0.92,
        "label": "button"
      }}
    ],
    "output_path": "/path/to/output.jpg",
    "color": "red",
    "line_width": 3,
    "draw_labels": true
  }}

Or with BoundingBoxOutput format (when using the full output dict from detect_bounding_box):
  Note: BoundingBoxOutput is a flat dict with 'width', 'height', and 'boxes' keys.
  When passed as the 'boxes' parameter, it creates: boxes['boxes'] contains the list.
  {{
    "image_path": "/path/to/image.jpg",
    "boxes": {{
      "width": 1920,
      "height": 1080,
      "boxes": [
        {{
          "confidence": 0.92,
          "xyxy": [420, 310, 615, 365]
        }}
      ]
    }},
    "output_path": "/path/to/output.jpg"
  }}

Output format:
  {{
    "output_path": "/path/to/output.jpg",
    "boxes_drawn": 1
  }}

The output contains:
  - output_path: Path where the annotated image was saved
  - boxes_drawn: Number of bounding boxes drawn on the image"""
    
    def _parse_boxes(self, boxes_data: Union[List[Dict[str, Any]], Dict[str, Any]]) -> List[BoundingBox]:
        """
        Parse boxes from various input formats.
        
        Args:
            boxes_data: Can be:
                - BoundingBoxOutput dict with 'width', 'height', 'boxes' keys
                - List of box dicts with 'xyxy' coordinates
        
        Returns:
            List of BoundingBox objects
        """
        boxes = []
        
        # Check if it's a BoundingBoxOutput format
        if isinstance(boxes_data, dict) and "boxes" in boxes_data:
            # It's a BoundingBoxOutput dict
            for box_dict in boxes_data["boxes"]:
                if "xyxy" not in box_dict:
                    continue
                boxes.append(BoundingBox(
                    confidence=float(box_dict.get("confidence", 1.0)),
                    xyxy=[float(x) for x in box_dict["xyxy"]]
                ))
        elif isinstance(boxes_data, list):
            # It's a list of box dicts
            for box_dict in boxes_data:
                if "xyxy" not in box_dict:
                    continue
                boxes.append(BoundingBox(
                    confidence=float(box_dict.get("confidence", 1.0)),
                    xyxy=[float(x) for x in box_dict["xyxy"]]
                ))
        else:
            raise ValueError(f"Invalid boxes format. Expected dict with 'boxes' key or list of box dicts. Got: {type(boxes_data)}")
        
        return boxes
    
    def _get_output_path(self, input_path: str, output_path: Optional[str] = None) -> str:
        """
        Generate output path if not provided.
        
        Args:
            input_path: Input image path
            output_path: Optional output path
        
        Returns:
            Output file path
        """
        if output_path:
            return output_path
        
        # Generate output path by adding '_annotated' before the extension
        import os
        base, ext = os.path.splitext(input_path)
        return f"{base}_annotated{ext}"
    
    def _hex_to_rgb(self, color: str) -> tuple:
        """
        Convert color string to RGB tuple.
        Supports color names and hex codes.
        
        Args:
            color: Color name (e.g., 'red') or hex code (e.g., '#FF0000')
        
        Returns:
            RGB tuple (r, g, b)
        """
        # Common color names mapping
        color_map = {
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "cyan": (0, 255, 255),
            "magenta": (255, 0, 255),
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "orange": (255, 165, 0),
            "purple": (128, 0, 128),
            "pink": (255, 192, 203),
        }
        
        # Check if it's a color name
        if color.lower() in color_map:
            return color_map[color.lower()]
        
        # Check if it's a hex code
        if color.startswith("#"):
            color = color[1:]
            if len(color) == 6:
                return tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
            elif len(color) == 3:
                return tuple(int(c * 2, 16) for c in color)
        
        # Default to red if parsing fails
        return (255, 0, 0)
    
    def execute(self, tool_use: ToolUse) -> DrawBoundingBoxOutput:
        """
        Execute the draw_bounding_box tool.
        
        Args:
            tool_use: ToolUse object containing the tool name and parameters
                Expected params:
                - image_path: Local file path to the image
                - boxes: Bounding boxes to draw (BoundingBoxOutput dict or list of box dicts)
                - output_path: Optional output file path
                - color: Optional color for boxes (default: 'red')
                - line_width: Optional line width (default: 3)
                - draw_labels: Optional flag to draw labels (default: false)
                - label_text: Optional label text
        
        Returns:
            DrawBoundingBoxOutput containing the output path and number of boxes drawn
        """
        # Validate tool name
        if tool_use.name != self.name:
            raise ValueError(f"Tool name mismatch: expected {self.name}, got {tool_use.name}")
        
        # Parse input parameters
        input_data = DrawBoundingBoxInput.from_dict(tool_use.params)
        
        if not input_data.image_path:
            raise ValueError("image_path parameter is required")
        if not input_data.boxes:
            raise ValueError("boxes parameter is required")
        
        # Load image
        try:
            image = Image.open(input_data.image_path)
            width, height = image.size
        except Exception as e:
            raise ValueError(f"Failed to load image from {input_data.image_path}: {e}")
        
        # Parse boxes
        try:
            boxes = self._parse_boxes(input_data.boxes)
        except Exception as e:
            raise ValueError(f"Failed to parse boxes: {e}")
        
        if not boxes:
            raise ValueError("No valid boxes found in the provided data")
        
        # Get output path
        output_path = self._get_output_path(input_data.image_path, input_data.output_path)
        
        # Convert color to RGB
        color_rgb = self._hex_to_rgb(input_data.color)
        
        # Create a copy of the image for drawing
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)
        
        # Draw each bounding box
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy
            
            # Convert normalized coordinates (0.0 to 1.0) to pixel coordinates
            # Check if coordinates are normalized (all between 0.0 and 1.0)
            if all(0.0 <= coord <= 1.0 for coord in [x1, y1, x2, y2]):
                # Coordinates are normalized, convert to pixels
                x1 = int(x1 * width)
                y1 = int(y1 * height)
                x2 = int(x2 * width)
                y2 = int(y2 * height)
            else:
                # Assume coordinates are already in pixel format (for backward compatibility)
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
            
            # Draw rectangle
            draw.rectangle(
                [x1, y1, x2, y2],
                outline=color_rgb,
                width=input_data.line_width
            )
            
            # Draw label if requested
            if input_data.draw_labels:
                # Determine label text
                if input_data.label_text:
                    label = input_data.label_text
                elif hasattr(box, 'label') and box.label:
                    label = box.label
                else:
                    label = f"{box.confidence:.2f}" if box.confidence < 1.0 else f"Box {i+1}"
                
                # Try to load a font, fallback to default if not available
                try:
                    # Try to use a default font
                    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
                except:
                    try:
                        font = ImageFont.load_default()
                    except:
                        font = None
                
                # Calculate text size
                if font:
                    bbox = draw.textbbox((0, 0), label, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                else:
                    # Approximate size if font loading fails
                    text_width = len(label) * 6
                    text_height = 12
                
                # Draw label background
                label_y = max(y1 - text_height - 4, 0)
                draw.rectangle(
                    [x1, label_y, x1 + text_width + 4, label_y + text_height + 4],
                    fill=color_rgb,
                    outline=color_rgb
                )
                
                # Draw label text
                text_color = (255, 255, 255) if sum(color_rgb) < 384 else (0, 0, 0)  # White or black based on background
                draw.text(
                    (x1 + 2, label_y + 2),
                    label,
                    fill=text_color,
                    font=font
                )
        
        # Save the annotated image
        try:
            draw_image.save(output_path)
            print(f"Drew {len(boxes)} bounding box(es) on image and saved to {output_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save annotated image to {output_path}: {e}")
        
        return DrawBoundingBoxOutput(
            output_path=output_path,
            boxes_drawn=len(boxes)
        )
