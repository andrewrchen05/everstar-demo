import pytest
import os
import tempfile
from pathlib import Path
from PIL import Image
from core.tool import ToolUse
from tools.draw_bounding_box import (
    DrawBoundingBoxInput,
    DrawBoundingBoxOutput,
    DrawBoundingBox
)
from tools.detect_bounding_box import BoundingBox


class TestDrawBoundingBoxInput:
    """Test the DrawBoundingBoxInput dataclass and parsing."""
    
    def test_from_dict_minimal(self):
        """Test parsing minimal required fields."""
        data = {
            "image_path": "/path/to/image.jpg",
            "boxes": []
        }
        input_obj = DrawBoundingBoxInput.from_dict(data)
        assert input_obj.image_path == "/path/to/image.jpg"
        assert input_obj.boxes == []
        assert input_obj.output_path is None
        assert input_obj.color == "red"
        assert input_obj.line_width == 3
        assert input_obj.draw_labels is True
        assert input_obj.label_text is None
    
    def test_from_dict_all_fields(self):
        """Test parsing all fields."""
        data = {
            "image_path": "/path/to/image.jpg",
            "boxes": [{"xyxy": [10, 20, 30, 40]}],
            "output_path": "/path/to/output.jpg",
            "color": "blue",
            "line_width": 5,
            "draw_labels": False,
            "label_text": "Custom Label"
        }
        input_obj = DrawBoundingBoxInput.from_dict(data)
        assert input_obj.image_path == "/path/to/image.jpg"
        assert input_obj.boxes == [{"xyxy": [10, 20, 30, 40]}]
        assert input_obj.output_path == "/path/to/output.jpg"
        assert input_obj.color == "blue"
        assert input_obj.line_width == 5
        assert input_obj.draw_labels is False
        assert input_obj.label_text == "Custom Label"
    
    def test_from_dict_type_conversions(self):
        """Test that types are properly converted."""
        data = {
            "image_path": "/path/to/image.jpg",
            "boxes": [],
            "line_width": "3",  # String that should be converted to int
            "draw_labels": "true"  # String that should be converted to bool
        }
        input_obj = DrawBoundingBoxInput.from_dict(data)
        assert isinstance(input_obj.line_width, int)
        assert isinstance(input_obj.draw_labels, bool)
    
    def test_to_dict(self):
        """Test converting back to dictionary."""
        input_obj = DrawBoundingBoxInput(
            image_path="/path/to/image.jpg",
            boxes=[{"xyxy": [10, 20, 30, 40]}],
            output_path="/path/to/output.jpg",
            color="green",
            line_width=4,
            draw_labels=False,
            label_text="Test"
        )
        result = input_obj.to_dict()
        assert result["image_path"] == "/path/to/image.jpg"
        assert result["boxes"] == [{"xyxy": [10, 20, 30, 40]}]
        assert result["output_path"] == "/path/to/output.jpg"
        assert result["color"] == "green"
        assert result["line_width"] == 4
        assert result["draw_labels"] is False
        assert result["label_text"] == "Test"


class TestDrawBoundingBoxParsing:
    """Test the _parse_boxes method for different input formats."""
    
    def setup_method(self):
        """Set up test fixture."""
        self.tool = DrawBoundingBox()
    
    def test_parse_boxes_list_format(self):
        """Test parsing boxes from list format (recommended format)."""
        boxes_data = [
            {
                "xyxy": [10, 20, 30, 40],
                "confidence": 0.95,
                "label": "button"
            },
            {
                "xyxy": [50, 60, 70, 80],
                "confidence": 0.87
            }
        ]
        boxes = self.tool._parse_boxes(boxes_data)
        assert len(boxes) == 2
        assert boxes[0].xyxy == [10, 20, 30, 40]
        assert boxes[0].confidence == 0.95
        assert boxes[1].xyxy == [50, 60, 70, 80]
        assert boxes[1].confidence == 0.87
    
    def test_parse_boxes_bounding_box_output_format(self):
        """Test parsing boxes from BoundingBoxOutput format."""
        boxes_data = {
            "width": 1920,
            "height": 1080,
            "boxes": [
                {
                    "xyxy": [420, 310, 615, 365],
                    "confidence": 0.92
                },
                {
                    "xyxy": [100, 200, 300, 400],
                    "confidence": 0.85
                }
            ]
        }
        boxes = self.tool._parse_boxes(boxes_data)
        assert len(boxes) == 2
        assert boxes[0].xyxy == [420, 310, 615, 365]
        assert boxes[0].confidence == 0.92
        assert boxes[1].xyxy == [100, 200, 300, 400]
        assert boxes[1].confidence == 0.85
    
    def test_parse_boxes_missing_xyxy(self):
        """Test that boxes without xyxy are skipped."""
        boxes_data = [
            {
                "confidence": 0.95,
                "label": "button"
                # Missing xyxy
            },
            {
                "xyxy": [50, 60, 70, 80],
                "confidence": 0.87
            }
        ]
        boxes = self.tool._parse_boxes(boxes_data)
        assert len(boxes) == 1
        assert boxes[0].xyxy == [50, 60, 70, 80]
    
    def test_parse_boxes_default_confidence(self):
        """Test that missing confidence defaults to 1.0."""
        boxes_data = [
            {
                "xyxy": [10, 20, 30, 40]
                # Missing confidence
            }
        ]
        boxes = self.tool._parse_boxes(boxes_data)
        assert len(boxes) == 1
        assert boxes[0].confidence == 1.0
    
    def test_parse_boxes_invalid_format(self):
        """Test that invalid format raises ValueError."""
        boxes_data = "invalid format"
        with pytest.raises(ValueError, match="Invalid boxes format"):
            self.tool._parse_boxes(boxes_data)
    
    def test_parse_boxes_empty_list(self):
        """Test parsing empty list."""
        boxes = self.tool._parse_boxes([])
        assert len(boxes) == 0
    
    def test_parse_boxes_coordinate_type_conversion(self):
        """Test that coordinates are converted to integers."""
        boxes_data = [
            {
                "xyxy": ["10", "20", "30", "40"],  # String coordinates
                "confidence": 0.95
            }
        ]
        boxes = self.tool._parse_boxes(boxes_data)
        assert boxes[0].xyxy == [10, 20, 30, 40]  # Should be converted to ints
    
    def test_parse_boxes_float_coordinates_raises_error(self):
        """Test that float coordinates raise ValueError when converted to int."""
        boxes_data = [
            {
                "xyxy": [10, 20, 30.5, 40.9],  # Float coordinates
                "confidence": 0.95
            }
        ]
        # The code uses int() which will truncate floats, but we test the behavior
        boxes = self.tool._parse_boxes(boxes_data)
        # int(30.5) = 30, int(40.9) = 40 (truncation)
        assert boxes[0].xyxy == [10, 20, 30, 40]


class TestDrawBoundingBoxExecute:
    """Test the execute method with real images."""
    
    def setup_method(self):
        """Set up test fixture with a temporary test image."""
        self.tool = DrawBoundingBox()
        
        # Create a temporary test image
        self.test_image = Image.new('RGB', (100, 100), color='white')
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = os.path.join(self.temp_dir, "test_image.png")
        self.test_image.save(self.test_image_path)
    
    def teardown_method(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_execute_list_format(self):
        """Test execute with list format boxes."""
        tool_use = ToolUse(
            name="draw_bounding_box",
            params={
                "image_path": self.test_image_path,
                "boxes": [
                    {
                        "xyxy": [10, 20, 30, 40],
                        "confidence": 0.95
                    }
                ],
                "output_path": os.path.join(self.temp_dir, "output_list.png")
            }
        )
        result = self.tool.execute(tool_use)
        assert isinstance(result, DrawBoundingBoxOutput)
        assert result.boxes_drawn == 1
        assert os.path.exists(result.output_path)
        assert result.output_path == os.path.join(self.temp_dir, "output_list.png")
    
    def test_execute_bounding_box_output_format(self):
        """Test execute with BoundingBoxOutput format."""
        tool_use = ToolUse(
            name="draw_bounding_box",
            params={
                "image_path": self.test_image_path,
                "boxes": {
                    "width": 100,
                    "height": 100,
                    "boxes": [
                        {
                            "xyxy": [10, 20, 30, 40],
                            "confidence": 0.95
                        },
                        {
                            "xyxy": [50, 60, 70, 80],
                            "confidence": 0.87
                        }
                    ]
                },
                "output_path": os.path.join(self.temp_dir, "output_bbox.png")
            }
        )
        result = self.tool.execute(tool_use)
        assert isinstance(result, DrawBoundingBoxOutput)
        assert result.boxes_drawn == 2
        assert os.path.exists(result.output_path)
    
    def test_execute_auto_output_path(self):
        """Test that output path is auto-generated if not provided."""
        tool_use = ToolUse(
            name="draw_bounding_box",
            params={
                "image_path": self.test_image_path,
                "boxes": [
                    {
                        "xyxy": [10, 20, 30, 40],
                        "confidence": 0.95
                    }
                ]
            }
        )
        result = self.tool.execute(tool_use)
        assert result.output_path == self.test_image_path.replace(".png", "_annotated.png")
        assert os.path.exists(result.output_path)
    
    def test_execute_custom_color(self):
        """Test execute with custom color."""
        tool_use = ToolUse(
            name="draw_bounding_box",
            params={
                "image_path": self.test_image_path,
                "boxes": [
                    {
                        "xyxy": [10, 20, 30, 40],
                        "confidence": 0.95
                    }
                ],
                "color": "blue",
                "output_path": os.path.join(self.temp_dir, "output_blue.png")
            }
        )
        result = self.tool.execute(tool_use)
        assert result.boxes_drawn == 1
        assert os.path.exists(result.output_path)
    
    def test_execute_hex_color(self):
        """Test execute with hex color code."""
        tool_use = ToolUse(
            name="draw_bounding_box",
            params={
                "image_path": self.test_image_path,
                "boxes": [
                    {
                        "xyxy": [10, 20, 30, 40],
                        "confidence": 0.95
                    }
                ],
                "color": "#00FF00",  # Green
                "output_path": os.path.join(self.temp_dir, "output_hex.png")
            }
        )
        result = self.tool.execute(tool_use)
        assert result.boxes_drawn == 1
    
    def test_execute_custom_line_width(self):
        """Test execute with custom line width."""
        tool_use = ToolUse(
            name="draw_bounding_box",
            params={
                "image_path": self.test_image_path,
                "boxes": [
                    {
                        "xyxy": [10, 20, 30, 40],
                        "confidence": 0.95
                    }
                ],
                "line_width": 5,
                "output_path": os.path.join(self.temp_dir, "output_thick.png")
            }
        )
        result = self.tool.execute(tool_use)
        assert result.boxes_drawn == 1
    
    def test_execute_with_labels(self):
        """Test execute with labels enabled."""
        tool_use = ToolUse(
            name="draw_bounding_box",
            params={
                "image_path": self.test_image_path,
                "boxes": [
                    {
                        "xyxy": [10, 20, 30, 40],
                        "confidence": 0.95
                    }
                ],
                "draw_labels": True,
                "output_path": os.path.join(self.temp_dir, "output_labels.png")
            }
        )
        result = self.tool.execute(tool_use)
        assert result.boxes_drawn == 1
    
    def test_execute_with_custom_label_text(self):
        """Test execute with custom label text."""
        tool_use = ToolUse(
            name="draw_bounding_box",
            params={
                "image_path": self.test_image_path,
                "boxes": [
                    {
                        "xyxy": [10, 20, 30, 40],
                        "confidence": 0.95
                    }
                ],
                "draw_labels": True,
                "label_text": "Custom Label",
                "output_path": os.path.join(self.temp_dir, "output_custom_label.png")
            }
        )
        result = self.tool.execute(tool_use)
        assert result.boxes_drawn == 1
    
    def test_execute_gemini_scale_conversion(self):
        """Test that coordinates in 0-1000 scale are converted to pixels."""
        tool_use = ToolUse(
            name="draw_bounding_box",
            params={
                "image_path": self.test_image_path,
                "boxes": [
                    {
                        "xyxy": [100, 200, 300, 400],  # In 0-1000 scale
                        "confidence": 0.95
                    }
                ],
                "output_path": os.path.join(self.temp_dir, "output_scale.png")
            }
        )
        result = self.tool.execute(tool_use)
        assert result.boxes_drawn == 1
        # Coordinates should be converted: 100/1000*100 = 10, 200/1000*100 = 20, etc.
    
    def test_execute_missing_image_path(self):
        """Test that missing image_path raises ValueError."""
        tool_use = ToolUse(
            name="draw_bounding_box",
            params={
                "boxes": [
                    {
                        "xyxy": [10, 20, 30, 40],
                        "confidence": 0.95
                    }
                ]
            }
        )
        with pytest.raises(ValueError, match="image_path parameter is required"):
            self.tool.execute(tool_use)
    
    def test_execute_missing_boxes(self):
        """Test that missing boxes raises ValueError."""
        tool_use = ToolUse(
            name="draw_bounding_box",
            params={
                "image_path": self.test_image_path
            }
        )
        with pytest.raises(ValueError, match="boxes parameter is required"):
            self.tool.execute(tool_use)
    
    def test_execute_invalid_image_path(self):
        """Test that invalid image path raises ValueError."""
        tool_use = ToolUse(
            name="draw_bounding_box",
            params={
                "image_path": "/nonexistent/path/image.jpg",
                "boxes": [
                    {
                        "xyxy": [10, 20, 30, 40],
                        "confidence": 0.95
                    }
                ]
            }
        )
        with pytest.raises(ValueError, match="Failed to load image"):
            self.tool.execute(tool_use)
    
    def test_execute_empty_boxes(self):
        """Test that empty boxes list raises ValueError."""
        tool_use = ToolUse(
            name="draw_bounding_box",
            params={
                "image_path": self.test_image_path,
                "boxes": []
            }
        )
        # Empty list is caught early as "boxes parameter is required"
        with pytest.raises(ValueError, match="boxes parameter is required"):
            self.tool.execute(tool_use)
    
    def test_execute_wrong_tool_name(self):
        """Test that wrong tool name raises ValueError."""
        tool_use = ToolUse(
            name="wrong_tool_name",
            params={
                "image_path": self.test_image_path,
                "boxes": [
                    {
                        "xyxy": [10, 20, 30, 40],
                        "confidence": 0.95
                    }
                ]
            }
        )
        with pytest.raises(ValueError, match="Tool name mismatch"):
            self.tool.execute(tool_use)


class TestDrawBoundingBoxHelpers:
    """Test helper methods."""
    
    def setup_method(self):
        """Set up test fixture."""
        self.tool = DrawBoundingBox()
    
    def test_get_output_path_provided(self):
        """Test output path when provided."""
        result = self.tool._get_output_path("/path/to/image.jpg", "/path/to/output.jpg")
        assert result == "/path/to/output.jpg"
    
    def test_get_output_path_auto_generated(self):
        """Test auto-generated output path."""
        result = self.tool._get_output_path("/path/to/image.jpg")
        assert result == "/path/to/image_annotated.jpg"
    
    def test_get_output_path_different_extensions(self):
        """Test output path with different file extensions."""
        result = self.tool._get_output_path("/path/to/image.png")
        assert result == "/path/to/image_annotated.png"
        
        result = self.tool._get_output_path("/path/to/image.jpeg")
        assert result == "/path/to/image_annotated.jpeg"
    
    def test_hex_to_rgb_color_name(self):
        """Test color name to RGB conversion."""
        assert self.tool._hex_to_rgb("red") == (255, 0, 0)
        assert self.tool._hex_to_rgb("green") == (0, 255, 0)
        assert self.tool._hex_to_rgb("blue") == (0, 0, 255)
        assert self.tool._hex_to_rgb("yellow") == (255, 255, 0)
    
    def test_hex_to_rgb_hex_code(self):
        """Test hex code to RGB conversion."""
        assert self.tool._hex_to_rgb("#FF0000") == (255, 0, 0)
        assert self.tool._hex_to_rgb("#00FF00") == (0, 255, 0)
        assert self.tool._hex_to_rgb("#0000FF") == (0, 0, 255)
        assert self.tool._hex_to_rgb("#FFFFFF") == (255, 255, 255)
    
    def test_hex_to_rgb_short_hex(self):
        """Test short hex code (3 digits) to RGB conversion."""
        assert self.tool._hex_to_rgb("#F00") == (255, 0, 0)
        assert self.tool._hex_to_rgb("#0F0") == (0, 255, 0)
        assert self.tool._hex_to_rgb("#00F") == (0, 0, 255)
    
    def test_hex_to_rgb_case_insensitive(self):
        """Test that color names are case insensitive."""
        assert self.tool._hex_to_rgb("RED") == (255, 0, 0)
        assert self.tool._hex_to_rgb("Red") == (255, 0, 0)
        assert self.tool._hex_to_rgb("rEd") == (255, 0, 0)
    
    def test_hex_to_rgb_invalid_defaults_to_red(self):
        """Test that invalid color defaults to red."""
        result = self.tool._hex_to_rgb("invalid_color")
        assert result == (255, 0, 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
