from typing import List
from dataclasses import dataclass
from .bounding_box import BoundingBox


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
    
    def to_dict(self) -> dict:
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
