from typing import Any, Dict
from dataclasses import dataclass


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
