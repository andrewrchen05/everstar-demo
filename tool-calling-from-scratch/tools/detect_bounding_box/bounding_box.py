from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class BoundingBox:
    """Represents a single detected bounding box."""
    confidence: float
    xyxy: List[float]  # [x1, y1, x2, y2] in normalized coordinates (0.0 to 1.0)
    
    def __post_init__(self):
        """Validate bounding box data."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        if len(self.xyxy) != 4:
            raise ValueError(f"xyxy must contain exactly 4 coordinates, got {len(self.xyxy)}")
        # Validate normalized coordinates
        for coord in self.xyxy:
            if not (0.0 <= coord <= 1.0):
                raise ValueError(f"Normalized coordinates must be between 0.0 and 1.0, got {coord}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "confidence": self.confidence,
            "xyxy": self.xyxy
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"BoundingBox(confidence={self.confidence:.2f}, xyxy={self.xyxy})"
