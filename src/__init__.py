"""Package initialization."""
from .config import settings
from .detector import get_detector
from .processor import VideoProcessor

__version__ = "1.0.0"
__all__ = ["settings", "get_detector", "VideoProcessor"]
