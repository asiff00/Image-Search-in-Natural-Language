"""
Manages the indexing process for the image gallery.
Handles tracking of indexing status, processed images, and concurrent indexing operations.
"""
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Set
from pathlib import Path
import time

class IndexingManager:
    """
    Manages the indexing process for images in the gallery.
    Provides thread-safe operations for tracking indexing status and processed images.
    """
    
    def __init__(self):
        """Initialize the indexing manager with default status and thread-safe components."""
        self.status = {
            "is_indexing": False,
            "total_images": 0,
            "processed_images": 0,
            "status": "waiting",
            "last_error": None,
            "is_initialized": False,
            "new_images_count": 0,
            "indexing_type": None,
        }
        self._lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._index_path = Path("Index/vector.index")
        self._processed_images: Set[str] = set()
        self._last_index_time = 0

    def update_status(self, **kwargs) -> None:
        """
        Thread-safe update of the indexing status.
        
        Args:
            **kwargs: Key-value pairs to update in the status dictionary
        """
        with self._lock:
            self.status.update(kwargs)

    def get_status(self) -> Dict:
        """
        Get a thread-safe copy of the current indexing status.
        
        Returns:
            Dict: Current indexing status
        """
        with self._lock:
            return self.status.copy()

    def needs_indexing(self) -> bool:
        """
        Check if indexing is needed based on current state.
        
        Returns:
            bool: True if indexing is needed, False otherwise
        """
        with self._lock:
            if self.status["is_indexing"]:
                return False
            
            if not self._index_path.exists():
                self.status["indexing_type"] = "full"
                return True
            
            if not self.status["is_initialized"] and self.status["status"] == "waiting":
                self.status["indexing_type"] = "full"
                return True
            
            if self.status["new_images_count"] > 0:
                self.status["indexing_type"] = "incremental"
                return True
            
            return False

    def mark_image_processed(self, image_path: str) -> None:
        """
        Mark an image as processed in a thread-safe manner.
        
        Args:
            image_path (str): Path to the processed image
        """
        with self._lock:
            resolved_path = str(Path(image_path).resolve())
            self._processed_images.add(resolved_path)
            self._last_index_time = time.time()

    def is_image_processed(self, image_path: str) -> bool:
        """
        Check if an image has been processed.
        
        Args:
            image_path (str): Path to the image to check
            
        Returns:
            bool: True if image has been processed, False otherwise
        """
        with self._lock:
            resolved_path = str(Path(image_path).resolve())
            return resolved_path in self._processed_images

    def add_new_images(self, count: int) -> None:
        """
        Add count of new images to be processed and update status accordingly.
        
        Args:
            count (int): Number of new images to be processed
        """
        with self._lock:
            self.status["new_images_count"] += count
            if self.status["status"] == "done":
                self.status["status"] = "waiting"
                self.status["is_indexing"] = False
                self.status["processed_images"] = 0
