import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Set
from pathlib import Path
import time

class IndexingManager:
    def __init__(self):
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
        with self._lock:
            self.status.update(kwargs)

    def get_status(self) -> Dict:
        with self._lock:
            return self.status.copy()

    def needs_indexing(self) -> bool:
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
        with self._lock:
            resolved_path = str(Path(image_path).resolve())
            self._processed_images.add(resolved_path)
            self._last_index_time = time.time()

    def is_image_processed(self, image_path: str) -> bool:
        with self._lock:
            resolved_path = str(Path(image_path).resolve())
            return resolved_path in self._processed_images

    def add_new_images(self, count: int) -> None:
        with self._lock:
            self.status["new_images_count"] += count
            if self.status["status"] == "done":
                self.status["status"] = "waiting"
                self.status["is_indexing"] = False
                self.status["processed_images"] = 0
