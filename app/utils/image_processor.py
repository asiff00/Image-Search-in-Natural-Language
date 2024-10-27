import hashlib
from pathlib import Path
from typing import List, Optional, Set
from datetime import datetime
from PIL import Image
from sentence_transformers import SentenceTransformer
import shutil

class ImageProcessor:
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

    def __init__(self, base_path: Path, model: SentenceTransformer):
        self.base_path = Path(base_path)
        self.model = model
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._image_hashes: Set[str] = set()
        self._load_existing_hashes()

    def _load_existing_hashes(self):
        for img_path in self.get_all_images():
            try:
                with open(img_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                    self._image_hashes.add(file_hash)
            except Exception as e:
                print(f"Error loading hash for {img_path}: {e}")

    def _get_file_hash(self, file) -> str:
        content = file.file.read()
        file.file.seek(0)
        return hashlib.md5(content).hexdigest()

    def is_duplicate(self, file) -> bool:
        file_hash = self._get_file_hash(file)
        return file_hash in self._image_hashes

    def save_uploaded_file(self, file) -> Path:
        if not file or not file.filename:
            raise ValueError("Invalid file")

        if self.is_duplicate(file):
            raise ValueError("This image has already been uploaded")

        clean_filename = Path(file.filename).name
        file_path = self.base_path / clean_filename
        
        counter = 1
        while file_path.exists():
            stem = Path(clean_filename).stem
            suffix = Path(clean_filename).suffix
            file_path = self.base_path / f"{stem}_{counter}{suffix}"
            counter += 1

        file_content = file.file.read()
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        file_hash = hashlib.md5(file_content).hexdigest()
        self._image_hashes.add(file_hash)

        return file_path

    def get_all_images(self) -> List[Path]:
        return [
            p for p in self.base_path.rglob('*')
            if p.suffix.lower() in self.SUPPORTED_FORMATS
            and not p.name.startswith('.')
        ]

    def process_image(self, image_path: Path):
        try:
            image = Image.open(image_path).convert("RGB")
            return self.model.encode(image)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            raise

    def get_relative_path(self, absolute_path: Path) -> Path:
        try:
            abs_path = Path(absolute_path).resolve()
            base = self.base_path.resolve()
            
            try:
                return abs_path.relative_to(base)
            except ValueError:
                new_path = self.base_path / abs_path.name
                if not new_path.exists():
                    shutil.copy2(abs_path, new_path)
                return new_path.relative_to(self.base_path)
        except Exception as e:
            print(f"Error converting path {absolute_path}: {e}")
            raise
