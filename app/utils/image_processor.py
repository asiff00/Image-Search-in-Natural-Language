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
        self._processed_paths: Set[str] = set()
        self._load_existing_hashes()

    def _load_existing_hashes(self):
        self._processed_paths.clear()
        self._image_hashes.clear()
        
        index_paths_file = Path("Index/vector.index.paths")
        if index_paths_file.exists():
            print("\nLoading existing index paths...")
            with open(index_paths_file, 'r') as f:
                for line in f:
                    path = Path(line.strip())
                    if path.exists():
                        self._processed_paths.add(str(path.resolve()))
                        try:
                            with open(path, 'rb') as img_file:
                                file_hash = hashlib.md5(img_file.read()).hexdigest()
                                self._image_hashes.add(file_hash)
                        except Exception as e:
                            print(f"Error loading hash for indexed path {path}: {e}")
            print(f"Loaded {len(self._processed_paths)} paths from index")
        else:
            print("\nNo existing index paths file found")

        print("\nScanning base path for images...")
        for img_path in self.get_all_images():
            try:
                resolved_path = str(img_path.resolve())
                if resolved_path not in self._processed_paths:
                    with open(img_path, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                        if file_hash not in self._image_hashes:
                            self._image_hashes.add(file_hash)
            except Exception as e:
                print(f"Error loading hash for {img_path}: {e}")

        print(f"Total unique image hashes: {len(self._image_hashes)}")
        print(f"Total processed paths: {len(self._processed_paths)}\n")

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
        all_files = list(self.base_path.rglob('*'))
        print(f"Found {len(all_files)} total files in {self.base_path}")
        
        images = [
            p for p in all_files
            if p.suffix.lower() in self.SUPPORTED_FORMATS
            and not p.name.startswith('.')
        ]
        
        print(f"Of which {len(images)} are supported images: {self.SUPPORTED_FORMATS}")
        return images

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

    def is_valid_image(self, path: Path) -> bool:
        try:
            with Image.open(path) as img:
                img.verify()
            return True
        except Exception:
            return False

    def get_unprocessed_images(self) -> List[Path]:
        all_images = self.get_all_images()
        unprocessed = []
        
        print(f"\nChecking for unprocessed images:")
        print(f"Total images found: {len(all_images)}")
        print(f"Already processed paths: {len(self._processed_paths)}")
        
        for img in all_images:
            resolved_path = str(img.resolve())
            if resolved_path not in self._processed_paths:
                if self.is_valid_image(img):
                    unprocessed.append(img)
                else:
                    print(f"Skipping invalid image: {img}")
        
        print(f"Found {len(unprocessed)} unprocessed valid images\n")
        return unprocessed

    def mark_as_processed(self, image_path: Path) -> None:
        self._processed_paths.add(str(image_path.resolve()))
