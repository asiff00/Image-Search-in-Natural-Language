from pathlib import Path
import logging
from sentence_transformers import SentenceTransformer
from app.models.indexing import IndexingManager
from app.utils.image_processor import ImageProcessor
from app.utils.search import create_faiss_index, load_faiss_index, retrieve_similar_images
import threading
import time
import torch

logger = logging.getLogger(__name__)

class AIPhotoGallery:
    def __init__(self):
        self.images_path = Path("images")
        self.index_path = Path("Index/vector.index")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer("clip-ViT-L-14", device=device)
        self.model = self.model.to(device)
        for param in self.model.parameters():
            param.data = param.data.contiguous()
        
        self.indexing_manager = IndexingManager()
        self.image_processor = ImageProcessor(self.images_path, self.model)
        
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.images_path.mkdir(parents=True, exist_ok=True)
        self._index_lock = threading.Lock()
        self._index_cache = None
        self._last_index_update = 0
        self._has_new_images = False

        if self.index_path.exists():
            self.indexing_manager.update_status(
                is_initialized=True,
                status="done"
            )

    def background_indexing(self) -> None:
        try:
            self.indexing_manager.update_status(
                status="indexing",
                is_indexing=True
            )

            all_images = set(str(p.resolve()) for p in self.image_processor.get_all_images())
            processed_images = set()
            
            if self.index_path.exists():
                _, existing_paths = load_faiss_index(self.index_path)
                processed_images = set(str(Path(p).resolve()) for p in existing_paths)
            
            new_images = all_images - processed_images
            
            if not new_images:
                print("No new images to index")
                self.indexing_manager.update_status(
                    status="done",
                    is_initialized=True
                )
                return

            print(f"Found {len(new_images)} new images to index")
            
            total_images = len(new_images)
            self.indexing_manager.update_status(
                total_images=total_images,
                indexing_type='incremental' if processed_images else 'full'
            )

            all_embeddings = []
            all_paths = []
            
            if processed_images:
                for img_path in processed_images:
                    try:
                        embedding = self.image_processor.process_image(img_path)
                        all_embeddings.append(embedding)
                        all_paths.append(str(Path(img_path).resolve()))
                    except Exception as e:
                        logger.error(f"Error processing existing image {img_path}: {e}")
                        print(f"Error processing existing image {img_path}: {e}")

            for i, img_path in enumerate(new_images):
                try:
                    print(f"Processing new image {i+1}/{total_images}: {img_path}")
                    embedding = self.image_processor.process_image(img_path)
                    all_embeddings.append(embedding)
                    all_paths.append(str(Path(img_path).resolve()))
                    
                    self.indexing_manager.update_status(processed_images=i + 1)

                except Exception as e:
                    logger.error(f"Error processing {img_path}: {e}")
                    print(f"Error processing {img_path}: {e}")

            if all_embeddings:
                with self._index_lock:
                    create_faiss_index(all_embeddings, all_paths, str(self.index_path))

            self._last_index_update = time.time()
            self._index_cache = None
            self.indexing_manager.update_status(
                status="done",
                is_initialized=True,
                new_images_count=0
            )

        except Exception as e:
            logger.error(f"Indexing error: {e}")
            print(f"Indexing error: {e}")
            self.indexing_manager.update_status(
                status="error",
                last_error=str(e)
            )
        finally:
            self.indexing_manager.update_status(is_indexing=False)

    def start_indexing(self) -> None:
        if self.indexing_manager.needs_indexing():
            self.indexing_manager.executor.submit(self.background_indexing)
            self._has_new_images = False

    def load_faiss_index(self):
        with self._index_lock:
            try:
                if self._index_cache is None or time.time() - self._last_index_update > 300:
                    self._index_cache = load_faiss_index(self.index_path)
                    self._last_index_update = time.time()
                return self._index_cache
            except Exception as e:
                print(f"Error loading index: {e}")
                if not self.index_path.exists() or not Path(str(self.index_path) + '.paths').exists():
                    print("Index files missing, starting indexing...")
                    self.start_indexing()
                return None, []

    def retrieve_similar_images(self, query, top_k=12):
        try:
            index_data = self.load_faiss_index()
            if index_data is None or index_data[1] == []:
                print("No valid index found, returning empty results")
                return query, []
            index, image_paths = index_data
            return retrieve_similar_images(query, self.model, index, image_paths, top_k)
        except Exception as e:
            print(f"Error retrieving similar images: {e}")
            return query, []

    def has_new_images(self):
        return self._has_new_images

    def mark_new_images(self):
        self._has_new_images = True
