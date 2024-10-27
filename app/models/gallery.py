from pathlib import Path
import logging
from sentence_transformers import SentenceTransformer
from app.models.indexing import IndexingManager
from app.utils.image_processor import ImageProcessor
from app.utils.search import (
    create_faiss_index, 
    load_faiss_index, 
    retrieve_similar_images,
    add_to_faiss_index,
    cleanup_faiss_index
)
import threading
import time
import torch

logger = logging.getLogger(__name__)

class AIPhotoGallery:
    def __init__(self):
        self.images_path = Path("images")
        self.index_path = Path("Index/vector.index")
        
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.images_path.mkdir(parents=True, exist_ok=True)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        self.model = SentenceTransformer("clip-ViT-L-14", device=device)
        self.model = self.model.to(device)
        for param in self.model.parameters():
            param.data = param.data.contiguous()
        
        self.indexing_manager = IndexingManager()
        self.image_processor = ImageProcessor(self.images_path, self.model)
        self._index_lock = threading.Lock()
        self._index_cache = None
        self._last_index_update = 0
        self._has_new_images = False

        self._initialize_index()

    def _initialize_index(self):
        print("\n=== Initializing Gallery ===")
        
        unprocessed_images = self.image_processor.get_unprocessed_images()
        print(f"Found {len(unprocessed_images)} unprocessed images")
        
        if self.index_path.exists():
            print("Found existing index, cleaning up...")
            cleanup_faiss_index(self.index_path)
            if not unprocessed_images:
                print("No new images to process")
                self.indexing_manager.update_status(
                    is_initialized=True,
                    status="done"
                )
                return
        
        if unprocessed_images:
            print(f"Starting initial indexing for {len(unprocessed_images)} images...")
            self.indexing_manager.add_new_images(len(unprocessed_images))
            self.mark_new_images()
            print("Running immediate indexing...")
            self.background_indexing()
        else:
            print("No images to index. Waiting for uploads...")
            self.indexing_manager.update_status(
                is_initialized=True,
                status="done"
            )
        
        print("=== Initialization Complete ===\n")

    def background_indexing(self) -> None:
        try:
            print("\n=== Starting Indexing Process ===")
            self.indexing_manager.update_status(
                status="indexing",
                is_indexing=True
            )

            new_images = self.image_processor.get_unprocessed_images()
            
            if not new_images:
                print("No new images to index")
                self.indexing_manager.update_status(
                    status="done",
                    is_initialized=True
                )
                return

            print(f"Processing {len(new_images)} new images...")
            
            total_images = len(new_images)
            self.indexing_manager.update_status(
                total_images=total_images,
                indexing_type='incremental' if self.index_path.exists() else 'full'
            )

            new_embeddings = []
            new_paths = []
            
            for i, img_path in enumerate(new_images, 1):
                try:
                    print(f"Processing image {i}/{total_images}: {img_path}")
                    embedding = self.image_processor.process_image(img_path)
                    new_embeddings.append(embedding)
                    new_paths.append(str(img_path.resolve()))
                    self.image_processor.mark_as_processed(img_path)
                    
                    self.indexing_manager.update_status(processed_images=i)

                except Exception as e:
                    logger.error(f"Error processing {img_path}: {e}")
                    print(f"Error processing {img_path}: {e}")

            if new_embeddings:
                with self._index_lock:
                    if not self.index_path.exists():
                        print("Creating new index...")
                        create_faiss_index(new_embeddings, new_paths, str(self.index_path))
                    else:
                        print("Adding to existing index...")
                        add_to_faiss_index(self.index_path, new_embeddings, new_paths)

            self._last_index_update = time.time()
            self._index_cache = None
            self.indexing_manager.update_status(
                status="done",
                is_initialized=True,
                new_images_count=0
            )
            print("=== Indexing Complete ===\n")

        except Exception as e:
            logger.error(f"Indexing error: {e}")
            print(f"Indexing error: {e}")
            self.indexing_manager.update_status(
                status="error",
                last_error=str(e)
            )
        finally:
            self.indexing_manager.update_status(is_indexing=False)

    def start_indexing(self, force_immediate: bool = False) -> None:
        if self.indexing_manager.needs_indexing():
            if force_immediate:
                print("Running immediate indexing...")
                self.background_indexing()
            else:
                print("Submitting indexing to executor...")
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
