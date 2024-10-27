import os
import numpy as np
import faiss
from PIL import Image
from typing import Tuple, List, Union
from pathlib import Path

def create_faiss_index(embeddings: List[np.ndarray], image_paths: List[str], index_path: Union[str, Path]) -> None:
    index_path = Path(index_path)
    dimension = len(embeddings[0])
    
    base_index = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIDMap(base_index)
    
    vectors = np.array(embeddings).astype(np.float32)
    faiss.normalize_L2(vectors)
    
    ids = np.arange(len(embeddings))
    index.add_with_ids(vectors, ids)
    
    faiss.write_index(index, str(index_path))
    
    paths_file = Path(str(index_path) + '.paths')
    with paths_file.open('w') as f:
        for img_path in image_paths:
            f.write(f"{Path(img_path).resolve().as_posix()}\n")

def load_faiss_index(index_path: Union[str, Path]) -> Tuple[faiss.Index, List[str]]:
    index_path = Path(index_path)
    
    try:
        index = faiss.read_index(str(index_path))
    except Exception as e:
        print(f"Error reading index file: {e}")
        return None, []
    
    try:
        paths_file = Path(str(index_path) + '.paths')
        with paths_file.open('r') as f:
            image_paths = [line.strip() for line in f]
        return index, image_paths
    except Exception as e:
        print(f"Error reading paths file: {e}")
        return index, []

def retrieve_similar_images(query: Union[str, Image.Image], model, index: faiss.Index, 
                          image_paths: List[str], top_k: int = 3) -> Tuple[Union[str, Image.Image], List[str]]:
    try:
        if isinstance(query, str):
            if query.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                query = Image.open(query)
            query_features = model.encode(query)
        else:
            query = Image.open(query) if isinstance(query, str) else query
            query_features = model.encode(query)
        
        query_features = query_features.astype(np.float32).reshape(1, -1)
        distances, indices = index.search(query_features, top_k)
        if len(indices) == 0 or len(indices[0]) == 0:
            return query, []
        retrieved_images = [image_paths[int(idx)] for idx in indices[0] if 0 <= int(idx) < len(image_paths)]
        return query, retrieved_images
    except faiss.FaissException as e:
        print(f"FAISS error in retrieve_similar_images: {str(e)}")
        return None, []
    except Exception as e:
        print(f"Error in retrieve_similar_images: {str(e)}")
        return None, []
