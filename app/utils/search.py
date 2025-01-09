"""
Utility functions for managing FAISS-based similarity search index.
Provides functionality for creating, loading, and searching image embeddings.
"""
import os
import numpy as np
import faiss
from PIL import Image
from typing import Tuple, List, Union
from pathlib import Path

def create_faiss_index(embeddings: List[np.ndarray], image_paths: List[str], index_path: Union[str, Path]) -> None:
    """
    Create a new FAISS index from image embeddings.
    
    Args:
        embeddings (List[np.ndarray]): List of image embeddings
        image_paths (List[str]): List of corresponding image paths
        index_path (Union[str, Path]): Path to save the FAISS index
    """
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
    """
    Load a FAISS index and its associated image paths from disk.
    
    Args:
        index_path (Union[str, Path]): Path to the FAISS index file
        
    Returns:
        Tuple[faiss.Index, List[str]]: Loaded index and list of image paths
    """
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
    """
    Find images similar to a query using the FAISS index.
    
    Args:
        query (Union[str, Image.Image]): Query image or text
        model: Model for generating embeddings
        index (faiss.Index): FAISS index for similarity search
        image_paths (List[str]): List of indexed image paths
        top_k (int): Number of similar images to retrieve
        
    Returns:
        Tuple[Union[str, Image.Image], List[str]]: Query and list of similar image paths
    """
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

def cleanup_faiss_index(index_path: Union[str, Path]) -> None:
    """
    Clean up duplicate entries in the FAISS index.
    
    Args:
        index_path (Union[str, Path]): Path to the FAISS index file
    """
    index_path = Path(index_path)
    paths_file = Path(str(index_path) + '.paths')
    
    if not paths_file.exists():
        return
        
    # Read all paths and their corresponding indices
    paths_with_indices = []
    unique_paths = set()
    duplicate_indices = set()
    
    with paths_file.open('r') as f:
        for idx, line in enumerate(f):
            resolved_path = Path(line.strip()).resolve().as_posix()
            if resolved_path in unique_paths:
                duplicate_indices.add(idx)
            else:
                unique_paths.add(resolved_path)
                paths_with_indices.append((idx, resolved_path))
    
    if not duplicate_indices:
        return
        
    index = faiss.read_index(str(index_path))
    
    dimension = index.d
    base_index = faiss.IndexFlatIP(dimension)
    new_index = faiss.IndexIDMap(base_index)
    
    all_vectors = []
    for i in range(index.ntotal):
        if i not in duplicate_indices:
            vector = index.reconstruct(i)
            all_vectors.append(vector)
    
    vectors = np.array(all_vectors).astype(np.float32)
    faiss.normalize_L2(vectors)
    
    # Add vectors to new index
    ids = np.arange(len(all_vectors))
    new_index.add_with_ids(vectors, ids)
    
    faiss.write_index(new_index, str(index_path))
    
    with paths_file.open('w') as f:
        for _, path in paths_with_indices:
            if path in unique_paths:
                f.write(f"{path}\n")

def add_to_faiss_index(index_path: Union[str, Path], new_embeddings: List[np.ndarray], 
                       new_image_paths: List[str]) -> None:
    """
    Add new embeddings to an existing FAISS index.
    
    Args:
        index_path (Union[str, Path]): Path to the FAISS index file
        new_embeddings (List[np.ndarray]): List of new image embeddings to add
        new_image_paths (List[str]): List of corresponding image paths
    """
    index_path = Path(index_path)
    
    cleanup_faiss_index(index_path)
    
    index = faiss.read_index(str(index_path))
    
    paths_file = Path(str(index_path) + '.paths')
    existing_paths = set()
    if paths_file.exists():
        with paths_file.open('r') as f:
            for line in f:
                existing_paths.add(Path(line.strip()).resolve().as_posix())
    
    filtered_embeddings = []
    filtered_paths = []
    for emb, path in zip(new_embeddings, new_image_paths):
        resolved_path = Path(path).resolve().as_posix()
        if resolved_path not in existing_paths:
            filtered_embeddings.append(emb)
            filtered_paths.append(resolved_path)
            existing_paths.add(resolved_path)
    
    if not filtered_embeddings:
        print("No new unique images to add to index")
        return
    
    vectors = np.array(filtered_embeddings).astype(np.float32)
    faiss.normalize_L2(vectors)
    
    start_id = index.ntotal
    ids = np.arange(start_id, start_id + len(filtered_embeddings))
    
    index.add_with_ids(vectors, ids)
    
    faiss.write_index(index, str(index_path))
    
    mode = 'a' if paths_file.exists() else 'w'
    with paths_file.open(mode) as f:
        for img_path in filtered_paths:
            f.write(f"{img_path}\n")
