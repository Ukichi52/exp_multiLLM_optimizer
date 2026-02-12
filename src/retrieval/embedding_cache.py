"""Embedding cache manager for precomputed features"""
import numpy as np
from typing import Dict, List
import json

class EmbeddingCache:
    """Manage precomputed CLIP embeddings"""
    
    def __init__(self):
        self.cache = {}
    
    def add(self, image_id: str, features: np.ndarray):
        """Add embedding to cache"""
        self.cache[image_id] = features
    
    def get(self, image_id: str) -> np.ndarray:
        """Get embedding from cache"""
        return self.cache.get(image_id)
    
    def has(self, image_id: str) -> bool:
        """Check if embedding exists"""
        return image_id in self.cache
    
    def save(self, filepath: str):
        """Save cache to file"""
        data = {k: v.tolist() for k, v in self.cache.items()}
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def load(self, filepath: str):
        """Load cache from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.cache = {k: np.array(v) for k, v in data.items()}