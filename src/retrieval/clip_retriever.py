"""CLIP-based image retrieval using local model"""
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from typing import List, Dict, Optional
import numpy as np
import os

class CLIPRetriever:
    """CLIP-based image similarity retrieval"""
    
    def __init__(self, model_path: str, gpu_ids: List[int] = [4, 5]):
        """
        Initialize CLIP retriever with local model
        
        Args:
            model_path: Path to local CLIP model
            gpu_ids: GPU device IDs to use
        """
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
        self.device = torch.device('cuda:0')
        
        self.model = CLIPModel.from_pretrained(
            model_path,
            local_files_only=True
        ).to(self.device)
        
        self.processor = CLIPProcessor.from_pretrained(
            model_path,
            local_files_only=True
        )
        
        self.model.eval()
    
    def encode_image(self, image_path: str) -> np.ndarray:
        """
        Encode single image to feature vector
        
        Args:
            image_path: Path to image file
            
        Returns:
            Normalized feature vector
        """
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        
        return features.cpu().numpy()[0]
    
    def encode_images_batch(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode multiple images in batches
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for encoding
            
        Returns:
            Array of normalized feature vectors
        """
        all_features = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images = [Image.open(p).convert('RGB') for p in batch_paths]
            
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
            
            all_features.append(features.cpu().numpy())
        
        return np.vstack(all_features)
    
    def find_most_similar(
        self,
        query_image_path: str,
        candidate_cases: List[Dict],
        threshold: float = 0.7
    ) -> Optional[Dict]:
        """
        Find most similar successful case
        
        Args:
            query_image_path: Query image path
            candidate_cases: List of candidate successful cases
            threshold: Similarity threshold
            
        Returns:
            Most similar case or None if below threshold
        """
        if not candidate_cases:
            return None
        
        query_features = self.encode_image(query_image_path)
        
        max_similarity = -1
        best_case = None
        
        for case in candidate_cases:
            case_features = np.array(case['image_features'])
            similarity = np.dot(query_features, case_features)
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_case = case
        
        if max_similarity >= threshold:
            return best_case
        
        return None