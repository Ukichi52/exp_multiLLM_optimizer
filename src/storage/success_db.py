"""Success cases database"""
import json
from pathlib import Path
from typing import List, Dict

class SuccessDB:
    """Manage successful jailbreak cases"""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.db_path.exists():
            self.db_path.touch()
    
    def add_case(
        self,
        image_id: str,
        image_path: str,
        query: str,
        score: float,
        image_features: List[float] = None
    ):
        """Add successful case"""
        case = {
            "image_id": image_id,
            "image_path": image_path,
            "successful_query": query,
            "final_score": score,
            "image_features": image_features
        }
        
        with open(self.db_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(case, ensure_ascii=False) + '\n')
    
    def load_all(self) -> List[Dict]:
        """Load all successful cases"""
        cases = []
        with open(self.db_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    cases.append(json.loads(line))
        return cases
    
    def update_embeddings(self, image_id: str, features: List[float]):
        """Update image features for a case"""
        cases = self.load_all()
        
        for case in cases:
            if case['image_id'] == image_id:
                case['image_features'] = features
                break
        
        with open(self.db_path, 'w', encoding='utf-8') as f:
            for case in cases:
                f.write(json.dumps(case, ensure_ascii=False) + '\n')