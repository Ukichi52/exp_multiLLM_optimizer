"""Logger for mutation history"""
import json
from pathlib import Path

class MutationLogger:
    """Log mutation history for each item"""
    
    def __init__(self, mutations_dir: str):
        self.mutations_dir = Path(mutations_dir)
        self.mutations_dir.mkdir(parents=True, exist_ok=True)
    
    def log_mutation(
        self,
        image_id: str,
        iteration: int,
        query: str,
        response: str,
        score: float,
        strategy: str
    ):
        """Log a single mutation"""
        mutation_file = self.mutations_dir / f"{image_id}_mutations.jsonl"
        
        entry = {
            "iteration": iteration,
            "query": query,
            "response": response,
            "score": score,
            "strategy": strategy
        }
        
        with open(mutation_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    def get_history(self, image_id: str) -> list:
        """Get complete history for an item"""
        mutation_file = self.mutations_dir / f"{image_id}_mutations.jsonl"
        
        if not mutation_file.exists():
            return []
        
        history = []
        with open(mutation_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    history.append(json.loads(line))
        
        return history