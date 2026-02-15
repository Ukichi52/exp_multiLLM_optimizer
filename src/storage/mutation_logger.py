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
        strategy: str,
        previous_score: float = None,
        score_improvement: float = None,
        gap_to_threshold: float = None,
        failure_analysis: str = None
    ):
        """Log a single mutation with score metrics"""
        mutation_file = self.mutations_dir / f"{image_id}_mutations.jsonl"
        
        entry = {
            "iteration": iteration,
            "query": query,
            "response": response,
            "score": score,
            "strategy": strategy
        }
        
        # Add score metrics if available
        if previous_score is not None:
            entry["previous_score"] = previous_score
        if score_improvement is not None:
            entry["score_improvement"] = score_improvement
        if gap_to_threshold is not None:
            entry["gap_to_threshold"] = gap_to_threshold
        
        # Add failure analysis if available
        if failure_analysis is not None:
            entry["failure_analysis"] = failure_analysis
        
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
