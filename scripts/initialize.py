"""Initialize dataset and precompute embeddings"""
import sys
sys.path.append('.')

import yaml
from pathlib import Path
from src.models.target_model import TargetModel
from src.models.judge_model import JudgeModel
from src.storage.index_manager import IndexManager
from src.storage.mutation_logger import MutationLogger
from src.retrieval.clip_retriever import CLIPRetriever
from src.utils.file_io import load_jsonl

def main():
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    target_model = TargetModel(config['models']['target'])
    judge_model = JudgeModel(config['models']['judge'])
    index_mgr = IndexManager(config['paths']['index'])
    mutation_logger = MutationLogger(config['paths']['mutations'])
    
    # Initialize CLIP
    retriever = CLIPRetriever(
        model_path=config['retrieval']['clip_model_path'],
        gpu_ids=config['retrieval']['gpu_ids']
    )
    
    # Load initial queries
    init_queries = load_jsonl(config['paths']['init_query'])
    
    print(f"Initializing {len(init_queries)} items...")
    
    for idx, item in enumerate(init_queries):
        image_id = item['image_id']
        query = item['query']
        image_path = str(Path(config['paths']['images']) / image_id)
        
        print(f"[{idx+1}/{len(init_queries)}] Processing {image_id}")
        
        # Test initial query
        response = target_model.call(image_path, query)
        score = judge_model.score(response)
        
        # Precompute embedding
        image_features = retriever.encode_image(image_path)
        
        # Create index entry
        index_entry = {
            "image_id": image_id,
            "image_path": image_path,
            "original_query": query,
            "AS": 1 if score > config['optimization']['jailbreak_threshold'] else 0,
            "best_score": score,
            "total_iterations": 0,
            "final_query": query,
            "last_response": response,
            "mutation_history_file": f"mutations/{image_id}_mutations.jsonl"
        }
        
        index_mgr.add_entry(index_entry)
        
        # Log initial state
        mutation_logger.log_mutation(
            image_id=image_id,
            iteration=0,
            query=query,
            response=response,
            score=score,
            strategy="original"
        )
        
        print(f"  Score: {score:.3f}, AS: {index_entry['AS']}")
    
    print("\nInitialization complete!")

if __name__ == '__main__':
    main()