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
import time

def main():
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    print("Initializing models...")
    target_model = TargetModel(config['models']['target'])
    judge_model = JudgeModel(config['models']['judge'])
    index_mgr = IndexManager(config['paths']['index'])
    mutation_logger = MutationLogger(config['paths']['mutations'])
    
    # Initialize CLIP
    print("Loading CLIP model...")
    retriever = CLIPRetriever(
        model_path=config['retrieval']['clip_model_path'],
        gpu_ids=config['retrieval']['gpu_ids']
    )
    
    # Load initial queries
    init_queries = load_jsonl(config['paths']['init_query'])
    
    threshold = config['optimization']['jailbreak_threshold']
    
    print(f"\nInitializing {len(init_queries)} items...")
    print(f"Success threshold: {threshold:.3f}")
    start_time = time.time()
    
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
        
        # Determine initial AS
        is_successful = 1 if score > threshold else 0
        gap_to_threshold = threshold - score
        
        # Create index entry
        index_entry = {
            "image_id": image_id,
            "image_path": image_path,
            "original_query": query,
            "AS": is_successful,
            "best_score": score,
            "total_iterations": 0,
            "final_query": query,
            "last_response": response,
            "mutation_history_file": f"mutations/{image_id}_mutations.jsonl"
        }
        
        index_mgr.add_entry(index_entry)
        
        # Log initial state with score metrics
        mutation_logger.log_mutation(
            image_id=image_id,
            iteration=0,
            query=query,
            response=response,
            score=score,
            strategy="original",
            gap_to_threshold=gap_to_threshold if not is_successful else 0.0
        )
        
        status = "SUCCESS" if is_successful else "PENDING"
        print(f"  Score: {score:.3f}, Gap: {gap_to_threshold:.3f}, Status: {status}")
    
    elapsed = time.time() - start_time
    
    # Summary statistics
    all_items = index_mgr.load_all()
    initial_asr = sum(1 for item in all_items if item['AS'] == 1) / len(all_items)
    
    print(f"\n{'='*60}")
    print("Initialization Complete")
    print(f"{'='*60}")
    print(f"Total items: {len(all_items)}")
    print(f"Initial ASR: {initial_asr:.2%}")
    print(f"Elapsed time: {elapsed:.2f}s")

if __name__ == '__main__':
    main()
