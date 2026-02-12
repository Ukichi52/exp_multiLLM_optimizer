"""Analyze optimization results"""
import sys
sys.path.append('.')

import yaml
from src.storage.index_manager import IndexManager
from src.storage.mutation_logger import MutationLogger
from src.utils.metrics import calculate_asr

def main():
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    index_mgr = IndexManager(config['paths']['index'])
    mutation_logger = MutationLogger(config['paths']['mutations'])
    
    all_items = index_mgr.load_all()
    
    print(f"\n{'='*60}")
    print("Optimization Results Analysis")
    print(f"{'='*60}")
    
    # Overall stats
    total = len(all_items)
    successful = sum(1 for item in all_items if item['AS'] == 1)
    asr = calculate_asr(all_items)
    
    print(f"\nOverall Statistics:")
    print(f"  Total items: {total}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {total - successful}")
    print(f"  ASR: {asr:.2%}")
    
    # Iteration stats
    iterations = [item['total_iterations'] for item in all_items]
    avg_iter = sum(iterations) / len(iterations) if iterations else 0
    
    print(f"\nIteration Statistics:")
    print(f"  Average iterations: {avg_iter:.2f}")
    print(f"  Max iterations: {max(iterations) if iterations else 0}")
    print(f"  Min iterations: {min(iterations) if iterations else 0}")
    
    # Score distribution
    scores = [item['best_score'] for item in all_items]
    avg_score = sum(scores) / len(scores) if scores else 0
    
    print(f"\nScore Distribution:")
    print(f"  Average best score: {avg_score:.3f}")
    print(f"  Max score: {max(scores) if scores else 0:.3f}")
    print(f"  Min score: {min(scores) if scores else 0:.3f}")
    
    # Success by iteration
    success_by_iter = {}
    for item in all_items:
        if item['AS'] == 1:
            iter_count = item['total_iterations']
            success_by_iter[iter_count] = success_by_iter.get(iter_count, 0) + 1
    
    print(f"\nSuccess Distribution by Iteration:")
    for iter_num in sorted(success_by_iter.keys()):
        print(f"  Iteration {iter_num}: {success_by_iter[iter_num]} items")

if __name__ == '__main__':
    main()