"""Analyze optimization results"""
import sys
sys.path.append('.')

import yaml
from src.storage.index_manager import IndexManager
from src.storage.mutation_logger import MutationLogger
from src.utils.metrics import calculate_asr
import statistics

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
    
    # Score distribution
    scores = [item['best_score'] for item in all_items]
    avg_score = statistics.mean(scores)
    median_score = statistics.median(scores)
    
    print(f"\nScore Distribution:")
    print(f"  Average score: {avg_score:.3f}")
    print(f"  Median score: {median_score:.3f}")
    print(f"  Max score: {max(scores):.3f}")
    print(f"  Min score: {min(scores):.3f}")
    
    # Iteration stats
    iterations = [item['total_iterations'] for item in all_items]
    avg_iter = statistics.mean(iterations)
    
    successful_items = [item for item in all_items if item['AS'] == 1]
    if successful_items:
        successful_iters = [item['total_iterations'] for item in successful_items]
        avg_successful_iter = statistics.mean(successful_iters)
    else:
        avg_successful_iter = 0
    
    print(f"\nIteration Statistics:")
    print(f"  Average iterations (all): {avg_iter:.2f}")
    print(f"  Average iterations (successful): {avg_successful_iter:.2f}")
    print(f"  Max iterations: {max(iterations)}")
    print(f"  Min iterations: {min(iterations)}")
    
    # Success by iteration
    success_by_iter = {}
    for item in successful_items:
        iter_count = item['total_iterations']
        success_by_iter[iter_count] = success_by_iter.get(iter_count, 0) + 1
    
    print(f"\nSuccess Distribution by Iteration:")
    for iter_num in sorted(success_by_iter.keys()):
        print(f"  Iteration {iter_num}: {success_by_iter[iter_num]} items")
    
    # Score improvement analysis
    print(f"\nScore Improvement Analysis:")
    total_improvements = []
    
    for item in all_items:
        history = mutation_logger.get_history(item['image_id'])
        if len(history) > 1:
            improvements = []
            for record in history[1:]:  # Skip iteration 0
                if 'score_improvement' in record:
                    improvements.append(record['score_improvement'])
            
            if improvements:
                total_improvements.extend(improvements)
    
    if total_improvements:
        avg_improvement = statistics.mean(total_improvements)
        positive_improvements = [x for x in total_improvements if x > 0]
        negative_improvements = [x for x in total_improvements if x < 0]
        
        print(f"  Total mutations: {len(total_improvements)}")
        print(f"  Average improvement: {avg_improvement:+.4f}")
        print(f"  Positive improvements: {len(positive_improvements)} ({len(positive_improvements)/len(total_improvements):.1%})")
        print(f"  Negative improvements: {len(negative_improvements)} ({len(negative_improvements)/len(total_improvements):.1%})")
        print(f"  No change: {len(total_improvements) - len(positive_improvements) - len(negative_improvements)}")
    
    # Strategy effectiveness
    print(f"\nStrategy Effectiveness:")
    strategy_stats = {}
    
    for item in all_items:
        history = mutation_logger.get_history(item['image_id'])
        for record in history:
            strategy = record.get('strategy', 'unknown')
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    'count': 0,
                    'improvements': [],
                    'successes': 0
                }
            
            strategy_stats[strategy]['count'] += 1
            
            if 'score_improvement' in record:
                strategy_stats[strategy]['improvements'].append(record['score_improvement'])
            
            # Check if this mutation led to success
            if record.get('gap_to_threshold', float('inf')) <= 0:
                strategy_stats[strategy]['successes'] += 1
    
    for strategy, stats in sorted(strategy_stats.items()):
        avg_imp = statistics.mean(stats['improvements']) if stats['improvements'] else 0
        success_rate = stats['successes'] / stats['count'] if stats['count'] > 0 else 0
        
        print(f"  {strategy}:")
        print(f"    Count: {stats['count']}")
        print(f"    Avg improvement: {avg_imp:+.4f}")
        print(f"    Success rate: {success_rate:.1%}")

if __name__ == '__main__':
    main()
