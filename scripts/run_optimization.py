"""Main optimization script"""
import sys
sys.path.append('.')

import yaml
import time
from src.models.target_model import TargetModel
from src.models.judge_model import JudgeModel
from src.models.mutator_model import MutatorModel
from src.models.analyzer_model import AnalyzerModel
from src.retrieval.clip_retriever import CLIPRetriever
from src.storage.index_manager import IndexManager
from src.storage.mutation_logger import MutationLogger
from src.storage.success_db import SuccessDB
from src.optimization.optimizer import ItemOptimizer
from src.optimization.convergence import ConvergenceDetector
from src.utils.metrics import calculate_asr

def main():
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize models
    print("Initializing models...")
    target_model = TargetModel(config['models']['target'])
    judge_model = JudgeModel(config['models']['judge'])
    mutator_model = MutatorModel(config['models']['mutator'])
    analyzer_model = AnalyzerModel(config['models']['analyzer'])
    
    # Initialize CLIP
    print("Loading CLIP model...")
    retriever = CLIPRetriever(
        model_path=config['retrieval']['clip_model_path'],
        gpu_ids=config['retrieval']['gpu_ids']
    )
    
    # Initialize storage
    index_mgr = IndexManager(config['paths']['index'])
    mutation_logger = MutationLogger(config['paths']['mutations'])
    success_db = SuccessDB(config['paths']['success_db'])
    
    # Initialize optimizer
    optimizer = ItemOptimizer(
        target_model=target_model,
        judge_model=judge_model,
        mutator_model=mutator_model,
        analyzer_model=analyzer_model,
        retriever=retriever,
        success_db=success_db,
        threshold=config['optimization']['jailbreak_threshold']
    )
    
    # Initialize convergence detector
    convergence = ConvergenceDetector(
        window=config['optimization']['convergence_window'],
        epsilon=config['optimization']['convergence_epsilon']
    )
    
    # Main loop
    max_iters = config['optimization']['max_global_iterations']
    start_time = time.time()
    
    for global_iter in range(max_iters):
        iter_start = time.time()
        
        print(f"\n{'='*60}")
        print(f"Global Iteration {global_iter + 1}/{max_iters}")
        print(f"{'='*60}")
        
        # Get pending items
        pending_items = index_mgr.get_pending_items()
        
        if not pending_items:
            print("All items successful, terminating early")
            break
        
        print(f"Optimizing {len(pending_items)} pending items...")
        
        # Track score improvements this iteration
        total_improvement = 0.0
        positive_improvements = 0
        
        # Optimize each item
        for idx, item in enumerate(pending_items):
            print(f"\n  [{idx+1}/{len(pending_items)}] {item['image_id']}")
            print(f"    Current best: {item['best_score']:.3f}")
            
            updated_item, mutation_entry = optimizer.optimize_step(item)
            
            if mutation_entry:
                # Update index
                index_mgr.update_entry(
                    image_id=item['image_id'],
                    updates=updated_item
                )
                
                # Log mutation with all score metrics
                mutation_logger.log_mutation(
                    image_id=item['image_id'],
                    iteration=mutation_entry['iteration'],
                    query=mutation_entry['query'],
                    response=mutation_entry['response'],
                    score=mutation_entry['score'],
                    strategy=mutation_entry['strategy'],
                    previous_score=mutation_entry.get('previous_score'),
                    score_improvement=mutation_entry.get('score_improvement'),
                    gap_to_threshold=mutation_entry.get('gap_to_threshold'),
                    failure_analysis=mutation_entry.get('failure_analysis')
                )
                
                # Display progress
                improvement = mutation_entry.get('score_improvement', 0.0)
                total_improvement += improvement
                if improvement > 0:
                    positive_improvements += 1
                
                improvement_sign = "+" if improvement >= 0 else ""
                print(f"    Iter {mutation_entry['iteration']}: "
                      f"Score {mutation_entry['score']:.3f} "
                      f"({improvement_sign}{improvement:.3f}), "
                      f"Gap {mutation_entry['gap_to_threshold']:.3f}, "
                      f"AS={updated_item['AS']}")
                
                # Show failure analysis preview if available
                if mutation_entry.get('failure_analysis'):
                    analysis_preview = mutation_entry['failure_analysis'][:80]
                    print(f"    Analysis: {analysis_preview}...")
        
        # Calculate ASR
        all_items = index_mgr.load_all()
        current_asr = calculate_asr(all_items)
        
        # Iteration statistics
        iter_elapsed = time.time() - iter_start
        avg_improvement = total_improvement / len(pending_items) if pending_items else 0
        improvement_rate = positive_improvements / len(pending_items) if pending_items else 0
        
        print(f"\n{'='*60}")
        print(f"Iteration {global_iter + 1} Summary:")
        print(f"  Current ASR: {current_asr:.2%}")
        print(f"  Avg score improvement: {avg_improvement:+.4f}")
        print(f"  Positive improvement rate: {improvement_rate:.1%}")
        print(f"  Iteration time: {iter_elapsed:.2f}s")
        print(f"{'='*60}")
        
        # Check convergence
        if convergence.check(current_asr):
            print(f"\nASR converged at {current_asr:.2%}, terminating")
            break
    
    # Final report
    total_elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("Optimization Complete")
    print(f"{'='*60}")
    
    all_items = index_mgr.load_all()
    final_asr = calculate_asr(all_items)
    
    # Calculate average iterations for successful items
    successful_items = [item for item in all_items if item['AS'] == 1]
    avg_iters = sum(item['total_iterations'] for item in successful_items) / len(successful_items) if successful_items else 0
    
    print(f"Final ASR: {final_asr:.2%}")
    print(f"Total iterations: {global_iter + 1}")
    print(f"Successful items: {len(successful_items)}/{len(all_items)}")
    print(f"Avg iterations to success: {avg_iters:.2f}")
    print(f"Total time: {total_elapsed:.2f}s")

if __name__ == '__main__':
    main()
